import streamlit as st
import pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import json
import requests

# JOINING_STR = "\n\n" + "-*"*50 + "\n\n"
URL = "http://localhost:11434/api/chat"

MODEL = "llama3.2:latest"

@st.cache_resource
def get_embedding_model():
    print("Loading embedding model...")
    return SentenceTransformer('all-MiniLM-L6-V2')

@st.cache_resource
def get_chroma_client():
    print("Initializing ChromaDB Client...")
    return chromadb.Client()

@st.cache_data
def process_pdf(file):
    try:
        doc = pymupdf.open(stream=file.read(), filetype="pdf")
        all_chunks = []
        all_metadata = []
        
        # splitter obj
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 150,
            length_function = len
        )
        
        for i, page in enumerate(doc):
            text = page.get_text()
            
            if text:
                page_chunk = text_splitter.split_text(text)
                
                for chunk in page_chunk:
                    all_chunks.append(chunk)
                    all_metadata.append({"page_number": i+1, "source": file.name})
        
        return all_chunks, all_metadata
        
    except Exception as e:
        st.error(f"Error processing PDF file: {e}")
        return None

# function to store embeddings into chromadb
def setup_vector_store(chunks, metadata, embedding_model, client):
    collection_name = "pdf_rag_collection"
    
    try:
        # if collection already exists, delete it
        if client.get_collection(name = collection_name):
            client.delete_collection(name=collection_name)
    except Exception as e:
        pass
    
    # create a brand new collection
    collection = client.get_or_create_collection(
        name = collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # embed the chunks
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    
    # create unique id for each chunk
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # store in chromadb
    collection.add(
        embeddings = embeddings,
        documents = chunks,
        metadatas = metadata,
        ids = ids
    )
    
    return collection

# function to get most simliar chunks
def query_vector_store(query, collection, embedding_model, top_n = 3):
    if collection is None:
        return None
    
    query_embedding = embedding_model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings = query_embedding,
        n_results = top_n,
        include = ['documents', 'metadatas', 'distances']
    )
    
    relevant_docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    return relevant_docs, metadatas, distances  

# function to create query to give to the llm
def generate_query(query, context, has_context):
    
    # if relevant docs are found from the pdf
    if has_context:
        prompt = f"""
        Based on the following context from a PDF document, please answer the user's question.
        Only use information from the context. If the context does not contain the answer, state that you cannot find the answer in the document.

        Context:
        ---
        {context}
        ---

        User Question: {query}
        """
        return prompt
    else:
        prompt = f"""
        The user asked a question, but there was no relevant information found in the user provided PDF document.
        Please provide a general, helpful answer to the user's question using the most silimar context from the pdf, even though they are not similar enough.
        
        NOTE: ONLY USE CONTEXT IF IT MAKES SENSE TO THE QUESTION, ELSE PROVIDE A GENERAL HELPFUL ANSWER. BUT NEVER PROVIDE AN ANSWER IF IT DOES NOT MAKE SENSE FROM THE GIVEN CONTEXT.

        Context:
        ---
        {context}
        ---

        User Question: {query}
        """
        return prompt

# function to get streams of responses
def get_streamed_reaponse(prompt):
    messages = st.session_state.messages
    
    json_data = {
        "model": MODEL,
        "messages": messages,
        "stream": True
    }
    
    # implement stream reponse from the llm
    in_think_block = False
    buffer = ""
    
    # return response as it's being generate by the llm
    try:
        response = requests.post(url=URL, json=json_data, stream=True)
        response.raise_for_status()
        
        # iterate over the response stream line by line
        for line in response.iter_lines():
            if line:
                try:
                    json_line = json.loads(line)
                    if "message" in json_line and "content" in json_line['message']:
                        token = json_line['message']['content']
                        buffer += token
                        
                        # process buffer text for think tags
                        while True:
                            if not in_think_block:
                                start_idx = buffer.find("<think>")
                                
                                if start_idx != -1:
                                    # from the first letter to the start of think tag as answer
                                    if start_idx > 0:
                                        yield "answer", buffer[:start_idx]
                                    buffer = buffer[start_idx + len("<think>"):]
                                    in_think_block = True   # Now, are inside the think block
                                else:
                                    # think tag is not found, the entire buffer text is answer
                                    yield "answer", buffer
                                    buffer = ""   # emptying buffer for next iteration
                                    break   # breaking out of processing buffer loop, as we didnt find any think tag
                            else:
                                end_index = buffer.find("</think>")
                                if end_index != -1:
                                    if end_index > 0:
                                        yield "think", buffer[:end_index]
                                    buffer = buffer[end_index + len("</think>"):]
                                    in_think_block = False
                                else:
                                    yield "think", buffer
                                    buffer = ""
                                    break
                                
                except Exception as e:
                    print(e)
                    break
        
    except Exception as e:
        print(e)


# -------------------- UI Configuration -------------------------

# Creating basic UI
# app config
st.set_page_config(
    page_title="RAG Doc Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("RAG based document chatbot")
st.caption("Upload any PDF file and chat with it...")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store_collection" not in st.session_state:
    st.session_state.vector_store_collection = None

# --------------------- Sidebar for file upload -------------------

# load model and client
embedding_model = get_embedding_model()
chroma_client = get_chroma_client()

with st.sidebar:
    # Loading uploaded pdf file
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader("Upload the file that you want to explore", type="pdf")
    
    if uploaded_file is not None:
        file_id = uploaded_file.file_id
        
        if "last_uploaded_id" not in st.session_state or st.session_state.last_uploaded_id != uploaded_file.file_id:
            st.session_state.last_uploaded_id = uploaded_file.file_id
            
            with st.spinner("Processing PDF..."):
                # st.session_state.pages = extract_pages(uploaded_file)
                chunks, metadata = process_pdf(uploaded_file)
                
                if chunks:
                    st.session_state.vector_store_collection = setup_vector_store(
                        chunks, metadata, embedding_model, chroma_client
                    )
                    
                    st.success("PDF processed successfully! Now you can ask questions.")
                else:
                    st.session_state.vector_store_collection = None
                    st.error("Failed to process PDF.")
    
    
# ---------------- Main Chat Interface ------------------
    
# displaying previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# chat area
if prompt := st.chat_input("What would you like to ask from the uploaded document?"):
    # added user prompt to previous messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # what user said
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # what assistant would say
    try:
        with st.chat_message("assistant"):
            with st.spinner("Loading response..."):
                if st.session_state.vector_store_collection is None:
                    st.warning("Please upload a PDF file first!")
                    reponse = "I need a pdf file first before I can answer."
                else:
                    # first, retrieve relevant chunks from the pdf
                    context_docs, metadatas, distances = query_vector_store(
                        prompt, st.session_state.vector_store_collection, embedding_model
                    )
                    
                    # second, decide if the content is relevant enough
                    RELEVANCE_THRESHOLD = 0.7
                    if context_docs and distances[0] < RELEVANCE_THRESHOLD:
                        context_for_llm = "\n\n".join(context_docs)
                        
                        # add page numbers for citation in the llm respone
                        sources = "\n".join([f" - Page {meta['page_number']}" for meta in metadatas])
                        context_for_llm += f"\n\nSources:\n{sources}"
                        has_context = True
                    else:
                        context_for_llm = "\n\n".join(context_docs)
                        has_context = False
                    
                    # third, generate prompt to give to llm
                    final_prompt = generate_query(query=prompt, context=context_for_llm, has_context=has_context)
                    
                    answer = ""
                    # finally, give prompt to llm and display streamed response
                    for part_type, token in get_streamed_reaponse(final_prompt):
                        if part_type == "think":
                            pass    # do nothing
                        else:
                            answer += token
                    
                    st.markdown(answer)
    
        # add assistant response to previous messages
        st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as e:
        st.error(f"Following issue was detected: {e}")