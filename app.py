import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb

# Import functions from other modules
from config import EMBEDDING_MODEL, RELEVANCE_THRESHOLD
from pdf_processor import process_pdf
from vector_store import setup_vector_store, query_vector_store
from llm_handler import generate_prompt, stream_handler

# --- App Configuration ---
st.set_page_config(
    page_title="RAG Doc Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("RAG-Based Document Chatbot")
st.caption("Upload any PDF file and chat with it...")

# --- Caching and Initialization ---
@st.cache_resource
def get_embedding_model():
    print("Loading embedding model...")
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def get_chroma_client():
    print("Initializing ChromaDB Client...")
    return chromadb.Client()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store_collection" not in st.session_state:
    st.session_state.vector_store_collection = None

embedding_model = get_embedding_model()
chroma_client = get_chroma_client()

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader("Upload a file to explore", type="pdf")
    
    if uploaded_file:
        if "last_uploaded_id" not in st.session_state or st.session_state.last_uploaded_id != uploaded_file.file_id:
            st.session_state.last_uploaded_id = uploaded_file.file_id
            
            with st.spinner("Processing PDF..."):
                chunks, metadata = process_pdf(uploaded_file)
                if chunks:
                    st.session_state.vector_store_collection = setup_vector_store(
                        chunks, metadata, embedding_model, chroma_client
                    )
                    st.success("PDF processed successfully! You can now ask questions.")
                else:
                    st.session_state.vector_store_collection = None
                    st.error("Failed to process the PDF.")

# --- Main Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if st.session_state.vector_store_collection is None:
                st.warning("Please upload a PDF file first.")
            else:
                try:
                    # Retrieve context
                    context_docs, metadatas, distances = query_vector_store(
                        prompt, st.session_state.vector_store_collection, embedding_model
                    )
                    
                    # Check relevance and prepare context for LLM
                    if context_docs and distances[0] < RELEVANCE_THRESHOLD:
                        context_for_llm = "\n\n".join(context_docs)
                        sources = "\n".join([f" - Page {meta['page_number']}" for meta in metadatas])
                        context_for_llm += f"\n\nSources:\n{sources}"
                        has_context = True
                    else:
                        context_for_llm = "No relevant context found in the document."
                        has_context = False

                    # Generate final prompt and get response
                    final_prompt = generate_prompt(query=prompt, context=context_for_llm, has_context=has_context)
                    response_generator = stream_handler(final_prompt)
                    full_response = st.write_stream(response_generator)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"An error occurred: {e}")

# --- Save Chat History ---
with open("history.txt", "w", encoding="utf-8") as f:
    for message in st.session_state.messages:
        f.write(f'{message["role"].capitalize()}: {message["content"]}\n')