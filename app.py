import streamlit as st
import pymupdf

@st.cache_data
def extract_pages(file):
    try:
        doc = pymupdf.open(stream=file.read(), filetype="pdf")
        text_list = []
        for page in doc:
            text = page.get_text()
            text_list.append(text)
        
        return text_list
    except Exception as e:
        st.error(f"Error processing PDF file: {e}")
        return None


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

# --------------------- Sidebar for file upload -------------------

with st.sidebar:
    # Loading uploaded pdf file
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader("Upload the file that you want to explore", type="pdf")
    
    if uploaded_file is not None:
        if "last_uploaded_id" not in st.session_state or st.session_state.last_uploaded_id != uploaded_file.file_id:
            st.session_state.last_uploaded_id = uploaded_file.file_id
            
            with st.spinner("Processing PDF..."):
                st.session_state.pages = extract_pages(uploaded_file)
                st.success("PDF processed successfully! Now you can ask questions.")
    
    
# ---------------- Main Chat Interface ------------------

# Initializing chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# displaying previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# chat area
if prompt := st.chat_input("What would you like to ask from the uploaded document?"):
    
    # what user said
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # added user prompt to previous messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        # response = f"User said: {prompt}"
        full_text = "\n\n".join(st.session_state.pages)
        response = f"I have the document text available. The user asked: {prompt}.\n{len(full_text)} characters found in the pdf file."
        st.markdown(response)
    
    # add assistant response to previous messages
    st.session_state.messages.append({"role": "assistant", "content": response})
