import streamlit as st
import pymupdf

def extract_pages(file):
    doc = pymupdf.open(stream=file.getvalue(), filetype="pdf")
    text_list = []
    for page in doc:
        text = page.get_text()
        text_list.append(text)
    
    return text_list

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

with st.sidebar:
    # Loading uploaded pdf file
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader("Upload the file that you want to explore", type="pdf")
    if uploaded_file and "file_uploaded" not in st.session_state:
        st.session_state.file_uploaded = True
        st.success("PDF file uploaded successfully!")
    
        pages = extract_pages(uploaded_file)
    
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
        response = f"User said: {prompt}"
        st.markdown(response)
    
    # add assistant response to previous messages
    st.session_state.messages.append({"role": "assistant", "content": response})
