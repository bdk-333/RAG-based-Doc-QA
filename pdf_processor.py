import streamlit as st
import pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

@st.cache_data
def process_pdf(file):
    """
    Reads a PDF file, extracts text, and splits it into chunks.
    """
    try:
        doc = pymupdf.open(stream=file.read(), filetype="pdf")
        all_chunks = []
        all_metadata = []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        
        for i, page in enumerate(doc):
            text = page.get_text()
            if text:
                page_chunks = text_splitter.split_text(text)
                for chunk in page_chunks:
                    all_chunks.append(chunk)
                    all_metadata.append({"page_number": i + 1, "source": file.name})
        
        return all_chunks, all_metadata
        
    except Exception as e:
        st.error(f"Error processing PDF file: {e}")
        return None, None