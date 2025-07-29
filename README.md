# 🧠 RAG-Based PDF Chatbot with Llama3 and Streamlit

A fully functional Retrieval-Augmented Generation (RAG) chatbot that lets you converse with your PDF documents. Powered by a local Large Language Model (LLM) via Ollama, your data stays private and secure.

## Built with a sleek Streamlit interface, this app makes it easy to upload PDFs, ask questions, and receive intelligent, context-aware answers sourced directly from your documents.

## 📸 Preview
(Insert a screenshot or demo GIF here)

## ✨ Features
1. Interactive Chat Interface
  - Simple and intuitive UI built with Streamlit.

2. PDF Document Processing
  - Upload any PDF and automatically extract, chunk, and process its text content.

3. Private & Local LLM
  - Uses a local LLM (e.g., Llama3.2 via Ollama) — no third-party APIs involved.

4. Retrieval-Augmented Generation (RAG)
  - Finds the most relevant passages from the PDF to answer your questions, reducing hallucinations and improving accuracy.

5. Source Citing
  - Responses include page numbers from the PDF used to generate the answer.
---
## 🛠️ Technical Stack
- Application Framework: Streamlit
- LLM Serving: Ollama with Llama3.2 (or other models as you wish)
- Text Processing: LangChain (for text splitting)
- Document Loading: PyMuPDF (previously Fitz)
- Embedding Model: Sentence-Transformers (all-MiniLM-L6-v2)
- Vector Database: ChromaDB
---
## 📁 Project Structure
```
/rag_pdf_chatbot
│
├── app.py              # Main Streamlit application entry point
├── llm_handler.py      # Functions for interacting with the LLM
├── pdf_processor.py    # Functions for processing the uploaded PDF
├── vector_store.py     # Functions for managing the ChromaDB vector store
├── config.py           # Configuration constants (model names, URLs, etc.)
├── requirements.txt    # Project dependencies
└── history.txt         # Stores the latest conversation history
```
---
## 🚀 Setup and Installation
1. Prerequisites:
- Python 3.8+
- Ollama installed and running with at least one model (I used Llama3.2 [https://ollama.com/library/llama3.2])

2. Install the LLM
- Pull the Llama3 model using:
    - (You can use a different model, but update the MODEL variable in config.py accordingly.)

3. Clone the Repository

4. Set Up a Virtual Environment & Install Dependencies

5. Run the Application. (Ensure Ollama is running in the background, then launch the app)

The application will open in your default web browser.
---
## 💬 How to Use
- Launch the application: run `streamlit run app.py` to run the app
- Use the sidebar to upload a PDF file. (Must upload a PDF to start the chat)
- Wait for the document to be processed. (App will take some time to process, please wait a little bit. Also PDF processing time depends of the size of the pdf.) `This app does not support images, it only supports simple text. You can modify pdf_processor.py file to make detailed extraction from pdf`.
- Type your question in the chat input box and press Enter.
- Receive a response with cited page numbers from your document. (If it can't find a suitable context from the PDF, it will try it's best to response according to the query. `The response depends on the LLM model used`)
