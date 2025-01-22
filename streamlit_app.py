import streamlit as st
import requests
import json
import PyPDF2
import io
from collections import defaultdict

# Streamlit interface setup
st.title("Document Chat Assistant")

# Initialize session state for storing document text
if 'document_texts' not in st.session_state:
    st.session_state.document_texts = []

# File uploader
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    # Process the uploaded PDF
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Store text in session state
    st.session_state.document_texts.append(text)
    st.success("Document processed and stored!")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Prepare context from all stored documents
    context = " ".join(st.session_state.document_texts)
    
    # Prepare prompt with context
    full_prompt = f"""Context: {context}\n\nQuestion: {prompt}\n\nAnswer based on the context provided:"""
    
    # Call DeepSeek API
    API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/deepseek-coder-33b-instruct"
    headers = {
        "Authorization": f"Bearer {st.secrets['HUGGINGFACE_TOKEN']}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": full_prompt,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        answer = response.json()['output']
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Sidebar with instructions
with st.sidebar:
    st.markdown("""
    ### How to use:
    1. Upload your PDF document
    2. Wait for processing confirmation
    3. Ask questions about your document
    4. The assistant will provide relevant answers
    
    ### Notes:
    - Supports PDF files only
    - Maximum file size: 200MB
    - Processing large files may take a few moments
    """)
