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

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get context from all stored documents
    context = " ".join(st.session_state.document_texts)
    
    # Create chat prompt
    chat_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"

    # Call DeepSeek-R1 API
    API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1"
    headers = {
        "Authorization": f"Bearer {st.secrets['HUGGINGFACE_TOKEN']}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": chat_prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95
        }
    }
    
    try:
        with st.spinner('Thinking...'):
            response = requests.post(API_URL, headers=headers, json=payload)
            st.write(f"Debug - API Response: {response.text}")  # Debug line
            
            if response.status_code == 200:
                response_data = response.json()
                # Check if response is a list
                if isinstance(response_data, list):
                    answer = response_data[0].get('generated_text', '')
                else:
                    answer = response_data.get('generated_text', '')
                
                if answer:
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                else:
                    st.error("Received empty response from API")
            else:
                st.error(f"API returned status code: {response.status_code}")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Response details for debugging:")
        st.write(response.text if 'response' in locals() else "No response received")

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
