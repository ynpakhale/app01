import streamlit as st
import requests
import json
import PyPDF2
import io
import time
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
    
    # Function to make API call with retry
    def query_model(max_retries=5, initial_wait=10):
        for attempt in range(max_retries):
            try:
                response = requests.post(API_URL, headers=headers, json=payload)
                response_json = response.json()
                
                # Check if model is loading
                if response.status_code == 503 and "estimated_time" in str(response_json):
                    wait_time = min(initial_wait * (attempt + 1), 60)  # Progressive wait, max 60 seconds
                    with st.spinner(f'Model is warming up... Waiting {wait_time} seconds. Attempt {attempt + 1}/{max_retries}'):
                        time.sleep(wait_time)
                    continue
                
                return response, response_json
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(initial_wait)
        return None, None

    try:
        with st.spinner('Processing your question...'):
            response, response_data = query_model()
            
            if response and response.status_code == 200:
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
            elif response:
                st.error(f"API returned status code: {response.status_code}")
                st.write("Response:", response_data)
            else:
                st.error("Failed to get response after multiple retries")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

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
    - First query may take longer as the model warms up
    """)
