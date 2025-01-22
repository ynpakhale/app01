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
    system_prompt = """You are a helpful AI assistant that answers questions based on the provided document context. 
    Keep your answers focused on the information in the context. If the answer cannot be found in the context, say so."""
    
    formatted_prompt = f"""### System:
{system_prompt}

### Context:
{context}

### Human: {prompt}

### Assistant:"""

    # Call DeepSeek-R1 API
    API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1"
    headers = {
        "Authorization": f"Bearer {st.secrets['HUGGINGFACE_TOKEN']}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 1000,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
            "return_full_text": False
        }
    }
    
    try:
        with st.spinner('Thinking...'):
            response = requests.post(API_URL, headers=headers, json=payload)
            response_data = response.json()
            
            # Extract the generated text from the response
            if isinstance(response_data, list) and len(response_data) > 0:
                answer = response_data[0].get('generated_text', '').strip()
            else:
                answer = "I apologize, but I couldn't process that request. Please try again."
            
            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try again or contact support if the problem persists.")

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
