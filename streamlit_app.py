import streamlit as st
import requests
import json
from chromadb import Client
from sentence_transformers import SentenceTransformer
import PyPDF2
import io

# Initialize ChromaDB
chroma_client = Client()
collection = chroma_client.create_collection(name="my_documents")

# Initialize the embedding model
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Streamlit interface setup
st.title("Document Chat Assistant")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    # Process the uploaded PDF
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Split text into chunks (simple approach)
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    
    # Generate embeddings and store in ChromaDB
    embeddings = embedder.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(chunks))]
    )
    st.success("Document processed and stored!")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Search relevant documents
    query_embedding = embedder.encode(prompt).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    
    context = " ".join(results['documents'][0])
    
    # Prepare prompt with context
    full_prompt = f"""Context: {context}\n\nQuestion: {prompt}\n\nAnswer based on the context provided:"""
    
    # Call DeepSeek API
    API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/deepseek-coder-33b-instruct"
    headers = {
        "Authorization": "Bearer YOUR_API_KEY",
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
