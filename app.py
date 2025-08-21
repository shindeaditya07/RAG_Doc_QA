import os
import streamlit as st
import tempfile
from typing import List, Dict
from dotenv import load_dotenv  
from groq import Groq
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

model_name="Llama3-8b-8192"

# Initialize initial configurations
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "docs_info" not in st.session_state:
    st.session_state.docs_info = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Save uploaded file to a temporary location
def save_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmpf.write(uploaded_file.read())
    tmpf.flush()
    tmpf.close()
    return tmpf.name

# Upload and load the document
def load_documents(file_path: str) -> List[Dict]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
    else:
        st.error("Unsupported file type. Please upload a PDF file.")
        return []   
    documents = loader.load()
    return documents

# Create chunks of the document's content
def chunk_documents(documents: List[Dict], chunk_size: int, chunk_overlap: int) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    return chunks

# Build embeddings model using HuggingFaceEmbeddings
def build_openai_embeddings_model(model_name, api_key):
    return OpenAIEmbeddings(model=model_name, openai_api_key = api_key)

def build_hf_embeddings_model(model_name):
    return HuggingFaceEmbeddings(model_name=model_name)

# Create a FAISS vector store from the document chunks
def create_vector_store(chunks, embed_model) -> FAISS:
    vectorstore = FAISS.from_documents(chunks, embed_model)
    return vectorstore

# Context retrieval using Groq
def retrieve_context(query: str, vectorstore: FAISS, top_k: int = 5) -> List[Dict]:
    docs = vectorstore.similarity_search(query, k=top_k)
    return docs

# Prompt for Groq API 
def groq_prompt(context_blocks: List[str], question: str) -> str:
    context_text = "\n".join(context_blocks)  
    prompt = f"""You're a helpful assistant. Answer the question asked using only the context. 
    If the answer is not present in the context then say that you couldn't find it.
    Context:\n{context_text}\n
    Question: {question}\n
    Answer:"""
    return prompt

# Calling Groq API to get the answer
def call_groq_api(prompt: str) -> str:
    groq = Groq(api_key=groq_api_key)
    response = groq.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.1
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("Document Q&A with Groq")
st.caption("Upload a PDF document to ask questions about its content.")

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        st.session_state.docs_info.append({"name": uploaded_file.name, "path": file_path})
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    emb_model_type = st.selectbox(
        "Embedding Type",
        ["Huggingface", "OpenAI"],
        index = 0,
        help= "Embeddings"
    )
    if emb_model_type == "OpenAI":
        model_options = ["text-embedding-3-small", "text-embedding-3-large"]
    else: 
        model_options = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
    emb_model_name = st.selectbox(
        "Embedding Model",
        options = model_options,
        index = 0,
        help = "Embeddings"
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.5)
    st.caption("Adjust the temperature for response variability.")

# Building index
if st.button("Build Index"):
    if not st.session_state.docs_info:
        st.error("Please upload a document first.")
    else:
        file_path = st.session_state.docs_info[-1]["path"]
        documents = load_documents(file_path)
        if documents:
            chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)
            if emb_model_type == "OpenAI":
                model_options = ["text-embedding-3-small", "text-embedding-3-large"]
                embed_model = build_openai_embeddings_model(model_name=emb_model_name, api_key = os.getenv("OPENAI_API_KEY"))  
            else: 
                model_options = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
                embed_model = build_hf_embeddings_model(emb_model_name)
            vectorstore = create_vector_store(chunks, embed_model)
            st.session_state.vectorstore = vectorstore
            st.success("Index built successfully!")

# Streamlit Q&A portion
st.header("Ask your question")
st.subheader("Chat History")
number_of_chunks = st.slider("How many chunks?", 1,8,4)
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
question = st.chat_input("Enter the question here")
if question:
    if st.session_state.vectorstore is None:
        st.error("Upload a file to build index first")
        st.stop()
    try:
        retrieved = retrieve_context(question, st.session_state.vectorstore, top_k=number_of_chunks)  
        if not retrieved:
            st.warning("No relevant chunks")
        else:
            context_blocks = []
            for d in retrieved:
                page = d.metadata.get("page")
                header = f"[page {page}]" if page is not None else ""
                context_blocks.append(header + d.page_content)
            prompt = groq_prompt(context_blocks, question)   
            answer = call_groq_api(prompt)
            st.session_state.chat_history.append({"role":"user", "content": question})
            st.session_state.chat_history.append({"role":"assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(question)
                st.write(answer)
        st.session_state.user_question = ""
    except Exception as e:
        st.error(f"Q&A failed {e}")
