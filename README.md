# RAG Document Q&A Application Using Groq and OpenAI

This project allows you to upload documents, build vector indexes, and perform semantic search using embeddings.

## Project Overview
The system uses the following workflow:
1. **Upload a Document** → File is saved to a temporary folder and stored in session state.
2. **Build Index** → The uploaded document is split into smaller chunks and indexed into a vector database using embeddings.
3. **Query** → The vector database can then be used to answer queries or perform semantic searches.

---

## Libraries and Uses
1. **streamlit** → Used to build the interactive web application where users can upload PDFs, enter queries, and view results in real-time.
2. **langchain** → Provides the framework for creating the conversational pipeline (document loaders, text splitters, embeddings, retrieval, and LLM chains).
3. **openai** → Gives access to OpenAI’s embedding models (text-embedding-ada-002) and LLMs for generating answers to user queries.
4. **faiss-cpu** → A vector database for storing and efficiently searching embeddings of PDF chunks, enabling fast similarity search.
5. **python-dotenv** → Loads API keys (OpenAI, Groq, etc.) from a .env file securely, keeping credentials out of the code.
6. **langchain-community** → Contains additional community-supported integrations like document loaders and utility tools not in the core LangChain package.
7. **langchain-openai** → Provides LangChain wrappers for OpenAI’s models (embeddings + chat models) to plug into the LangChain pipeline seamlessly.
8. **langchain-groq** → Lets you integrate Groq’s LLMs into the chatbot as an alternative to OpenAI, giving flexibility in model usage.
9. **pypdf** → A lightweight library to read and parse text from PDF documents before splitting into chunks.
10. **pymupdf** → Another powerful PDF parser that supports extracting not just text but also metadata and images, ensuring robust PDF handling.
11. **langchain-text-splitters** → Splits large chunks of PDF text into smaller, manageable segments so they can be embedded and searched efficiently.

### File Upload
When you upload a file in Streamlit, the system stores it locally and keeps track of it in `st.session_state.docs_info`.

- `uploaded_file` → The file object uploaded through Streamlit.  
- `file_path` → The path where the uploaded file is stored locally.  
- `st.session_state.docs_info` → A list storing metadata of uploaded files, including their **name** and **path**.

---

### Building the Index
When you click **Build Index**, the system takes the **last uploaded document** from `st.session_state.docs_info`, loads it, chunks it, and creates embeddings.

- `file_path` → Fetches the **path of the most recently uploaded file** from `st.session_state.docs_info`.  
- `load_documents(file_path)` → Loads the document from disk.  
- `chunk_documents(...)` → Splits the document into smaller overlapping chunks.  
- `build_embeddings_model(...)` → Creates an embedding model (can be OpenAI, HuggingFace, etc.).  
- `create_vector_store(...)` → Stores the chunks into a vector database for semantic search.

---

## Example of `docs_info`

If you upload two files:  
- `report.pdf` (stored at `uploaded_docs/report.pdf`)  
- `notes.txt` (stored at `uploaded_docs/notes.txt`)  


When you click **Build Index**, the system will use the **last uploaded file** (`notes.txt`) for indexing.

---

## Summary
- **`file_path`** is a local reference to the uploaded document stored inside `uploaded_docs/`.  
- It ensures that your app can reload and process the document even after upload.  
- `st.session_state.docs_info` acts as a document tracker, keeping both the file name and its stored location.

This modular design makes it easy to switch embeddings (OpenAI, HuggingFace, TF-IDF) and use different vector databases.
