
# RAG Document Q&A Application Using Groq and OpenAI

This project allows you to upload documents, build vector indexes, and perform semantic search using embeddings.

---

## 🚀 **Project Overview**

The system uses the following workflow:

1. **Upload a Document** → File is saved to a temporary folder and stored in session state.
2. **Build Index** → The uploaded document is split into smaller chunks and indexed into a vector database using embeddings.
3. **Query** → The vector database can then be used to answer queries or perform semantic searches.
4. **Chat History** → All user questions and assistant answers are saved and displayed, so you can view the full conversation flow while interacting with your document.

---

## 📚 **Libraries and Their Uses**

- **streamlit** → Interactive web application where users can upload PDFs, enter queries, and view results in real-time.
- **langchain** → Framework for creating the conversational pipeline (document loaders, text splitters, embeddings, retrieval, and LLM chains).
- **openai** → Provides access to OpenAI’s embedding models (`text-embedding-ada-002`) and LLMs for generating answers to user queries.
- **faiss-cpu** → Vector database for storing and efficiently searching embeddings of PDF chunks, enabling fast similarity search.
- **python-dotenv** → Loads API keys (OpenAI, Groq, etc.) from a `.env` file securely, keeping credentials out of the code.
- **langchain-community** → Additional community-supported integrations like document loaders and utility tools.
- **langchain-openai** → LangChain wrappers for OpenAI’s models (embeddings + chat models) to integrate into the pipeline seamlessly.
- **langchain-groq** → Integrates Groq’s LLMs into the chatbot as an alternative to OpenAI.
- **pypdf** → Lightweight library to read and parse text from PDF documents.
- **pymupdf** → Advanced PDF parser that extracts text, metadata, and images.
- **langchain-text-splitters** → Splits large chunks of PDF text into smaller, manageable segments for embedding and searching.

---

## 📂 **File Upload Process**

When you upload a file in **Streamlit**, the system:

- Stores it locally.
- Keeps track of it in `st.session_state.docs_info`.

**Key Attributes:**

- `uploaded_file` → The file object uploaded through Streamlit.
- `file_path` → The path where the uploaded file is stored locally.
- `st.session_state.docs_info` → A list storing metadata of uploaded files, including their name and path.

---

## 🛠 **Building the Index**

When you click **Build Index**, the system takes the last uploaded document and:

1. Loads it using `load_documents(file_path)`.
2. Splits it into smaller overlapping chunks using `chunk_documents(...)`.
3. Creates embeddings using `build_embeddings_model(...)`.
4. Stores the chunks into a vector database using `create_vector_store(...)` for semantic search.

---

## 💬 **Chat History Feature**

The app includes a **chat interface** where both user queries and assistant responses are displayed in sequence:

- Each user input is tagged as **User**, and the model’s response as **Assistant**.
- History persists throughout the session, allowing you to scroll back and review earlier questions and answers.
- Provides a **continuous conversational way** of exploring your documents.

---

## 📄 **Example of `docs_info`**

If you upload two files:

- `report.pdf` → stored at `uploaded_docs/report.pdf`
- `notes.txt` → stored at `uploaded_docs/notes.txt`

When you click **Build Index**, the system will use the **last uploaded file (`notes.txt`)** for indexing.

---

## ✅ **Summary**

- **file_path** → Local reference to the uploaded document stored inside `uploaded_docs/`.
- **st.session_state.docs_info** → Tracks documents with file names and stored locations.
- **Chat History** → Stores all queries and answers, ensuring a smooth conversational experience.

---
