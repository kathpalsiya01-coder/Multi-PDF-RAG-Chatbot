# 📄 Multi-PDF RAG Chatbot

An AI-powered chatbot that lets you upload multiple PDFs
and ask questions across all of them simultaneously.

Powered by LangChain, Groq, FAISS, and HuggingFace.

---

## 🎯 What It Does

- Upload multiple PDF documents at once
- Processes and indexes all PDFs into a vector store
- Ask questions in natural language
- Get accurate answers with source references
- Maintains full chat history

No manual reading. No ctrl+F.
Just upload and ask.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| LangChain | RAG pipeline orchestration |
| Groq (llama-3.3-70b) | LLM for answer generation |
| FAISS | Vector store for similarity search |
| HuggingFace Embeddings | Convert text chunks to vectors |
| pypdf | Extract text from PDF files |
| RecursiveCharacterTextSplitter | Split docs into chunks |
| Streamlit | Frontend UI |

---

## 🧠 How RAG Works Here

1. PDFs uploaded → text extracted via pypdf
2. Text split into chunks (1000 chars, 200 overlap)
3. Chunks converted to embeddings (all-MiniLM-L6-v2)
4. Embeddings stored in FAISS vector store
5. User asks question → question embedded
6. Similar chunks retrieved from FAISS
7. Chunks + question sent to Groq LLM
8. LLM answers based only on your documents

---

## 🚀 How To Run

### 1. Clone the repo
git clone https://github.com/kathpalsiya01-coder/Multi-PDF-RAG-Chatbot.git
cd Multi-PDF-RAG-Chatbot

### 2. Create virtual environment
conda create -n rag-chatbot python=3.12
conda activate rag-chatbot

### 3. Install dependencies
pip install -r requirements.txt

### 4. Add your API key
Create a .env file:
GROQ_API_KEY=your_groq_key_here

### 5. Run the app
streamlit run app.py

---

## 📁 Project Structure

multi-pdf-rag/
├── rag_pipeline.py    # Core RAG logic
├── app.py             # Streamlit UI
├── requirements.txt   # Dependencies
├── .env               # API key (not pushed)
└── .gitignore         # Ignores .env and venv

---

## 🔑 Get API Key

- Groq API → https://console.groq.com

---

## 🧠 Concepts Covered

- Retrieval Augmented Generation (RAG)
- Text chunking strategies
- Vector embeddings and similarity search
- FAISS vector store
- ConversationalRetrievalChain
- HuggingFace sentence transformers
- Streamlit session state for chat history

---

Built by Siya Kathpal | Undergraduate ML Engineer
