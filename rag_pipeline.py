import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain.schema import Document

def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF documents and wraps them in Document objects."""
    documents = []
    for pdf in pdf_docs:
        # Streamlit UploadedFile is a BytesIO object. We must seek to the beginning 
        # in case it was already read in a previous run.
        pdf.seek(0)
        
        # Use PdfReader from pypdf
        from pypdf import PdfReader
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        documents.append(Document(page_content=text, metadata={"source": pdf.name}))
    return documents

def get_text_chunks(documents):
    """Splits the documents into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_vector_store(document_chunks):
    """Creates a FAISS vector store from document chunks using HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents=document_chunks, embedding=embeddings)
    return vectorstore

def get_conversational_chain(vectorstore):
    """Initializes the ConversationalRetrievalChain with Groq."""
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile"
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 8, "fetch_k": 30}
        ),
        memory=memory,
        return_source_documents=True
    )
    
    return conversation_chain
