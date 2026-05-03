import streamlit as st
from rag_pipeline import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain
from dotenv import load_dotenv
import os

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process PDFs first!")
        return

    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response['chat_history']
    source_documents = response.get('source_documents', [])

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)
                if i == len(st.session_state.chat_history) - 1 and source_documents:
                    with st.expander("Sources"):
                        for j, doc in enumerate(source_documents):
                            source_name = doc.metadata.get('source', 'Unknown')
                            st.markdown(f"**Snippet {j+1} from {source_name}**:")
                            st.info(doc.page_content)


def main():
    load_dotenv()
    
    # Check if Groq API Key is set
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not found. Please set it in your .env file.")
        st.stop()

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")

    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload some PDFs first.")
            else:
                with st.spinner("Processing..."):
                    # 1. get pdf documents
                    raw_documents = get_pdf_text(pdf_docs)

                    # 2. get the text chunks
                    text_chunks = get_text_chunks(raw_documents)

                    # 3. create vector store
                    vectorstore = get_vector_store(text_chunks)

                    # 4. create conversation chain
                    st.session_state.conversation = get_conversational_chain(vectorstore)
                    
                    st.success("Documents processed successfully! You can now start chatting.")

if __name__ == '__main__':
    main()
