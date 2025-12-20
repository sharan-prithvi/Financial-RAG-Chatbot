import streamlit as st
from docProcessor import docProcessor
from vector_store import vectorStore
from chat_engine import chatEngine
import os
import time

st.set_page_config(page_title="Financial Document Chatbot", page_icon="💹",layout="wide")

st.markdown("""
            <style>
            .main-header{
                font-size:2.5rem;
                color:#1f77b4;
                text-align:center;
                margin-bottom:2rem;
            }
            .source-box{
                background-color:#f0f2f6;
                padding:1rem;
                border-radius:0.5rem;
                margin-bottom:0.5rem 0;
            }
            </style>
            """, unsafe_allow_html=True)

def initialize_state_session():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    if 'chat_engine' not in st.session_state:
        st.session_state.chat_engine = None
        
    if 'doc_processed' not in st.session_state:
        st.session_state.doc_processed = False

def process_documents():
    with st.spinner("Processing documents... This may take a few minutes."):
        try:
            processor = docProcessor()
            chunks = processor.process_documents()
            if not chunks:
                st.error("No documents found in the specified directory.")
                return False
            v_store_manager = vectorStore()
            v_store = v_store_manager.create_vector_store(chunks)
            
            chat_engine_instance = chatEngine(vector_store=v_store)
            
            st.session_state.vector_store = v_store
            st.session_state.chat_engine = chat_engine_instance
            st.session_state.doc_processed = True
            
            st.success(f"Processed {len(chunks)} from documents!")
            return True
        except Exception as e:
            st.error(f"An error occurred during document processing: {e}")
            return False

def load_existing_vector_store():
    with st.spinner("Loading existing vector store..."):
        try:
            v_store_manager = vectorStore()
            v_store = v_store_manager.load_vector_store()
            chat_engine_instance = chatEngine(vector_store=v_store)
            
            st.session_state.vector_store = v_store
            st.session_state.chat_engine = chat_engine_instance
            st.session_state.doc_processed = True
            
            return True
        except Exception as e:
            st.error(f"An internal server error occurred.")
            return False
        
def display_chat_message(message, role, sources = None):
    with st.chat_message(role):
        st.markdown(message)
        if sources and role == 'assistant':
            with st.expander("View Source Documents"):
                for i, source in enumerate(sources,1):
                    st.markdown(f"**Source {i}:**")
                    st.markdown(f"- **Content:** {source['content']}")
                    st.markdown(f"- **Metadata:** {source['metadata'].get('source','Unknown')}")
                    st.markdown("---")
                    
def main():
    st.markdown("<h1 class='main-header'>💹 Financial Document Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("""
    **How it works:**
    1. Upload PDF financial documents
    2. Documents are chunked and embedded
    3. Ask questions in natural language
    4. Get accurate answers with source citations
    """)
    
    initialize_state_session()
    
    with st.sidebar:
        st.header("📁 Document Management")
        
        if os.path.exists('chroma_db'):
            if st.button("Load Existing Documents"):
                if load_existing_vector_store():
                    st.success("Documents Loaded Successfully!")
                    st.rerun()
        
        if st.button("Process New Documents"):
            if process_documents():
                st.rerun()
                
        if st.session_state.doc_processed:
            st.success("📊 System Ready!")
            st.metric("Chat history", len(st.session_state.chat_history))
            
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared.")
            st.rerun()
            
    if not st.session_state.doc_processed:
        st.warning("⚠️ Please process or load documents to start chatting.")
        st.markdown("### Example Questions:")
        st.markdown("""
        - What was the revenue in Q2?
        - Summarize the key financial metrics
        - What were the main risk factors mentioned?
        - How did operating expenses change every year?
        """)
        return
    
    for message in st.session_state.chat_history:
        display_chat_message(message['content'], message['role'], message.get('sources'))
        
    if prompt := st.chat_input("Ask a question about your financial documents..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        display_chat_message(prompt, "user")
        
        with st.spinner("Analysing the documents..."):
            start_time = time.time()
            result= st.session_state.chat_engine.get_answer(prompt)
            elapsed = time.time() - start_time
            formatted_sources = st.session_state.chat_engine.format_sources(result['sources'])
            st.session_state.chat_history.append({"role": "assistant", "content": result['answer'], "sources": formatted_sources})
            display_chat_message(result['answer'], "assistant", formatted_sources)
            st.caption(f"Response time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
    