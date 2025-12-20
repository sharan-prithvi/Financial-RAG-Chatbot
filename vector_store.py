from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os

class vectorStore:
    def __init__(self, persist_directory='chroma_db', collection_name='FinancialDocs'):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = OllamaEmbeddings(model='mistral', base_url='http://localhost:11434')
        print('Initialized Ollama Embeddings and Vector Store settings.')

    def create_vector_store(self, chunks):
        print("Creating vector store...")
        vector_store = Chroma.from_documents(
            documents= chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        vector_store.persist()
        print("Vector store created and persisted.")
        return vector_store

    def load_vector_store(self):
        
        if not os.path.exists(self.persist_directory):
            raise ValueError(f"No existing vector store found at {self.persist_directory}.")
        
        print("Loading existing vector store...")
        vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        print("Vector store loaded.")
            
        return vector_store
    
    def similarity_search(self,vector_store, query, k=4):
        results = vector_store.similarity_search(query, k=k)
        print(f"Retrieved {len(results)} similar documents for the query.")
        return results