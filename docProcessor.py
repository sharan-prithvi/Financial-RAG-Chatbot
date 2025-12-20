from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import os

class docProcessor:
    def __init__(self, data_path = 'data/Documents'):
        self.data_path = data_path

    def load_pdf(self):
        print("Loading PDF files from :", self.data_path)
        loader = DirectoryLoader(self.data_path, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
        
        documents = loader.load()
        print(f"Loaded {len(documents)} document pages.")
        return documents

    def chunk_documents(self, documents, chunk_size = 1000, chunk_overlap = 200):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks.")        
        return chunks
    
    def process_documents(self):
        documents = self.load_pdf()
        chunks = self.chunk_documents(documents)
        return chunks