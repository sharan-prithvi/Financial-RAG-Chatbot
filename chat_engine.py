from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import os

class chatEngine:
    def __init__(self, vector_store,model_name=None):
        self.vector_store = vector_store
        model_name = model_name or os.getenv('OLLAMA_MODEL', 'mistral')
        ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.llm = Ollama(model=model_name, base_url=ollama_base_url,temperature=0.1)
        print(f'Initialized Ollama LLM for Chat Engine with model: {model_name}.')

        self.prompt_template = """You are an expert financial analyst. Use the following context from financial documents to answer the question accurately.

        Context from documents:
        {context}
        
        Question: {question}
        Instructions:
        1. Answer based ONLY on the provided context
        2. If the context doesn't contain the answer, say "I don't have enough information to answer this question."
        3. Cite specific numbers, dates, or facts from the context
        4. Be concise but complete
        5. If discussing financial metrics, include the time period

        Answer:"""
        
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=self.prompt_template
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 4}
            ),
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )
        print('Chat Engine initialized with RetrievalQA chain.')

    def get_answer(self, question):
        print(f"Processing question: {question}")
        # `Chain.__call__` is deprecated; use `invoke` to avoid deprecation warnings
        result = self.qa_chain.invoke({"query": question})
        answer = result['result']
        sources = result['source_documents']
        print("Answer generated.")
        return {"answer": answer, "sources": sources}
    
    def format_sources(self, sources):
        formatted_sources = []
        for i, doc in enumerate(sources):
            source_info = {
                "chunk_number": i + 1,
                "content": doc.page_content[:200] +"...",
                "metadata": doc.metadata
            }
            formatted_sources.append(source_info)
        return formatted_sources
    