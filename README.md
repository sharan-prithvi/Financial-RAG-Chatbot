# Financial Document Chatbot

## Overview

This repository implements a local retrieval-augmented generation (RAG) system for answering questions about financial PDF documents. It ingests PDFs, chunks them, builds a vector store (Chroma) with embeddings from a local Ollama instance, and exposes a Streamlit chat UI for natural-language queries.

Key components:
- `app.py` — Streamlit frontend and orchestration.
- `docProcessor.py` — PDF loading and text chunking.
- `vector_store.py` — Vector store creation, persistence, loading, and similarity search using Chroma + Ollama embeddings.
- `chat_engine.py` — RetrievalQA chain using an Ollama LLM and the vector store.


**Repository layout**

- `app.py` — Streamlit UI, session management, and user flows (process documents, load existing vector store, chat).
- `docProcessor.py` — Document loader and text splitter.
- `vector_store.py` — Chroma vector store wrapper, uses `OllamaEmbeddings`.
- `chat_engine.py` — Builds `RetrievalQA` chain using `Ollama` LLM and the provided vector store.
- `data/Documents/` — Expected location for PDF files to be processed.
- `chroma_db/` — Persistence directory for Chroma (created after vector store persists).


**Important prerequisites**

- Python 3.10+ (3.11 recommended).
- Local Ollama server running and accessible at `http://localhost:11434` with the `mistral` model available for both embeddings and generation.
- Chroma-compatible setup (the `langchain_community` Chroma wrapper persists to `chroma_db`).
- Typical Python packages: `streamlit`, `langchain_community`, `langchain_text_splitters`, plus PDF parsing dependencies (e.g., `pypdf`, `pdfminer.six`) required by `PyPDFLoader`.

Recommended (example) pip install list (create a `requirements.txt` accordingly):

```bash
streamlit
langchain-community
langchain-text-splitters
pypdf
# plus any package required by Ollama client wrapper or your environment
```

Adjust package names/versions to your environment and the `langchain_community` variant you're using.

# Financial Document Chatbot — Simple Guide

This project is a local chatbot that answers questions about financial PDF documents you provide. It works by:

- Reading your PDF files.
- Splitting them into small text chunks.
- Turning those chunks into embeddings (vectors) using a local Ollama service.
- Storing embeddings in a local Chroma database for fast search.
- Running a Streamlit chat UI where you ask questions and the app returns answers with source snippets.

This README explains how to set it up and use it in plain, easy steps.

---

**Quick overview of important files**

- `app.py` — Streamlit frontend (UI) and the main app flow.
- `docProcessor.py` — Loads PDFs and splits them into chunks.
- `vector_store.py` — Creates, saves, and loads the Chroma vector store using Ollama embeddings.
- `chat_engine.py` — Connects the vector store to an LLM (Ollama) and runs the retrieval + answer flow.
- `data/Documents/` — Put your PDF files here.
- `chroma_db/` — Where Chroma stores the persisted vector database after processing.

---

Prerequisites

- Python 3.10 or newer (3.11 recommended).
- A local Ollama server running and reachable at `http://localhost:11434`. The example code uses the `mistral` model for embeddings and generation.
- Install required Python packages (example below).

Install Python packages (example)

```bash
pip install streamlit langchain-community langchain-text-splitters pypdf
```

Note: package names and exact versions may vary. If you have a `requirements.txt` you prefer, use that.

---

Quick start — run the app

1. Put your PDF files into `data/Documents/` (you can keep subfolders).
2. Start Ollama locally and make sure the `mistral` model is available.
3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. In the app sidebar choose:
- `Process New Documents` to read PDFs, chunk, embed, and persist a new vector store.
- Or `Load Existing Documents` if you already have a `chroma_db` directory.

5. After documents are processed/loaded, type questions in the chat input. Answers will include short source snippets and file metadata.

---

Simple usage tips

- If nothing happens when processing, check that `data/Documents` contains PDF files.
- If loading fails, make sure `chroma_db/` exists and is not corrupted.
- To clear chat history, use the `Clear Chat History` button in the sidebar.

---

Troubleshooting

- Ollama not reachable: confirm Ollama is running and the base URL is correct in `vector_store.py` and `chat_engine.py`.
- Chroma load error: inspect `chroma_db/` for missing files or permission issues.
- Missing source paths in results: `docProcessor` expects `DirectoryLoader` to attach `metadata['source']`; if not present, inspect `docProcessor.py`.

If you see exceptions in the Streamlit UI, run the app from your terminal to get the full traceback for debugging.

---

Recommended next steps (optional)

- Add a `requirements.txt` with pinned versions used in your environment.
- Add a Streamlit `file_uploader` to let users upload PDFs via the UI.
- Make `OLLAMA_URL`, `MODEL_NAME`, and `CHROMA_DIR` configurable via environment variables.
- Add basic per-file error handling and logging in `docProcessor.py`.

---

Want me to do one of these for you? I can:

- Create a `requirements.txt`.
- Add a Streamlit file uploader and process uploaded PDFs automatically.
- Fix the `embedding` vs `embedding_function` mismatch if your Chroma wrapper needs that.

Tell me which one you want next and I will implement it.
## How to run locally
