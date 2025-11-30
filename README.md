📚 AI Knowledge Base Agent (RAG System)

A fast, lightweight, and accurate document-based AI assistant powered by Jina Embeddings, ChromaDB, FastAPI, Streamlit, and Groq Llama-3.

## Overview
A Retrieval-Augmented Generation (RAG) Knowledge Base Agent. Upload files (PDF/DOCX/TXT), it ingests, chunks, stores embeddings in Chroma, and answers user queries using LLM + retrieved context.

## Tech stack
- Backend: FastAPI
- Ingestion & RAG: LangChain + OpenAI embeddings
- Vector DB (demo): Chroma (local). Swap to Pinecone for scale.
- Frontend: Streamlit

## Run locally
1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Backend
