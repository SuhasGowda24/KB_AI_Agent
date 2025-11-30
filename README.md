📚 AI Knowledge Base Agent (RAG System)

A fast, lightweight, and accurate document-based AI assistant powered by Jina Embeddings, ChromaDB, FastAPI, Streamlit, and Groq Llama-3.

🚀 Overview

The AI Knowledge Base Agent is a Retrieval-Augmented Generation (RAG) system that allows you to:

✔ Upload PDFs, DOCX, and TXT files</br>
✔ Extract text (including OCR for scanned PDFs)</br>
✔ Generate embeddings (Jina AI)</br>
✔ Store vectors locally using ChromaDB</br>
✔ Query the document knowledge base</br>
✔ Get accurate, context-aware answers using Groq Llama-3.1-8B-Instant</br>
✔ Chat through a beautiful ChatGPT-style Streamlit UI</br>

This tool makes it extremely easy to build your own ChatGPT for documents — locally and for free.

🧠 Key Features</br>
🔍 Document Upload & Ingestion

Handles PDF, DOCX, TXT

Automatic OCR using Tesseract + Poppler

Smart text chunking (LangChain)

Embedding generation using Jina AI v2 Base EN

🗃 Vector Storage

Local and fast using ChromaDB

Ability to isolate knowledge by project name

🤖 LLM Querying

Context retrieval using RAG pipeline

Response generation using Groq Llama-3.1-8B-Instant

Avoids hallucinations by grounding answers in real context

🖥 Frontend

Clean ChatGPT-style UI

Streamlit-based

Chat bubbles (user + bot)

Typing animation

Source citations

Multi-project support

⚡ Fast & Free

Fully local embeddings

Free Groq API model

Zero hosting cost

🛠 Tech Stack
Backend

FastAPI

LangChain (RAG pipeline)

Jina AI Embeddings

ChromaDB local persistence

Tesseract OCR

Poppler (PDF parsing)

Groq Llama-3.1-8B-Instant (LLM)

Frontend

Streamlit

Custom HTML/CSS for ChatGPT-style chat interface

📦 Installation
1. Create & Activate Virtual Environment
  python -m venv venv
  venv\Scripts\activate       # Windows

3. Install Requirements
  pip install -r requirements.txt

4. Install Poppler + Tesseract
Windows:
1) Poppler: extract to: C:\Users\<You>\Downloads\poppler\bin

2) Tesseract:
Install to: C:\Program Files\Tesseract-OCR\tesseract.exe
Add both to PATH.

5. Create .env file inside: </br>
GROQ_API_KEY=your_groq_key</br>
JINA_API_KEY=your_jina_key</br>
TESSERACT_PATH=C:/Program Files/Tesseract-OCR/tesseract.exe</br>
POPPLER_PATH=C:/Users/<You>/Downloads/poppler/bin</br>

🚀 Run the Project
1. Start Backend
cd backend
uvicorn app:app --reload --port 8000

2. Start Frontend
cd frontend
streamlit run app_streamlit.py

🧪 API Endpoints
Upload Documents
POST /upload

Form-data:
files[]
project

Ask Questions
POST /query

Form-data:
project
question
