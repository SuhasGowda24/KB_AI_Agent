# app_streamlit.py
"""
Single-file Streamlit app that:
- accepts file uploads (pdf / txt / docx)
- extracts text (pdf uses PyPDF2 fallback → OCR fallback)
- chunks text
- creates embeddings using sentence-transformers (free)
- stores vectors in local Chroma (persisted on disk)
- answers queries by retrieving top-k chunks and calling Groq LLM
"""

import os
import shutil
from pathlib import Path
import streamlit as st
import tempfile
import json
from typing import List

# document processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# file loaders
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

# OCR (optional, requires poppler & tesseract installed)
from pdf2image import convert_from_path
import pytesseract

# embeddings + vector DB
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# LLM (Groq)
from groq import Groq

# CONFIG: directories
BASE_DIR = Path.cwd()
UPLOAD_DIR = BASE_DIR / "uploads"
VECTOR_DIR = BASE_DIR / "vectorstore"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
VECTOR_DIR.mkdir(exist_ok=True, parents=True)

# Streamlit config
st.set_page_config(page_title="KB Agent (Single-file)", layout="wide", page_icon="📚")

# Load secrets from Streamlit (or environment)
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
# Optional: If you want to use Jina or other embedding service, set keys here. The default below uses sentence-transformers.
# JINA_API_KEY = st.secrets.get("JINA_API_KEY", os.getenv("JINA_API_KEY"))

# Initialize Groq client if available
client = None
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)

# Sentence-transformers model (free)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# Chromadb client (persist to disk)
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=str(VECTOR_DIR),
))

# ---------------------------
# Utilities: Loaders / OCR
# ---------------------------
def ocr_pdf(path: str) -> List[Document]:
    """Use pdf2image + pytesseract to extract text from scanned PDFs."""
    pages = convert_from_path(path)
    text = ""
    for p in pages:
        text += pytesseract.image_to_string(p) + "\n\n"
    return [Document(page_content=text, metadata={"source": str(path)})]

def load_pdf_text(path: str) -> List[Document]:
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    # If any page has text, return
    if any(d.page_content.strip() for d in docs):
        return docs
    # fallback to OCR
    return ocr_pdf(path)

def load_loader(path: Path) -> List[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf_text(str(path))
    if ext in [".txt", ".md"]:
        return TextLoader(str(path), encoding="utf-8").load()
    if ext == ".docx":
        return Docx2txtLoader(str(path)).load()
    raise Exception(f"Unsupported file type: {ext}")

# ---------------------------
# Embeddings & Chroma helpers
# ---------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    return embedder.encode(texts, show_progress_bar=False).tolist()

def ensure_collection(name: str):
    # return existing collection or create new
    try:
        return chroma_client.get_collection(name)
    except Exception:
        return chroma_client.create_collection(name)

def persist_client():
    chroma_client.persist()

# ---------------------------
# Ingest pipeline
# ---------------------------
def ingest_files(filepaths: List[str], project: str):
    all_docs: List[Document] = []
    for p in filepaths:
        all_docs.extend(load_loader(Path(p)))

    if not all_docs:
        raise Exception("No readable documents found.")

    # chunk documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    # clean and dedupe
    texts = [c.page_content.strip() for c in chunks if c.page_content.strip()]
    metadatas = [c.metadata for c in chunks if c.page_content.strip()]

    if not texts:
        raise Exception("No non-empty chunks extracted.")

    embeddings = embed_texts(texts)

    # store to chroma
    col = ensure_collection(project)
    # create ids
    ids = [f"{project}_{i}" for i in range(len(texts))]
    col.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    persist_client()
    return {"chunks": len(texts)}

# ---------------------------
# Retrieval + LLM
# ---------------------------
def query_project(project: str, question: str, k: int = 4):
    # check collection
    try:
        col = chroma_client.get_collection(project)
    except Exception:
        return {"answer": None, "sources": [], "error": "Project not ingested."}

    # embed question
    q_emb = embed_texts([question])[0]
    results = col.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas", "distances"])

    docs = []
    if results and "documents" in results and len(results["documents"])>0:
        retrieved_texts = results["documents"][0]
        retrieved_metas = results["metadatas"][0]
        for t, m in zip(retrieved_texts, retrieved_metas):
            docs.append(Document(page_content=t, metadata=m or {}))

    # build context
    context_text = "\n\n".join([d.page_content for d in docs])

    # Build final prompt
    prompt_template = f"""
You are an expert AI assistant. Use the context when relevant.
If the context does not contain the answer, answer from your general knowledge.
Do NOT invent citations.

Context:
{context_text}

Question:
{question}

Answer:
"""
    # If Groq configured -> call it, otherwise return retrieved context only
    answer_text = None
    if client:
        try:
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role":"user", "content": prompt_template}]
            )
            answer_text = res.choices[0].message.content
        except Exception as e:
            answer_text = f"(LLM call failed) {e}"
    else:
        # no LLM key configured — return assembled context as "answer candidate"
        answer_text = "No LLM API key configured. Showing retrieved context:\n\n" + (context_text or "No context found.")

    # dedupe sources
    seen = set()
    sources = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            snippet = d.page_content[:300].replace("\n", " ")
            sources.append({"source": src, "snippet": snippet})

    return {"answer": answer_text, "sources": sources}

# ---------------------------
# Streamlit UI
# ---------------------------
st.write("# 📚 Knowledge Base Agent (single-file)")

# Sidebar: upload + ingest
with st.sidebar:
    st.header("Upload & Ingest")
    project = st.text_input("Project name", value="default")
    uploaded = st.file_uploader("Upload files (pdf / txt / docx)", type=["pdf","txt","docx"], accept_multiple_files=True)
    ingest_btn = st.button("Upload & Ingest")

    st.markdown("---")
    st.write("Settings")
    k = st.number_input("Retrieval top-k", min_value=1, max_value=10, value=4)

    st.markdown("**Notes**")
    st.info("This app uses sentence-transformers (local) + Chroma (local folder). For OCR, ensure Poppler & Tesseract are installed and available in PATH.")

if ingest_btn:
    if not uploaded:
        st.error("Upload at least one file.")
    else:
        # save files to disk under uploads/project
        target_dir = UPLOAD_DIR / project
        target_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for f in uploaded:
            dest = target_dir / f.name
            with open(dest, "wb") as out:
                out.write(f.getbuffer())
            saved_paths.append(str(dest))

        try:
            with st.spinner("Ingesting..."):
                res = ingest_files(saved_paths, project)
            st.success(f"Ingested {res['chunks']} chunks for project '{project}'.")
        except Exception as e:
            st.exception(e)

# Main chat UI
st.subheader("Ask a question")
project_q = st.text_input("Project to query", value=project if project else "default")
question = st.text_input("Type your question here...")
ask = st.button("Ask")

# conversation state
if "history" not in st.session_state:
    st.session_state.history = []

if ask and question.strip():
    with st.spinner("Retrieving and generating answer..."):
        try:
            out = query_project(project_q, question, k=int(k))
        except Exception as e:
            st.error(f"Inference error: {e}")
            out = {"answer": None, "sources": [], "error": str(e)}

    st.session_state.history.append({"q": question, "a": out.get("answer")})
    # show latest
    st.markdown(f"**You:** {question}")
    st.markdown(f"**Agent:** {out.get('answer')}")
    if out.get("sources"):
        st.markdown("**Sources**")
        for s in out["sources"]:
            st.markdown(f"- `{s['source']}` — {s['snippet']}...")

# show history
if st.session_state.history:
    st.markdown("---")
    st.markdown("### Conversation history")
    for h in reversed(st.session_state.history[-10:]):
        st.markdown(f"**You:** {h['q']}")
        st.markdown(f"**Agent:** {h['a']}")

