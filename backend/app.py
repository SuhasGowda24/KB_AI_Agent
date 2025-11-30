import os
from pathlib import Path
import shutil

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ingest_jina import ingest_documents, get_retriever
from retriever_local import answer_query


app = FastAPI(title="Free Knowledge Base Agent (Jina + Groq Llama3)")


# ------------------------------
# CORS
# ------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------
# Upload folder
# ------------------------------
UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------
# UPLOAD ENDPOINT
# ------------------------------
@app.post("/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    project: str = Form("default")
):
    project_dir = UPLOAD_DIR / project
    project_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for f in files:
        file_path = project_dir / f.filename

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)

        saved_files.append(str(file_path))

    # Ingest into vectorstore
    result = ingest_documents(saved_files, project)

    return {
        "status": "success",
        "project": project,
        "files_ingested": saved_files,
        "chunks": result["chunks"]
    }


# ------------------------------
# QUERY ENDPOINT
# ------------------------------
@app.post("/query")
async def query(
    project: str = Form("default"),
    question: str = Form(...)
):
    retriever = get_retriever(project)

    if retriever is None:
        return JSONResponse(
            {"error": "No docs found for this project. Upload first."},
            status_code=404
        )

    answer = answer_query(retriever, question)

    return {
        "project": project,
        "question": question,
        "answer": answer["answer"],
        "sources": answer["sources"]
    }
