import os
import requests
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# OCR imports
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from uuid import uuid4

load_dotenv()

# Load Tesseract
TESSERACT_PATH = os.getenv("TESSERACT_PATH")
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_URL = "https://api.jina.ai/v1/embeddings"
CHROMA_DIR = "./vectorstore"


# ---------------------------
#  WRAPPER FOR JINA EMBEDDINGS
# ---------------------------
class JinaEmbedding:
    def embed_documents(self, texts):
        return jina_embed(texts)

    def embed_query(self, text):
        return jina_embed([text])[0]



# ---------------------------
#  RAW EMBEDDING CALL
# ---------------------------
def jina_embed(texts: list[str]):
    headers = {"Authorization": f"Bearer {JINA_API_KEY}"}
    payload = {
        "model": "jina-embeddings-v2-base-en",
        "input": texts
    }

    res = requests.post(JINA_URL, json=payload, headers=headers)
    data = res.json()

    if "data" not in data:
        print("Jina API error:", data)
        raise Exception(f"Jina API error: {data}")

    return [item["embedding"] for item in data["data"]]


# ---------------------------
#  DOCUMENT LOADERS
# ---------------------------
def ocr_pdf(path: str):
    """ Extract text from scanned PDFs using OCR """
    pages = convert_from_path(path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page) + "\n\n"
    return [Document(page_content=text, metadata={"source": path})]


def load_pdf_text(path: str):
    """ Try normal extraction first, fallback to OCR """
    loader = PyPDFLoader(path)
    docs = loader.load()

    if any(d.page_content.strip() for d in docs):
        return docs

    print("⚠ PDF contains no extractable text — using OCR")
    return ocr_pdf(path)


def load_loader(path: str):
    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        return load_pdf_text(path)
    if ext == ".txt":
        return TextLoader(path, encoding="utf-8").load()
    if ext == ".docx":
        return Docx2txtLoader(path).load()

    raise Exception(f"Unsupported file type: {ext}")


# ---------------------------
#  INGEST DOCUMENTS
# ---------------------------
def ingest_documents(filepaths: list, project: str):
    docs = []

    # Load docs
    for p in filepaths:
        docs.extend(load_loader(p))

    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    print("DEBUG: Extracted documents =", len(docs))
    print("DEBUG: Chunks =", len(chunks))

    lc_docs = [Document(page_content=c.page_content, metadata=c.metadata) for c in chunks]

    # Remove empty text
    texts = [d.page_content.strip() for d in lc_docs if d.page_content.strip()]

    if not texts:
        raise Exception("No readable text found. The file may be image-only.")

    # Embed chunks
    vectors = jina_embed(texts)

    # Prepare vectors to store
    embeddings_to_add = []
    documents_to_add = []
    metadatas_to_add = []
    ids_to_add = []

    vec_idx = 0
    for doc in lc_docs:
        text = doc.page_content.strip()
        if not text:
            continue
        embeddings_to_add.append(vectors[vec_idx])
        documents_to_add.append(text)
        metadatas_to_add.append(doc.metadata)
        ids_to_add.append(str(uuid4()))
        vec_idx += 1

    # Save to Chroma
    persist_path = Path(CHROMA_DIR) / project
    persist_path.mkdir(parents=True, exist_ok=True)

    vectordb = Chroma(
        collection_name=project,
        embedding_function=None,
        persist_directory=str(persist_path)
    )

    vectordb._collection.add(
        ids=ids_to_add,
        embeddings=embeddings_to_add,
        documents=documents_to_add,
        metadatas=metadatas_to_add,
    )

    vectordb.persist()

    return {"status": "ingested", "chunks": len(embeddings_to_add)}



def get_retriever(project: str):
    persist_path = Path(CHROMA_DIR) / project
    if not persist_path.exists():
        return None

    vectordb = Chroma(
        collection_name=project,
        embedding_function=JinaEmbedding(),  # Wrapper class
        persist_directory=str(persist_path)
    )

    return vectordb.as_retriever(search_kwargs={"k": 4})

