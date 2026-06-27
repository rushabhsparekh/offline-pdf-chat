from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Optional 
from retrieve import get_indexed_docs, ensure_collection, ask, index_document, add_to_registry
from qdrant_client import QdrantClient
from pdf_to_json import get_or_create_chunks, build_converter, build_chunker
from pathlib import Path

app = FastAPI()

client = QdrantClient(path="qdrant_storage")
ensure_collection(client)
converter = build_converter()
chunker = build_chunker()


class QueryRequest(BaseModel):
    question: str
    source_filter: Optional[str] = None
    multi_sources: Optional[list[str]] = None
    
@app.get("/documents")
def list_documents():
    return get_indexed_docs()
    
@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    pdf_path = Path("pdfs") / file.filename
    with open(pdf_path, "wb") as f:
        f.write(await file.read())
    chunks = get_or_create_chunks(pdf_path, converter, chunker)
    points = index_document(pdf_path.stem, client)
    add_to_registry(file.filename, points)
    return get_indexed_docs()
    
    
@app.post("/query")
def query(request: QueryRequest):
    return ask(request.question, client, 
    source_filter=request.source_filter, 
    multi_sources=request.multi_sources
    )