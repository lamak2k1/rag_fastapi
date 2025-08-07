from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import logging

# LlamaIndex / Chroma
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# std-lib
import chromadb
import os

# ------------------------------------------------------------------#
#  App-level setup
# ------------------------------------------------------------------#
load_dotenv()                              # reads .env -> GOOGLE_API_KEY, etc.

# Configure logging (optional tweak as you like)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(title="RAG-API")

# ------------------------------------------------------------------#
#  Pydantic models
# ------------------------------------------------------------------#
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None  # future-proofing

class ChatResponse(BaseModel):
    answer: str

# ------------------------------------------------------------------#
#  Start-up: load the persisted Chroma collection once
# ------------------------------------------------------------------#
@app.on_event("startup")
async def load_rag():
    global qa_engine

    persist_dir = "index"  # your Chroma files live here
    client = chromadb.PersistentClient(path=persist_dir)

    collections = client.list_collections()
    if not collections:
        raise RuntimeError(f"No Chroma collections found in {persist_dir}")

    collection = collections[0]  # reuse existing
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    # Use the **same** embedding model/dimensions the collection was built with
    embed_model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001",
        task_type="retrieval_document",
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_ctx,
        embed_model=embed_model,
    )
    qa_engine = index.as_query_engine()
    logging.info("✅ Chroma-based RAG index loaded")

# ------------------------------------------------------------------#
#  Health route for Render & k8s probes
# ------------------------------------------------------------------#
@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/ping")
async def ping():
    return {"message": "pong"}

# ------------------------------------------------------------------#
#  Main chat endpoint — now with graceful error handling
# ------------------------------------------------------------------#
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        answer = qa_engine.query(req.query)
        return ChatResponse(answer=str(answer))

    except Exception as exc:
        # Logs full traceback to STDERR so Render captures it
        logging.exception("RAG query failed")
        # Propagate a safe error to caller (400 = client-side issue)
        raise HTTPException(
            status_code=400,
            detail=f"Query failed: {type(exc).__name__}",
        )

