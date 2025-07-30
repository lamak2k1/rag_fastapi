from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding


# ------------------------------------------------------------------
load_dotenv()                     # ⬅️  reads your .env for OPENAI_API_KEY
app = FastAPI(title="RAG‑API")

# ---------- Pydantic models ---------- #
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str

# ---------- load the index once at startup ---------- #
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext

@app.on_event("startup")
async def load_rag():
    global qa_engine

    persist_dir = "index"                     # ← where your files sit
    client = chromadb.PersistentClient(path=persist_dir)

    # pick the first existing collection; if none, raise an error
    collections = client.list_collections()
    if not collections:
        raise RuntimeError(f"No Chroma collections found in {persist_dir}")
    collection = collections[0]               # reuse the persisted one

    vector_store   = ChromaVectorStore(chroma_collection=collection)
    storage_ctx    = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001",task_type="retrieval_document")
    index  = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_ctx, embed_model=embed_model)

    qa_engine = index.as_query_engine()
    print("✅ Chroma‑based RAG index loaded")


# ---------- routes ---------- #
@app.get("/ping")
async def ping():
    return {"message": "pong"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    response = qa_engine.query(req.query)
    return ChatResponse(answer=str(response))

