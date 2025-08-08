# main.py — Karl AI RAG (FastAPI, OpenAI embeds, Chroma 0.5, parent-expansion)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple, Literal
from functools import lru_cache
from dotenv import load_dotenv

import os
import logging
import re

# ---------- Vector store / embeddings ----------
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb

# ---------- OpenAI chat (SDK v1) ----------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # we'll hard-fail on startup if missing

# ---------- Tokenizer (optional; we fallback if not present) ----------
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("o200k_base")
except Exception:
    _ENC = None

# =================== App bootstrap ===================

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("karl-ai")

app = FastAPI(title="Karl AI – RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# =================== Schemas ===================

class ChatRequest(BaseModel):
    query: str = Field(..., description="User question")
    mode: Optional[
        Literal[
            "smalltalk", "identity", "lesson_fact", "teach_concept",
            "decision_guide", "howto", "compare", "summarize"
        ]
    ] = Field(None, description="Force a response mode")
    k_parents: int = Field(3, ge=1, le=5)
    debug: bool = Field(False)

class ChatResponse(BaseModel):
    answer: str
    mode: Optional[str] = None
    lessons: Optional[List[str]] = None   # labels like ["[1] Title", "[2] ..."] when debug=True

class DebugRetrieveRequest(BaseModel):
    query: str
    top_chunks: int = 12
    k_parents: int = 3

class DebugRetrieveResponse(BaseModel):
    parents: List[Dict[str, Any]]  # {"id": "...", "label": "...", "max_score": float, "snippet": str}
    top_score: float

# =================== Globals ===================

INDEX: Optional[VectorStoreIndex] = None
CHROMA_COLLECTION = None
OPENAI_CLIENT: Any = None
OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o")

# =================== Utilities ===================

def _token_len(text: str) -> int:
    if _ENC is None:
        return max(1, len(text) // 4)
    try:
        return len(_ENC.encode(text))
    except Exception:
        return max(1, len(text) // 4)

def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if _ENC is None:
        return text[: max_tokens * 4]
    toks = _ENC.encode(text)
    if len(toks) <= max_tokens:
        return text
    return _ENC.decode(toks[:max_tokens])

# System prompts

def _sys_prompt_feynman() -> str:
    return (
        "You are Karl AI, trained on Karl Kaufman’s course material (American Dream Investing). "
        "Explain concepts so a smart 12-year-old could re-teach them. "
        "Use ONLY the provided lessons. If missing, say “I don’t know.” "
        "Always include bracket citations like [1], [2] tied to the provided lesson labels. "
        "No quizzes; expository teaching only. Prefer short sentences, plain English, and analogies. "
        "Write a single flowing explanation that covers: plain-English explanation; one analogy; a short precise definition "
        "from the lessons (with citations); a worked example (with citations); common pitfalls & how to avoid them; "
        "when to use / when not to; a mini checklist; finish with a brief recap."
    )

SYS_DIRECT = (
    "You are Karl AI. Answer ONLY from the provided lessons. If the answer is missing, say “I don’t know.” "
    "Be concise (2–5 sentences) and include bracket citations like [1], [2]."
)
SYS_DECISION = (
    "You are Karl AI, a teacher. Use ONLY the provided lessons. Do NOT give personal financial advice. "
    "Goal: help the learner reason with a clear framework.\n"
    "- Short answer — 1–2 sentences, with citations.\n"
    "- Trade-offs — pros/cons grounded in the lessons (bullets, with citations).\n"
    "- Decision frame — bullets: when to concentrate vs diversify; assumptions to check.\n"
    "- Guardrails — bullets on risk controls (position sizing, volatility tolerance, review cadence).\n"
    "- 30s recap — 3 bullets.\n"
    "Use citations like [1], [2]."
)
SYS_HOWTO = (
    "You are Karl AI, a teacher. Use ONLY the provided lessons. "
    "Write a step-by-step procedure with 5–9 steps, each 1–2 short sentences, grounded with citations. "
    "Then add a Checklist (4–6 bullets) and Common mistakes (3–5 bullets)."
)
SYS_COMPARE = (
    "You are Karl AI, a teacher. Use ONLY the provided lessons. "
    "Compare the two ideas side-by-side with a one-line takeaway, strengths/weaknesses, and when to choose A vs B. "
    "Use citations like [1], [2]."
)
SYS_SUMMARIZE = (
    "You are Karl AI. Summarize the requested lesson(s) using ONLY the provided text. "
    "Output 5–10 bullet key takeaways, then one-sentence TL;DR. Use citations like [1], [2]."
)

# Patterns for router

SMALLTALK_PAT = re.compile(r"^(hi|hello|hey|yo|sup|good\s*(morning|evening|afternoon)|how('?| )are (you|u))\b", re.I)
THANKS_PAT    = re.compile(r"\b(thanks|thank you|ty|appreciate it)\b", re.I)
IDENTITY_PAT  = re.compile(r"\b(who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|who\s+is\s+karl\s+ai)\b", re.I)
DECISION_PAT  = re.compile(r"\b(should|do i|is it (better|good|bad)|which (one|is better)|pros and cons|trade[- ]offs)\b", re.I)
HOWTO_PAT     = re.compile(r"\b(how do i|how to|steps?|process|guide|checklist|framework)\b", re.I)
COMPARE_PAT   = re.compile(r"\b(vs\.?|versus|difference between|compare)\b", re.I)
SUMMARIZE_PAT = re.compile(r"\b(summarize|tl;dr|key takeaways|outline|in one sentence)\b", re.I)
QUOTE_PAT     = re.compile(r"\b(quote|exact words|what did .* say)\b", re.I)

# =================== Startup ===================

@app.on_event("startup")
async def _startup():
    global INDEX, CHROMA_COLLECTION, OPENAI_CLIENT, OPENAI_CHAT_MODEL

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    if OpenAI is None:
        raise RuntimeError("openai SDK v1 not installed. `pip install openai`")

    OPENAI_CLIENT = OpenAI()
    OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o")

    chroma_path = os.environ.get("CHROMA_PATH", "./index")
    collection_name = os.environ.get("CHROMA_COLLECTION", "").strip()

    client = chromadb.PersistentClient(path=chroma_path)

    if collection_name:
        CHROMA_COLLECTION = client.get_or_create_collection(collection_name)
    else:
        cols = client.list_collections()
        if not cols:
            raise RuntimeError(f"No Chroma collections found in {chroma_path}")
        CHROMA_COLLECTION = cols[0]
        log.info(f"Using Chroma collection: {CHROMA_COLLECTION.name}")

    vector_store = ChromaVectorStore(chroma_collection=CHROMA_COLLECTION)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = OpenAIEmbedding(model=os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-large"))

    INDEX = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_ctx,
        embed_model=embed_model,
    )

    # Warm a retriever only (avoid pulling LLM deps)
    INDEX.as_retriever(similarity_top_k=1).retrieve("ping")
    log.info("✅ RAG index loaded and ready")

# =================== Parent text access ===================

@lru_cache(maxsize=2048)
def _parent_text_and_meta(parent_id: str) -> Tuple[str, Dict[str, Any]]:
    """
    Fetch ALL chunks for a given parent from Chroma by `airtable_id` (preferred).
    Fallbacks try `ref_doc_id` and `doc_id` if your metadata differs.
    """
    if CHROMA_COLLECTION is None:
        return "", {}

    def _fetch(where: Dict[str, Any]):
        return CHROMA_COLLECTION.get(where=where, include=["documents", "metadatas"])

    # 1) airtable_id (our standard)
    res = _fetch({"airtable_id": parent_id})
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    if not docs:
        # 2) fallback to ref_doc_id
        res = _fetch({"ref_doc_id": parent_id})
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
    if not docs:
        # 3) fallback to doc_id
        res = _fetch({"doc_id": parent_id})
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []

    if not docs:
        return "", {}

    joined = "\n\n".join(docs)
    md_sample: Dict[str, Any] = {}
    for m in metas:
        if m:
            md_sample = m
            break
    return joined, md_sample

def _choose_parent_ids(query: str, top_chunks: int = 12, k_parents: int = 3) -> Tuple[List[str], float]:
    """Select top parent lessons by max chunk score; prefer airtable_id as the parent key."""
    assert INDEX is not None
    hits = INDEX.as_retriever(similarity_top_k=top_chunks).retrieve(query)
    grouped: Dict[str, List[float]] = {}
    for h in hits:
        node = h.node
        # prefer airtable_id; fallback to ref_doc_id if missing
        pid = (node.metadata or {}).get("airtable_id") or getattr(node, "ref_doc_id", None)
        if pid:
            grouped.setdefault(pid, []).append(h.score or 0.0)
    ranked = sorted(grouped.items(), key=lambda kv: max(kv[1]), reverse=True)
    top_score = max((h.score or 0.0 for h in hits), default=0.0)
    return [pid for pid, _ in ranked[:k_parents]], float(top_score)

def _pack_parents(parent_ids: List[str], max_tokens: int) -> Tuple[List[str], str]:
    """Build context under a token budget; labels like “[i] Title or Slug”."""
    labels: List[str] = []
    blocks: List[str] = []
    used = 0

    for i, pid in enumerate(parent_ids, 1):
        full, md = _parent_text_and_meta(pid)
        if not full:
            continue
        title = (md or {}).get("Title") or (md or {}).get("Slug") or "Lesson"
        label = f"[{i}] {title}"

        tlen = _token_len(full)
        if used + tlen <= max_tokens:
            blocks.append(f"{label}\n{full}"); labels.append(label); used += tlen
        else:
            remain = max_tokens - used
            if remain > 200:
                blocks.append(f"{label}\n{_truncate_to_tokens(full, remain)}"); labels.append(label)
            break

    return labels, "\n\n---\n\n".join(blocks)

# =================== Router & answerers ===================

def _classify(query: str, top_score: float) -> str:
    q = query.strip()
    if not q or THANKS_PAT.search(q) or SMALLTALK_PAT.search(q):
        return "smalltalk"
    if IDENTITY_PAT.search(q):
        return "identity"
    if top_score < 0.25:
        return "smalltalk"
    if DECISION_PAT.search(q):
        return "decision_guide"
    if HOWTO_PAT.search(q):
        return "howto"
    if COMPARE_PAT.search(q):
        return "compare"
    if SUMMARIZE_PAT.search(q):
        return "summarize"
    tokens = len(re.findall(r"\w+", q))
    if QUOTE_PAT.search(q) or (tokens <= 10 and top_score >= 0.55):
        return "lesson_fact"
    return "teach_concept"

def _openai_chat(system: str, user: str, temperature: float = 0.2) -> str:
    resp = OPENAI_CLIENT.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content

def _answer_direct(query: str, labels: List[str], ctx: str) -> str:
    return _openai_chat(
        SYS_DIRECT,
        f"{ctx}\n\nQuestion: {query}\n\nAnswer (with citations):",
        temperature=0.1,
    )

def _answer_decision(query: str, labels: List[str], ctx: str) -> str:
    return _openai_chat(
        SYS_DECISION,
        f"{ctx}\n\nQuestion: {query}\n\nAnswer (with citations):",
        temperature=0.2,
    )

def _answer_howto(query: str, labels: List[str], ctx: str) -> str:
    return _openai_chat(
        SYS_HOWTO,
        f"{ctx}\n\nTask: {query}\n\nWrite the procedure (with citations):",
        temperature=0.2,
    )

def _answer_compare(query: str, labels: List[str], ctx: str) -> str:
    return _openai_chat(
        SYS_COMPARE,
        f"{ctx}\n\nCompare request: {query}\n\nCompare (with citations):",
        temperature=0.2,
    )

def _answer_summarize(query: str, labels: List[str], ctx: str) -> str:
    return _openai_chat(
        SYS_SUMMARY := SYS_SUMMARIZE,
        f"{ctx}\n\nSummarize request: {query}\n\nSummary (with citations):",
        temperature=0.1,
    )

def _answer_feynman(query: str, labels: List[str], ctx: str) -> str:
    return _openai_chat(
        _sys_prompt_feynman(),
        f"{ctx}\n\nQuestion: {query}\n\nTeach this concept in a flowing explanation (with citations):",
        temperature=0.2,
    )

# =================== Routes ===================

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/ping")
async def ping():
    return {"message": "pong"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        if INDEX is None or CHROMA_COLLECTION is None or OPENAI_CLIENT is None:
            raise RuntimeError("Server not initialized")

        parent_ids, top_score = _choose_parent_ids(req.query, k_parents=req.k_parents)
        mode = req.mode or _classify(req.query, top_score)

        budgets = {
            "lesson_fact": 2500, "howto": 6500, "compare": 6500, "summarize": 5500,
            "decision_guide": 7000, "teach_concept": 7000, "smalltalk": 0, "identity": 0,
        }

        if mode in ("smalltalk", "identity"):
            if mode == "smalltalk":
                return ChatResponse(answer="Hey! I’m Karl AI — ready when you are. Ask me a topic or a question from the lessons.", mode=mode)
            return ChatResponse(answer="I’m Karl AI — trained on Karl Kaufman’s course content (American Dream Investing). I explain ideas with examples and citations from the lessons, or give concise answers when you just want the fact.", mode=mode)

        labels, ctx = _pack_parents(parent_ids, max_tokens=budgets.get(mode, 3000))
        if not labels:
            return ChatResponse(answer="I don't know. (Nothing relevant in the lessons.)", mode=mode, lessons=[])

        if mode == "lesson_fact":
            ans = _answer_direct(req.query, labels, ctx)
        elif mode == "decision_guide":
            ans = _answer_decision(req.query, labels, ctx)
        elif mode == "howto":
            ans = _answer_howto(req.query, labels, ctx)
        elif mode == "compare":
            ans = _answer_compare(req.query, labels, ctx)
        elif mode == "summarize":
            ans = _answer_summarize(req.query, labels, ctx)
        else:
            ans = _answer_feynman(req.query, labels, ctx)

        return ChatResponse(answer=ans, mode=mode, lessons=(labels if req.debug else None))

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat failed")
        raise HTTPException(status_code=500, detail=f"Chat failed: {type(e).__name__}: {e}")

# -------- optional: debug retrieval (no LLM) --------
@app.post("/debug/retrieve", response_model=DebugRetrieveResponse)
async def debug_retrieve(req: DebugRetrieveRequest):
    try:
        if INDEX is None:
            raise RuntimeError("Server not initialized")

        hits = INDEX.as_retriever(similarity_top_k=req.top_chunks).retrieve(req.query)
        grouped: Dict[str, Dict[str, Any]] = {}

        def _snippet(text: str, kws: List[str]) -> str:
            if not text: return ""
            tl = text.lower()
            pos = min([tl.find(k) for k in kws if tl.find(k) != -1] + [0])
            start = max(0, pos - 200); end = min(len(text), start + 420)
            return text[start:end].replace("\n", " ")

        kws = re.findall(r"\b[\w'-]{4,}\b", req.query.lower())

        for h in hits:
            node = h.node
            pid = (node.metadata or {}).get("airtable_id") or getattr(node, "ref_doc_id", None)
            if not pid: 
                continue
            g = grouped.setdefault(pid, {"scores": [], "label": "", "snippet": ""})
            g["scores"].append(h.score or 0.0)
            full, md = _parent_text_and_meta(pid)
            title = (md or {}).get("Title") or (md or {}).get("Slug") or "Lesson"
            g["label"] = title
            if not g["snippet"]:
                g["snippet"] = _snippet(node.get_content(), kws)

        ranked = sorted(grouped.items(), key=lambda kv: max(kv[1]["scores"]), reverse=True)[:req.k_parents]
        parents = [{
            "id": pid,
            "label": f"[{i+1}] {info['label']}",
            "max_score": max(info["scores"]),
            "snippet": info["snippet"],
        } for i, (pid, info) in enumerate(ranked)]
        top_score = max((h.score or 0.0 for h in hits), default=0.0)
        return DebugRetrieveResponse(parents=parents, top_score=top_score)

    except Exception as e:
        log.exception("Debug retrieve failed")
        raise HTTPException(status_code=500, detail=f"Debug retrieve failed: {type(e).__name__}: {e}")

# Local dev entrypoint:
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))

