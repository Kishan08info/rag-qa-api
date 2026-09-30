"""
RAG Q&A API — retrieval-augmented generation over your own documents.

Pipeline:  .txt docs -> chunk -> embed (sentence-transformers) -> FAISS index
           question -> embed -> top-k retrieve -> generate answer (LLM or extractive fallback)

Run:
    pip install -r requirements.txt
    uvicorn app:app --reload
    # then: curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
    #            -d '{"question": "What is quantization?"}'
"""
import json
import os
import urllib.request
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

DATA_DIR = Path(__file__).parent / "data"
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "3"))
CHUNK_WORDS = 400
CHUNK_OVERLAP = 50

# Optional OpenAI-compatible LLM for answer generation.
# If unset, /ask falls back to an extractive answer from the top chunk.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")   # e.g. https://api.openai.com
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

app = FastAPI(title="RAG Q&A API")

_embedder = None
_index = None
_chunks: List[str] = []
_sources: List[str] = []


def get_embedder():
    """Lazy-load so `import app` stays light (useful for tests)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {EMBED_MODEL}")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping word-chunks for retrieval."""
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return chunks


def build_index() -> int:
    """(Re)build the FAISS index from data/*.txt. Returns chunk count."""
    global _index, _chunks, _sources
    import faiss

    _chunks, _sources = [], []
    for f in sorted(DATA_DIR.glob("*.txt")):
        for c in chunk_text(f.read_text(encoding="utf-8")):
            _chunks.append(c)
            _sources.append(f.name)
    if not _chunks:
        raise RuntimeError(f"No documents found in {DATA_DIR}/")

    emb = get_embedder().encode(_chunks, convert_to_numpy=True, normalize_embeddings=True)
    _index = faiss.IndexFlatIP(emb.shape[1])  # cosine similarity via inner product
    _index.add(emb.astype(np.float32))
    print(f"Indexed {len(_chunks)} chunks from {DATA_DIR}/")
    return len(_chunks)


def retrieve(question: str, k: int) -> List[dict]:
    if _index is None:
        raise RuntimeError("Index not built — call build_index() first.")
    q = get_embedder().encode([question], convert_to_numpy=True,
                              normalize_embeddings=True).astype(np.float32)
    scores, idx = _index.search(q, min(k, len(_chunks)))
    return [{"text": _chunks[i], "source": _sources[i], "score": float(scores[0][j])}
            for j, i in enumerate(idx[0])]


def generate_answer(question: str, contexts: List[dict]) -> str:
    context_block = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in contexts)
    if LLM_BASE_URL and LLM_API_KEY:
        payload = json.dumps({
            "model": LLM_MODEL,
            "messages": [
                {"role": "system",
                 "content": "Answer the question using ONLY the provided context. "
                            "Cite sources by filename. If the answer is not in the context, say so."},
                {"role": "user",
                 "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
            ],
            "temperature": 0.2,
        }).encode()
        req = urllib.request.Request(
            f"{LLM_BASE_URL.rstrip('/')}/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_API_KEY}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    # Extractive fallback: return the most relevant chunk verbatim.
    top = contexts[0]
    return (f"(extractive fallback — set LLM_BASE_URL/LLM_API_KEY for generative answers)\n\n"
            f"Most relevant passage from {top['source']}:\n{top['text'][:1200]}")


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    scores: List[float]


@app.on_event("startup")
def _startup():
    DATA_DIR.mkdir(exist_ok=True)
    try:
        build_index()
    except RuntimeError as e:
        print(f"Warning: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(_chunks)}


@app.post("/rebuild")
def rebuild():
    try:
        n = build_index()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "rebuilt", "chunks_indexed": n}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")
    dest = DATA_DIR / file.filename
    dest.write_bytes(await file.read())
    n = build_index()
    return {"status": "ingested", "file": file.filename, "chunks_indexed": n}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    k = req.top_k or TOP_K
    try:
        contexts = retrieve(req.question, k)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    answer = generate_answer(req.question, contexts)
    return AskResponse(answer=answer,
                       sources=[c["source"] for c in contexts],
                       scores=[c["score"] for c in contexts])
