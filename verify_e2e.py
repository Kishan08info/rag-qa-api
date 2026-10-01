"""End-to-end verification of the RAG Q&A API (CPU).
Boots the FastAPI app with TestClient, exercises every endpoint, times each step,
and prints real outputs. Exits non-zero on any failure.
Run: ~/workspace/ml-projects/venv/bin/python verify_e2e.py
"""
import time

t0 = time.time()
import app as rag_app

t_import = time.time()
print(f"[import] app.py imported in {t_import - t0:.1f}s")

from fastapi.testclient import TestClient

t1 = time.time()
with TestClient(rag_app.app) as client:
    t2 = time.time()
    print(f"[startup] app startup + FAISS index build: {t2 - t1:.1f}s "
          f"(includes embedding-model download on first run)")

    # --- 1. /health ---
    r = client.get("/health")
    assert r.status_code == 200, r.text
    health = r.json()
    print(f"[health] {health}")
    assert health["chunks_indexed"] > 0, "index is empty!"

    # --- 2. /ask (extractive fallback; no LLM key set) ---
    question = "What is the accuracy tradeoff of post-training quantization?"
    t3 = time.time()
    r = client.post("/ask", json={"question": question})
    t4 = time.time()
    assert r.status_code == 200, r.text
    resp = r.json()
    print(f"[ask] '{question}' -> {t4 - t3:.1f}s")
    print(f"[ask] sources={resp['sources']} scores={[round(s, 4) for s in resp['scores']]}")
    print(f"[ask] answer preview: {resp['answer'][:300]}")
    assert resp["sources"], "no sources returned"
    assert 0.0 <= resp["scores"][0] <= 1.0, "score out of range"

    # --- 3. /ingest a new doc ---
    doc = ("Transformer attention scales quadratically with sequence length, which "
           "motivates techniques like FlashAttention and KV caching for long contexts.")
    t5 = time.time()
    r = client.post("/ingest", files={"file": ("attention_notes.txt", doc.encode())})
    t6 = time.time()
    assert r.status_code == 200, r.text
    print(f"[ingest] {r.json()} in {t6 - t5:.1f}s")

    # --- 4. /ask about the newly ingested doc (tests retrieval of fresh content) ---
    r = client.post("/ask", json={"question": "Why does attention scale quadratically?"})
    assert r.status_code == 200, r.text
    resp2 = r.json()
    print(f"[ask-new] sources={resp2['sources']} "
          f"scores={[round(s, 4) for s in resp2['scores']]}")
    assert "attention_notes.txt" in resp2["sources"], "fresh doc not retrieved!"
    print(f"[ask-new] answer preview: {resp2['answer'][:200]}")

    # --- 5. /rebuild ---
    t7 = time.time()
    r = client.post("/rebuild")
    t8 = time.time()
    assert r.status_code == 200, r.text
    print(f"[rebuild] {r.json()} in {t8 - t7:.1f}s")

    # --- 6. Edge cases ---
    r = client.post("/ask", json={"question": "   "})
    assert r.status_code == 400, "empty question should 400"
    r = client.post("/ingest", files={"file": ("bad.pdf", b"not-a-txt")})
    assert r.status_code == 400, "non-txt ingest should 400"
    print("[edge] empty-question -> 400, non-txt ingest -> 400 (as designed)")

print(f"\nALL CHECKS PASSED in {time.time() - t0:.1f}s total")
