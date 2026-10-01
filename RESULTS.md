# Verification Results — RAG Q&A API

**Date:** 2026-10-01
**Script:** `verify_e2e.py` (committed in this repo)
**Result:** ALL CHECKS PASSED in 25.9s

## Environment

- Linux VM, Python 3.12, CPU-only
- Isolated venv with: `fastapi`, `sentence-transformers`, `faiss-cpu`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (~90MB, downloaded
  once from Hugging Face Hub, then cached)

## Command

```bash
~/workspace/ml-projects/venv/bin/python verify_e2e.py
```

The script boots the FastAPI app with `TestClient` (running the real startup:
model load + FAISS index build over `data/`) and exercises every endpoint.

## What was verified

| Check | Result |
|---|---|
| `GET /health` | `{"status": "ok", "chunks_indexed": 3}` |
| `POST /ask` ("What is the accuracy tradeoff of post-training quantization?") | 200, top source `quantization.txt` (score 0.631), sensible ranking of the other two docs |
| `POST /ingest` (new doc `attention_notes.txt`) | `{"status": "ingested", "chunks_indexed": 3}` in 0.4s |
| `POST /ask` about the newly ingested doc | 200, `attention_notes.txt` retrieved as top source (score 0.578) — proves fresh content is searchable |
| `POST /rebuild` | `{"status": "rebuilt", "chunks_indexed": 3}` in 0.4s |
| Edge cases | empty question → 400, non-`.txt` upload → 400 (as designed) |

## Limitations (read before citing this project)

- Answers use the **extractive fallback** (most relevant passage returned
  verbatim). Generative answers require `LLM_BASE_URL`/`LLM_API_KEY`, which
  were not configured in this run.
- Test corpus is 3 small `.txt` files; this verifies the pipeline mechanics,
  not retrieval quality at scale.
- No latency/throughput benchmarks were run.
