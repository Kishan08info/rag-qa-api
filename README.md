# RAG Q&A API

A retrieval-augmented generation service over your own documents, built with **FastAPI**,
**sentence-transformers**, and **FAISS**.

## Architecture

```
.txt docs ──▶ chunk (400 words, 50 overlap) ──▶ embed (all-MiniLM-L6-v2)
                                                        │
                                                        ▼
question ──▶ embed ──▶ FAISS top-k search ──▶ context ──▶ LLM ──▶ cited answer
```

- **Ingestion:** documents are chunked with word overlap, embedded into dense vectors,
  L2-normalized, and stored in a FAISS `IndexFlatIP` index (cosine similarity).
- **Retrieval:** the question is embedded with the same model; top-k chunks are fetched
  by inner-product search.
- **Generation:** retrieved chunks are injected into an LLM prompt with a
  "use only the context, cite sources" system instruction. Without an LLM key configured,
  the API falls back to returning the most relevant passage (extractive mode).

## Run it

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Try it:

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What is the accuracy tradeoff of post-training quantization?"}'

# Add your own documents:
curl -X POST localhost:8000/ingest -F "file=@notes.txt"
```

For generative answers, set `LLM_BASE_URL` and `LLM_API_KEY` (any OpenAI-compatible
endpoint) before starting the server.

## What this demonstrates

- End-to-end RAG pipeline: chunking strategy, dense retrieval, prompt grounding
- Vector search with FAISS (embeddings, normalization, inner-product ranking)
- Production API design with FastAPI: file ingestion, index rebuilds, health checks
- Graceful degradation (extractive fallback when no LLM is configured)

## Interview talking points

- Why chunk with overlap? (Boundary context loss; 400/50 balances relevance vs. context.)
- Why normalize embeddings + inner product? (Equals cosine similarity; FAISS has no native cosine index.)
- Failure modes: retrieval misses (fix with better chunking/re-ranking), unfaithful
  generation (fix with stricter prompting + citation requirements).
