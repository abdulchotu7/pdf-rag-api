# PDF RAG API

Production-ready FastAPI application for PDF Question & Answer using Retrieval-Augmented Generation (RAG). Built with LlamaIndex, DeepSeek V4 Flash, Qdrant, BAAI/bge-m3 embeddings, and BAAI/bge-reranker-v2-m3.

## Architecture

```
app/
├── api/            # FastAPI route handlers
│   ├── health.py   # Health check endpoint
│   ├── upload.py   # PDF upload endpoint (sync + async)
│   ├── chat.py     # Q&A endpoint (sync + streaming SSE)
│   ├── documents.py # Document management (list, delete)
│   └── tasks.py    # Async task status endpoint
├── core/           # Config, logging, exceptions
├── services/       # Business logic, task manager
├── llm/            # LLM abstraction (DeepSeek)
├── embeddings/     # Embedding abstraction (BGE)
├── vectordb/       # Vector DB abstraction (Qdrant)
├── reranker/       # Reranker abstraction (BGE reranker)
├── rag/            # RAG pipeline (ingestion, retrieval, query, hybrid)
├── schemas/        # Pydantic models
├── static/         # Web UI
└── main.py         # App entry point
```

Clean layered architecture with SOLID principles — routes never talk to LlamaIndex or Qdrant directly.

## Retrieval Pipeline

### Dense (default)

```
Question → Embedding (BGE) → Qdrant (top_k) → DeepSeek → Answer + Sources
```

### Dense + Reranker

```
Question → Embedding (BGE) → Qdrant (top_k=20) → BGE Reranker (top_n=5) → DeepSeek → Answer + Sources
```

### Hybrid (dense + sparse + RRF fusion + reranker)

```
Question
  → Dense Retriever (BGE embedding + Qdrant, top_k=10)
  → BM25 Retriever (keyword, top_k=10)
  → QueryFusionRetriever (Reciprocal Rank Fusion, top_k=15)
  → BGE Reranker (top_n=5)
  → DeepSeek → Answer + Sources
```

## Tech Stack

- **Python 3.11+** / **FastAPI** (tested on 3.11 and 3.14)
- **LlamaIndex** — PDF parsing, chunking, fusion retriever
- **DeepSeek V4 Flash** — LLM via OpenAI-compatible API (thinking disabled)
- **BAAI/bge-m3** — Local embeddings via HuggingFace (float16 on MPS/CUDA)
- **BAAI/bge-reranker-v2-m3** — Cross-encoder reranker (float16 on MPS/CUDA)
- **Qdrant** — Vector database
- **BM25S** — Sparse keyword retrieval
- **Apple Silicon (MPS)** — Auto-detected and optimized
- **Docker** & **Docker Compose**

## Prerequisites

- Python 3.11+ (local dev) or Docker (containerized)
- DeepSeek API key
- Apple Silicon (M1–M5) recommended for best local performance

## Environment Variables

Copy `.env` and configure:

### Core

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | DeepSeek API key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | API endpoint |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | Model name |

### Qdrant

| Variable | Default | Description |
|---|---|---|
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_COLLECTION` | `documents` | Collection name |

### Embeddings

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HuggingFace embedding model |
| `EMBEDDING_DEVICE` | `auto` | Device: `auto` (MPS→CUDA→CPU), `mps`, `cuda`, `cpu` |
| `EMBEDDING_DIMENSION` | `1024` | Vector dimension |

### Chunking

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `512` | Text chunk size (tokens) |
| `CHUNK_OVERLAP` | `64` | Chunk overlap |

### Retrieval

| Variable | Default | Description |
|---|---|---|
| `TOP_K` | `10` | Default top-k (dense only) |
| `RERANK_ENABLED` | `true` | Enable reranker |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker model |
| `RETRIEVAL_TOP_K` | `20` | Initial retrieval count (before reranker) |
| `FINAL_TOP_K` | `5` | Final count (after reranker) |

### Hybrid Search

| Variable | Default | Description |
|---|---|---|
| `HYBRID_ENABLED` | `true` | Enable dense + sparse fusion |
| `VECTOR_TOP_K` | `10` | Dense retriever top-k |
| `BM25_TOP_K` | `10` | BM25 keyword retriever top-k |
| `FUSION_TOP_K` | `15` | RRF fusion top-k (input to reranker) |

### Server

| Variable | Default | Description |
|---|---|---|
| `APP_HOST` | `0.0.0.0` | Bind address |
| `APP_PORT` | `8000` | Port |
| `APP_TIMEOUT` | `600` | Request timeout (seconds) |
| `MAX_UPLOAD_SIZE` | `1073741824` | Max file size (bytes, default 1 GB) |
| `UPLOAD_DIR` | `uploads` | Temp upload directory |
| `LOG_LEVEL` | `INFO` | Logging level |

## Running Locally

```bash
uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt

# Start Qdrant (Docker)
docker run -d -p 6333:6333 qdrant/qdrant

# Set your API key in .env
# Run the app
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000` for the web UI.

**First run**: BGE-M3 (~2.3 GB) and the reranker (~1.1 GB) download from HuggingFace on startup, then stay cached — expect a ~15–30 s cold start, not an error.

**Stop the app**: `Ctrl+C` in the terminal that started it. If the terminal was closed first, the server keeps running in the background (orphaned) — find and kill it:

```bash
lsof -i :8000            # find the PID holding the port
kill <PID>               # Ctrl+C equivalent; kill -9 if it won't die
```

**`address already in use` on startup** means a stale server still holds port 8000 (usually from a closed terminal). Run the two commands above before starting again.

**Don't move or delete the venv while the server is running.** A server keeps running from a venv that was later trashed or renamed — it serves stale code, keeps the port, and breaks once Trash is emptied. Restart with the current `.venv` after recreating it.

## Running with Docker

```bash
docker compose up --build
```

Starts the API on `http://localhost:8000` and Qdrant on `http://localhost:6333`.

## Web UI

The web UI at `http://localhost:8000` provides a chat interface with real-time streaming, status indicators, and latency metrics.

## API

### Health

```bash
curl http://localhost:8000/health
```

```json
{ "status": "ok" }
```

### Upload PDF (sync)

Returns immediately after ingestion completes. Best for files < 100 MB. Duplicate files (same content, any filename) are automatically detected and skipped — returns existing `document_id` with `chunks: 0`.

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@document.pdf"
```

```json
[
  {
    "document_id": "uuid",
    "filename": "document.pdf",
    "chunks": 42,
    "pages": 15
  }
]
```

Multiple files supported:

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf"
```

### Upload PDF (async)

Returns immediately with a `task_id`. File is streamed to disk in 8 MB chunks (memory-safe for large PDFs). Processing happens in the background. Duplicate detection applies here too.

```bash
curl -X POST http://localhost:8000/upload/async \
  -F "files=@document.pdf"
```

```json
{
  "tasks": [
    {
      "filename": "document.pdf",
      "task_id": "uuid"
    }
  ]
}
```

### Check Task Status

```bash
curl http://localhost:8000/tasks/{task_id}
```

```json
{
  "task_id": "uuid",
  "status": "completed",
  "progress": 15,
  "total": 15,
  "result": {
    "document_id": "uuid",
    "filename": "document.pdf",
    "chunks": 42,
    "pages": 15
  },
  "error": null
}
```

Status values: `pending` → `processing` → `completed` / `failed`.

### List Documents

```bash
curl http://localhost:8000/documents
```

```json
[
  {
    "document_id": "uuid",
    "filename": "document.pdf",
    "chunks": 42,
    "pages": 15
  }
]
```

### Delete Document

```bash
curl -X DELETE http://localhost:8000/documents/{document_id}
```

Returns `204 No Content`.

### Ask a Question (sync)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain chapter 4"}'
```

```json
{
  "answer": "Chapter 4 discusses vector databases... (Source: document.pdf, Page 17)",
  "sources": [
    {
      "page": 17,
      "filename": "document.pdf",
      "score": 4.85
    }
  ]
}
```

Scores reflect the reranker relevance score (not cosine similarity) when reranker is enabled.

### Ask a Question (streaming)

Returns Server-Sent Events with status updates, sources, tokens, and metrics. Provides much better perceived latency than the sync endpoint.

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain chapter 4"}'
```

Event stream:

```
event: status
data: {"type":"status","message":"Embedding query..."}

event: status
data: {"type":"status","message":"Hybrid search (dense + BM25)..."}

event: status
data: {"type":"status","message":"Reranking results..."}

event: status
data: {"type":"status","message":"Generating answer..."}

event: sources
data: {"type":"sources","sources":[{"page":17,"filename":"document.pdf","score":4.85}]}

event: token
data: {"type":"token","token":"Chapter"}

event: token
data: {"type":"token","token":" 4"}

...

event: metrics
data: {"type":"metrics","latency_ms":3180,"retrieval_ms":472,"rerank_ms":1184,"llm_ms":1524,"tokens_in":1820,"tokens_out":326}

event: done
data: {"type":"done"}
```

## Reranker

The BGE cross-encoder reranker re-ranks retrieved chunks by relevance to the query:

1. Retrieves `RETRIEVAL_TOP_K` (or `FUSION_TOP_K` in hybrid mode) chunks from Qdrant
2. Reranks all chunks using `BAAI/bge-reranker-v2-m3` (batched inference, float16 on MPS/CUDA)
3. Keeps top `FINAL_TOP_K` chunks
4. Passes only reranked chunks to the LLM

Set `RERANK_ENABLED=false` to disable.

## Hybrid Search

Combines dense vector search (semantic) with BM25 keyword search (lexical) using Reciprocal Rank Fusion via `QueryFusionRetriever`:

1. **Dense**: Embed query → search Qdrant (`VECTOR_TOP_K`)
2. **Sparse**: BM25 keyword search over all indexed chunks (`BM25_TOP_K`)
3. **Fusion**: RRF with `num_queries=1` → top `FUSION_TOP_K`
4. **Reranker**: BGE reranker → top `FINAL_TOP_K`
5. **LLM**: Generate answer from reranked context

Set `HYBRID_ENABLED=false` to disable (falls back to dense only).

## Duplicate Detection

Files are deduplicated by SHA-256 **content hash**, not filename. Uploading the same PDF twice (even with different filenames) returns the existing `document_id` with zero processing cost. The hash is stored in the Qdrant payload for future lookups.

## Apple Silicon Optimization

When `EMBEDDING_DEVICE=auto`, the app detects the best available device:

1. **MPS** (Apple Silicon) — fastest, used on M1–M5 Macs
2. **CUDA** — NVIDIA GPU
3. **CPU** — fallback

Both BGE-M3 embeddings and BGE reranker use `torch.float16` on MPS/CUDA for ~2x inference speedup. Reranker uses `torch.inference_mode()` for minimal autograd overhead.

## Processing Large PDFs

The async upload endpoint (`POST /upload/async`) handles 1 GB+ PDFs:

- File streamed to disk in 8 MB chunks (not loaded into memory)
- Pages processed one at a time via `pypdf` (never all in memory)
- Embeddings generated in batches of 16
- Progress tracked via task status endpoint
- Temp files cleaned up after processing

## Latency Logging

Per-query latency breakdown logged as both structured `extra` (for log aggregators) and inline message:

```
mode=hybrid embed=145ms dense_search=15ms bm25=101ms rerank=1259ms llm=1661ms total=3181ms retrieved=15 final=5
```

| Field | Description |
|---|---|
| `mode` | `dense` or `hybrid` |
| `embed_ms` | Time to embed the query |
| `dense_search_ms` | Time for Qdrant vector search |
| `bm25_ms` | Time for BM25 + RRF fusion |
| `rerank_ms` | Time for cross-encoder reranking |
| `llm_ms` | Time for DeepSeek API call |
| `total_ms` | Total request time |
| `retrieved_count` | Nodes before reranker |
| `final_count` | Nodes after reranker |

All models are preloaded and warmed up during application startup — zero cold start on first request.

## Folder Structure

```
app/
├── api/
│   ├── chat.py          # POST /chat (sync), POST /chat/stream (SSE)
│   ├── dependencies.py  # DI container, lifespan, model startup
│   ├── documents.py     # GET /documents, DELETE /documents/{id}
│   ├── health.py        # GET /health
│   ├── tasks.py         # GET /tasks/{task_id}
│   └── upload.py        # POST /upload, POST /upload/async
├── core/
│   ├── config.py        # Pydantic Settings from .env
│   ├── exceptions.py    # Typed exception hierarchy
│   └── logging.py       # Structured logging
├── embeddings/
│   ├── base.py          # Abstract BaseEmbedding
│   └── bge.py           # BGEEmbeddingProvider (float16, MPS/CUDA/CPU)
├── llm/
│   ├── base.py          # Abstract BaseLLM
│   └── deepseek.py      # DeepSeekProvider (V4 Flash, thinking disabled)
├── rag/
│   ├── hybrid_retriever.py  # Dense + BM25 + RRF fusion retriever
│   ├── query_engine.py      # Sync + streaming query orchestration
│   ├── retrieval.py         # Dense-only retriever (base path)
│   └── streaming_ingestion.py  # Page-at-a-time ingestion with dedup
├── reranker/
│   ├── base.py           # Abstract BaseReranker
│   ├── bge.py            # BGEReranker (float16, inference_mode)
│   └── factory.py        # RerankerFactory
├── schemas/
│   └── models.py         # Pydantic models
├── services/
│   ├── chat_service.py       # Chat orchestration
│   ├── ingest_service.py     # Upload + dedup + document management
│   └── task_manager.py       # In-memory async task tracking
├── static/
│   └── index.html            # Web UI (streaming SSE chat)
├── utils/
│   └── device.py             # Device resolution (auto/mps/cuda/cpu)
├── vectordb/
│   ├── base.py               # Abstract BaseVectorStore
│   └── qdrant.py             # QdrantVectorStore (async, dedup)
├── main.py                   # FastAPI app, static serving, error handlers
```

## Roadmap

- [ ] OCR for scanned documents
- [ ] Authentication & multi-tenancy
- [ ] Chat history & conversation
- [ ] LangGraph agent workflows
- [ ] Embedding cache
- [ ] Persistent document registry (beyond Qdrant scroll)
- [ ] Unit + integration tests
- [ ] CI/CD pipeline
- [ ] RAG evaluation (Ragas / DeepEval)
