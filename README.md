# RAG Service

A local CPU-friendly Retrieval-Augmented Generation service built with FastAPI,
ChromaDB, and sentence-transformers. Create multiple datasets, upload documents
into them, and run reference-aware semantic search over each dataset.

## Features

- Multi-dataset knowledge bases — each dataset is an isolated collection of documents
- Supported file types: `.txt`, `.md`, `.pdf`, `.html` / `.htm`
- Local embeddings (`sentence-transformers/all-MiniLM-L6-v2`) — no API key, CPU-only
- ChromaDB persistent vector store (cosine similarity)
- SQLite metadata store (Postgres-ready — change one URL)
- Server-rendered web UI (Jinja + vanilla JS, no build step)
- Every search hit returns a `reference`: dataset name, filename, chunk index, score
- Iterative re-indexing: re-uploading an unchanged file is a no-op (SHA-256 check)

## Prerequisites

- **Python 3.12** (standard CPython — `py -V` should report `3.12.x`)
- Windows / macOS / Linux
- ~1 GB free disk for dependencies (torch, sentence-transformers, etc.)

## Quick start (Windows + PowerShell)

```powershell
# 1. (one-time) allow venv activation for the current user
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force

# 2. Create and activate the virtual environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies (first time ~5 min; pulls in torch)
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the server (port 8765 — Windows reserves 8000 for Hyper-V)
uvicorn app.main:app --host 127.0.0.1 --port 8765 --reload
```

Then open:

- **UI:** http://127.0.0.1:8765/
- **Swagger API:** http://127.0.0.1:8765/docs

`Ctrl+C` to stop. `deactivate` to leave the venv.

### Quick start (macOS / Linux)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8765 --reload
```

### Without activating the venv

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8765 --reload
```

## Using the app

1. Open http://127.0.0.1:8765/ — go to **Datasets**
2. Create a dataset (e.g. `policies`)
3. Open the dataset, **upload** one or more documents (txt / md / pdf / html)
4. Go to **Search**, pick the dataset, ask a question
5. Each result card shows the passage plus its reference (filename + chunk index + dataset)

## REST API

| Method | Path | Purpose |
|---|---|---|
| GET    | `/api/health`                                  | Liveness check |
| POST   | `/api/datasets`                                | Create a dataset `{name, description?}` |
| GET    | `/api/datasets`                                | List datasets |
| GET    | `/api/datasets/{id}`                           | Dataset details |
| DELETE | `/api/datasets/{id}`                           | Delete dataset (cascades chunks + files) |
| GET    | `/api/datasets/{id}/documents`                 | List documents in a dataset |
| POST   | `/api/datasets/{id}/documents/upload`          | Multipart upload + index |
| DELETE | `/api/datasets/{id}/documents/{doc_id}`        | Remove a document |
| POST   | `/api/search`                                  | `{dataset_id, query, top_k}` → ranked hits with references |

See `/docs` for the full interactive Swagger spec.

## Configuration

All settings have defaults. Create a `.env` in the project root to override (prefix with `RAG_`):

```env
RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=120
RAG_DEFAULT_TOP_K=5
RAG_DATABASE_URL=sqlite:///./data/app.db
RAG_DOCS_DIR=data/docs
RAG_CHROMA_DIR=data/chroma
RAG_COLLECTION_NAME=documents
```

To migrate to PostgreSQL later:

```env
RAG_DATABASE_URL=postgresql+psycopg://user:pass@host:5432/dbname
```

(Then `pip install "psycopg[binary]"`.)

## Project layout

```
app/
  config.py        # pydantic-settings, env-driven
  db.py            # SQLAlchemy engine + session
  models.py        # ORM: Dataset, Document
  schemas.py       # Pydantic request/response
  repository.py    # SQL CRUD helpers
  loader.py        # file -> text (.txt/.md/.pdf/.html)
  chunker.py       # text -> overlapping windows
  embedder.py      # sentence-transformers wrapper (CPU, normalized)
  store.py         # ChromaDB wrapper, dataset-scoped queries
  indexer.py       # orchestration + SHA-256 change detection
  deps.py          # FastAPI Depends singletons
  main.py          # app factory + lifespan + router mounting
  routers/
    datasets.py    # /api/datasets/*
    search.py      # /api/search
    ui.py          # /, /datasets, /search HTML pages
  templates/       # Jinja: base, datasets, dataset, search
  static/          # app.js, styles.css
data/
  app.db           # SQLite (auto-created)
  chroma/          # vector index (auto-created)
  docs/<id>/       # uploaded files, isolated per dataset
requirements.txt
.env.example
```

## Troubleshooting

- **`Activate.ps1` not found** — your venv was created by a base Python (e.g. miniconda) that ships without activate templates. Delete `.venv` and recreate with `py -m venv .venv` from a standard CPython 3.12 install.
- **`WinError 10013` on port 8000** — Windows reserves a port range for Hyper-V. Use `--port 8765` (or any other free port).
- **First search/upload is slow** — the embedding model loads lazily on first use; subsequent calls are fast.
- **Re-upload returns `status: "skipped"`** — SHA-256 matched; the file content is unchanged. Edit the file (or rename it) to force re-indexing.

## Notes

- **No authentication** — single-operator mode. Add an auth layer before exposing on a network.
- **Retrieval only** — the server returns ranked passages with references. There's no LLM generation step; that's deliberate.
- **One dataset per search** in v1; multi-dataset is a small extension when needed.
