# RAG Service

A local CPU-friendly Retrieval-Augmented Generation service built with FastAPI,
ChromaDB, and sentence-transformers. Create multiple datasets, upload documents
into them, run reference-aware semantic search, and (optionally) have a configurable
LLM generate grounded answers from the retrieved passages.

## Features

- Multi-dataset knowledge bases — each dataset is an isolated collection of documents
- Supported file types: `.txt`, `.md`, `.pdf`, `.html` / `.htm`
- Local embeddings (`sentence-transformers/all-MiniLM-L6-v2`) — no API key, CPU-only
- ChromaDB persistent vector store (cosine similarity), per-dataset filtering
- SQLite metadata store (Postgres-ready — change one URL)
- Drag-and-drop multi-file upload with per-file status (web UI)
- **Folder upload** — pick a folder or drop one onto the upload zone; nested subfolders are expanded and relative paths are preserved (e.g. `policies/2024/handbook.pdf`)
- **Google Drive sync** — link a public Drive folder URL per dataset; lists remote files first, stores per-file metadata, skips unchanged files on re-sync, indexes only new or modified documents
- **Calibrated confidence** on every hit: percentage + High / Medium / Low badge, plus an overall verdict banner
- **LLM-backed answers** via any OpenAI-compatible endpoint (Ollama out of the box, real OpenAI by editing one file)
- Four LLM-related endpoints: full RAG (`/api/ask`), ask-with-supplied-context (`/api/llm/answer`), direct chat (`/api/llm/chat`), and config inspection (`/api/llm/info`)
- Server-rendered web UI (Jinja + vanilla JS, no build step)
- Every search hit returns a `reference` (dataset, filename, chunk index) and the LLM is instructed to cite as `[1]`, `[2]`, …
- Iterative re-indexing: re-uploading an unchanged file is a no-op (SHA-256 check)

## Prerequisites

- **Python 3.12** (standard CPython — `py -V` should report `3.12.x`)
- Windows / macOS / Linux
- ~1 GB free disk for dependencies (torch, sentence-transformers, etc.)
- An OpenAI-compatible LLM endpoint **only if** you want generated answers (Ollama works; see "LLM configuration" below)

## Deploy with Docker (Ubuntu VPS)

Run the whole service with one command. Persistent data is kept in a Docker named volume; the LLM config is bind-mounted so you can edit it on the host and restart without rebuilding.

### Server prerequisites

- Ubuntu 22.04+ (or any distro with Docker Engine 24+ and Compose v2)
- ~4 GB free disk (image is ~3 GB; data volume grows with your documents)
- ~2 GB free RAM at runtime
- TCP port 8765 reachable (firewall + provider security group)

Install Docker if it's not there:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER     # log out and back in for this to take effect
```

### One-time setup on the VPS

```bash
# 1. Get the code onto the server
git clone <your-repo-url> rag && cd rag        # or: scp -r . user@vps:~/rag

# 2. Set the LLM config (Ollama or OpenAI — see "LLM configuration" below)
nano llm_config.json

# 3. Build the image (first time: 5–10 min — torch CPU + model bake)
docker compose build
```

### Run

```bash
docker compose up -d              # start in the background
docker compose logs -f rag        # tail logs (Ctrl+C to detach)
```

Verify:

```bash
curl -s http://localhost:8765/api/health    # {"status":"ok"}
```

Then visit `http://<vps-ip>:8765/` for the UI and `/docs` for Swagger.

### Day-to-day commands

```bash
docker compose restart rag        # apply changes to llm_config.json
docker compose pull && docker compose up -d   # if you ship images via a registry
docker compose down               # stop (data preserved)
docker compose down -v            # stop AND wipe the data volume (datasets, chroma, uploads)
```

### Updating the app

```bash
git pull
docker compose up -d --build      # rebuild + restart
```

### Persistence

SQLite, Chroma, and uploaded files all live in the `rag-data` named volume. Inspect with:

```bash
docker volume ls
docker volume inspect rag_rag-data       # path on the host
```

Backup the volume to a tarball:

```bash
docker run --rm -v rag_rag-data:/data -v "$PWD":/backup alpine \
  tar czf /backup/rag-data-$(date +%F).tar.gz -C / data
```

(Restore: `docker run --rm -v rag_rag-data:/data -v "$PWD":/backup alpine tar xzf /backup/rag-data-YYYY-MM-DD.tar.gz -C /`.)

### Editing the LLM config

`./llm_config.json` is bind-mounted read-only into the container. Change it on the host and run:

```bash
docker compose restart rag
```

(The config is loaded once at startup and cached via `lru_cache`.)

### Switching to Postgres

A commented stanza is staged in `docker-compose.yml`. To activate:

1. Uncomment the `db:` service and the two `rag.environment` lines that set `RAG_DATABASE_URL` + `depends_on`.
2. Add `psycopg[binary]>=3.2` to `requirements.txt`.
3. `docker compose up -d --build`.

The existing SQLite data **does not** migrate automatically — wipe the relational rows (and re-index documents) or write a one-off migration script.

### Production hardening (recommended)

- **Reverse proxy** — terminate TLS with Caddy / nginx / Traefik on the VPS and proxy to `127.0.0.1:8765`. Don't expose 8765 directly with a real-world LLM key inside.
- **Bind to localhost** in `docker-compose.yml` once the proxy is in place: change `"8765:8765"` to `"127.0.0.1:8765:8765"`.
- **Auth** — this service has no authentication. Add one (basic auth at the proxy is the easiest) before allowing inbound traffic.
- **Secrets** — once `llm_config.json` holds a real OpenAI key, restrict its permissions (`chmod 600 llm_config.json`) and never commit it.

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

1. Open http://127.0.0.1:8765/ → **Datasets**
2. Create a dataset (e.g. `policies`)
3. Open the dataset and **drag files or a folder** onto the upload zone (or click to browse files, or use **Upload folder**). Each supported file gets its own status row: `pending → uploading → ok / skipped / error`. Nested paths inside a folder are kept (e.g. `docs/guide.md`).
4. Optionally, under **Google Drive**, paste a public folder link and click **Link folder** — the first sync runs automatically; use **Sync now** later to fetch new or changed files only.
5. Go to **Search**, pick the dataset, type a question
   - Click **Search** to get ranked passages with confidence badges
   - Click **Ask LLM** to additionally have the configured LLM generate a grounded answer that cites the passages it used
6. Each result card shows the passage, a colored confidence badge (`HIGH / MEDIUM / LOW · NN%`), and its reference (filename + chunk index + dataset). An overall verdict banner above the results summarizes the response confidence.

## REST API

### Datasets and documents

| Method | Path | Purpose |
|---|---|---|
| GET    | `/api/health`                                  | Liveness check |
| POST   | `/api/datasets`                                | Create a dataset `{name, description?}` |
| GET    | `/api/datasets`                                | List datasets |
| GET    | `/api/datasets/{id}`                           | Dataset details |
| DELETE | `/api/datasets/{id}`                           | Delete dataset (cascades chunks + files) |
| GET    | `/api/datasets/{id}/documents`                 | List documents in a dataset |
| POST   | `/api/datasets/{id}/documents/upload`          | Multipart upload + index (one file per call) |
| DELETE | `/api/datasets/{id}/documents/{doc_id}`        | Remove a document |

### Google Drive sync

| Method | Path | Purpose |
|---|---|---|
| POST   | `/api/datasets/{id}/drive/sources` | Link a public Drive folder/file URL `{url}` |
| GET    | `/api/datasets/{id}/drive/sources` | List linked Drive sources |
| GET    | `/api/datasets/{id}/drive/sources/{source_id}/files` | Cached remote file metadata (snapshots) |
| POST   | `/api/datasets/{id}/drive/sources/{source_id}/sync` | Incremental sync — list, compare metadata, index new/changed only |
| DELETE | `/api/datasets/{id}/drive/sources/{source_id}` | Remove link + snapshots (indexed documents stay) |

### Search, ask, and LLM

| Method | Path | Purpose |
|---|---|---|
| POST   | `/api/search`     | `{dataset_id, query, top_k}` → ranked hits + confidence + references |
| POST   | `/api/ask`        | `{dataset_id, query, top_k}` → retrieves first, then the LLM answers from those chunks (full RAG round trip) |
| POST   | `/api/llm/answer` | `{query, passages: [{filename, chunk_index, text}], ...}` → LLM answers from caller-supplied passages (no retrieval) |
| POST   | `/api/llm/chat`   | `{query, ...}` → raw LLM call, no context, no retrieval |
| GET    | `/api/llm/info`   | Current LLM provider, base URL, resolved model, defaults |

`top_k` is optional (defaults to `RAG_DEFAULT_TOP_K`). `temperature`, `model`, and `system_prompt` can be overridden per request on the LLM endpoints.

See `/docs` for the full interactive Swagger spec.

## Google Drive sync

Drive sync works with **public** folders shared as **Anyone with the link can view**. No Google user sign-in is required on the server, but you must configure a **Google Cloud API key** (free tier is fine).

### One-time API key setup

1. Open [Google Cloud Console](https://console.cloud.google.com/) → create or select a project
2. Enable **Google Drive API**
3. **Credentials** → **Create credentials** → **API key**
4. Restrict the key to the Drive API (recommended)
5. Add to `.env`:

```env
RAG_GOOGLE_API_KEY=your-api-key-here
```

6. Share the Drive folder as **Anyone with the link can view**

Restart the server after changing `.env`.

### Sync behavior

1. **List** — recursively walks the linked folder via Drive API v3
2. **Compare** — each remote file is matched by Google file ID; stored metadata includes `modifiedTime`, `md5Checksum` (when available), and `size`
3. **Skip** — unchanged files are not re-downloaded or re-embedded
4. **Index** — new or modified supported files are downloaded to `data/docs/{dataset_id}/drive/{source_id}/…` and indexed as `{root_name}/{relative_path}`

Supported from Drive: PDF, plain text, markdown, HTML, and native Google Docs (exported as `.txt`). Other types are listed in metadata but marked `unsupported`.

### Limitations (v1)

- Public links + API key only (no OAuth or service account)
- Manual **Sync now** only (no background scheduler)
- Removing a Drive link does **not** delete already-indexed documents
- Files removed from Drive are **not** auto-deleted from the dataset
- Native Google Sheets/Slides are not supported

## LLM configuration

The LLM is configured by a single JSON file at the project root: **`llm_config.json`**. Edit it to swap providers — no code change required.

Ships with **Ollama via an OpenAI-compatible endpoint**:

```json
{
  "provider": "ollama",
  "base_url": "https://your-ollama-host/v1",
  "api_key": "ollama",
  "model": null,
  "headers": { "ngrok-skip-browser-warning": "true" },
  "temperature": 0.1,
  "max_context_chars": 8000,
  "request_timeout_seconds": 120,
  "system_prompt": "..."
}
```

- `provider`: `"ollama"` or `"openai"`. For Ollama, if `model` is `null` the server auto-discovers the first model via `/api/tags`. For `"openai"` you must pin a model.
- `base_url`: the OpenAI-compatible chat endpoint (usually ends in `/v1`).
- `api_key`: any non-empty string for Ollama; a real key for OpenAI.
- `headers`: extra HTTP headers (e.g. `ngrok-skip-browser-warning` when fronting Ollama with ngrok).
- `temperature` / `system_prompt`: defaults the LLM endpoints use unless overridden per request.
- `max_context_chars`: cap on concatenated retrieved-passage characters in the prompt.

### Swap to real OpenAI

A ready template lives at **`llm_config.openai.json`**. To switch:

```powershell
Copy-Item .\llm_config.json .\llm_config.ollama.json    # back up the current config
Copy-Item -Force .\llm_config.openai.json .\llm_config.json
notepad .\llm_config.json                                # paste your sk-... key
# restart the server
```

To switch back: `Copy-Item -Force .\llm_config.ollama.json .\llm_config.json` and restart.

> If you `git init` this project, add `llm_config.json` to `.gitignore` once it contains a real key. Commit only the `*.openai.json` / `*.ollama.json` templates with placeholder keys.

## Configuration (env vars)

All app settings have defaults. Create a `.env` in the project root to override (prefix with `RAG_`):

```env
RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=120
RAG_DEFAULT_TOP_K=5

RAG_DATABASE_URL=sqlite:///./data/app.db
RAG_DOCS_DIR=data/docs
RAG_CHROMA_DIR=data/chroma
RAG_COLLECTION_NAME=documents

# Confidence calibration (tuned for all-MiniLM-L6-v2; see "Tuning confidence" below)
RAG_CONFIDENCE_FULL_SCORE=0.60
RAG_CONFIDENCE_HIGH_THRESHOLD=0.70
RAG_CONFIDENCE_MEDIUM_THRESHOLD=0.40

# Google Drive sync (see "Google Drive sync" section above)
RAG_GOOGLE_API_KEY=
RAG_DRIVE_REQUEST_TIMEOUT_SECONDS=60
```

To migrate to PostgreSQL later:

```env
RAG_DATABASE_URL=postgresql+psycopg://user:pass@host:5432/dbname
```

(Then `pip install "psycopg[binary]"`.)

### Tuning confidence

Raw cosine similarity from `all-MiniLM-L6-v2` rarely exceeds ~0.7 in practice. The calibration maps `cosine / RAG_CONFIDENCE_FULL_SCORE` clamped to `[0, 1]`, then labels:

- `confidence ≥ RAG_CONFIDENCE_HIGH_THRESHOLD`   → **high** (green)
- `confidence ≥ RAG_CONFIDENCE_MEDIUM_THRESHOLD` → **medium** (amber)
- otherwise                                       → **low** (red)

If your queries consistently produce 0.2–0.35 raw scores and you'd like those to read higher, lower `RAG_CONFIDENCE_FULL_SCORE` (e.g. to `0.35`) and re-tune the thresholds.

## Project layout

```
app/
  config.py        # pydantic-settings, env-driven
  db.py            # SQLAlchemy engine + session
  models.py        # ORM: Dataset, Document
  schemas.py       # Pydantic request/response (datasets, search, ask, llm)
  repository.py    # SQL CRUD helpers
  paths.py         # upload path sanitization
  drive/           # public Drive client + incremental sync
  loader.py        # file -> text (.txt/.md/.pdf/.html)
  chunker.py       # text -> overlapping windows
  embedder.py      # sentence-transformers wrapper (CPU, normalized)
  store.py         # ChromaDB wrapper, dataset-scoped queries
  indexer.py       # orchestration + SHA-256 change detection
  confidence.py    # score → confidence calibration + label + overall verdict
  retrieval.py     # shared retrieve() used by /api/search and /api/ask
  llm.py           # LLM config, OpenAI-SDK client, prompt builder, model auto-discovery
  deps.py          # FastAPI Depends singletons (embedder, store, indexer)
  main.py          # app factory + lifespan + router mounting
  routers/
    datasets.py    # /api/datasets/*
    drive.py       # /api/datasets/{id}/drive/*
    search.py      # /api/search
    ask.py         # /api/ask (retrieve + LLM)
    llm.py         # /api/llm/{info, answer, chat}
    ui.py          # /, /datasets, /search HTML pages
  templates/       # Jinja: base, datasets, dataset, search
  static/          # app.js, styles.css
data/
  app.db           # SQLite metadata (auto-created)
  chroma/          # vector index (auto-created)
  docs/<id>/       # uploaded files, isolated per dataset
llm_config.json         # active LLM provider config
llm_config.openai.json  # template for switching to real OpenAI
requirements.txt
.env.example
```

## Troubleshooting

- **Docker build is huge / pulls CUDA** — make sure you're using the provided `Dockerfile`; it installs CPU-only torch from `https://download.pytorch.org/whl/cpu` *before* `requirements.txt`. Default PyPI torch on Linux x86_64 is the CUDA build (~2.5 GB extra).
- **`Activate.ps1` not found** — your venv was created by a base Python (e.g. miniconda) that ships without activate templates. Delete `.venv` and recreate with `py -m venv .venv` from a standard CPython 3.12 install.
- **`WinError 10013` on port 8000** — Windows reserves a port range for Hyper-V. Use `--port 8765` (or any other free port).
- **First search/upload is slow** — the embedding model loads lazily on first use; subsequent calls are fast.
- **Re-upload returns `status: "skipped"`** — SHA-256 matched; the file content is unchanged. Edit (or rename) the file to force re-indexing.
- **`/api/ask` returns 502** — the LLM endpoint is unreachable. Check that `llm_config.json` points at a live server, the API key is correct, and (for ngrok-fronted Ollama) the `ngrok-skip-browser-warning` header is present. `GET /api/llm/info` is a quick health probe.
- **Drive sync returns 503** — set `RAG_GOOGLE_API_KEY` in `.env` and restart.
- **Drive sync returns 502 / 403** — confirm the folder is shared as **Anyone with the link can view** and the API key has Drive API enabled.
- **`llm_config.json` changes don't take effect** — the config is cached per process via `lru_cache`. Restart the server after edits.

## Notes

- **No authentication** — single-operator mode. Add an auth layer before exposing on a network.
- **Retrieval is independent of generation** — `/api/search` never calls an LLM. Generation is opt-in via `/api/ask`, `/api/llm/answer`, or `/api/llm/chat`. Confidence and references are computed for every retrieval response, so callers can decide what to do downstream.
- **One dataset per search/ask** in v1; multi-dataset is a small extension when needed.
