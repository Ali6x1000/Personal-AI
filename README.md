# AliJR — short overview

## Design (how it works end-to-end)

AliJR is a **voice-first assistant** backed by LiveKit Cloud: the user speaks in the browser (Next.js client), joins a LiveKit room, and a Python **livekit-agents worker** attaches to that room job. Speech is decoded with Google Cloud Speech-to-Text (**STT**), interpreted and answered by a **Gemini** model routed through **Vertex AI**, then spoken back with Google Cloud Text-to-Speech (**TTS**, model configurable).

For **private knowledge**, documents live under `alijr_knowledge_base/` grouped by folder (`submissions`, `research`, `course_lectures`, `projects`). `rag_indexer.py` ingests PDF/Markdown/text into a local **Chroma** vector database (`./chroma_db`). At runtime, the agent exposes a **`search_documents`** tool that queries Chroma **with optional `folder_category` filters** so the model can retrieve only relevant material. Retrieved chunks are returned to the Gemini session through the LiveKit pipeline for grounding and reply.

Authentication for Google-facing pieces uses **GCP service-account JSON** via `GOOGLE_APPLICATION_CREDENTIALS` (no Google AI Studio API key in this project). LlamaIndex is configured so Vertex-capable embeddings and chat models align with GCP project and region.

---

## How RAG is integrated

1. **Ingestion** (`rag_indexer.py`): `SimpleDirectoryReader` loads files recursively; each file gets metadata `folder_category` from its top-level subdirectory. Text is chunked and embedded (**Ollama** or **Vertex `GoogleGenAIEmbedding`**, selectable with `EMBED_PROVIDER`). Vectors persist in **Chroma** with a LlamaIndex **docstore** for incremental refreshes.

2. **Query time** (`agent.py`): LlamaIndex loads the same collection. The `KnowledgeBaseTools` tool builds a vector retriever, applies `MetadataFilters` when `folder_category` isn’t `"all"`, and returns concatenated excerpts to the conversational model.

Indexing is **incremental** when embeddings config is unchanged (`refresh_ref_docs`, deletes for removed paths). If embedding settings change, the indexer can rebuild safely using `embedding_manifest.json`.

---

## Tools and frameworks

| Area | Choice |
|------|--------|
| Real-time audio / rooms | LiveKit (`livekit-client`, LiveKit Cloud) |
| Python voice agent | `livekit-agents`, `livekit-plugins-google` |
| Knowledge indexing & retrieval | `llama-index`, `llama-index-vector-stores-chroma` |
| Vector store | ChromaDB (local persist under `./chroma_db`) |
| Embeddings | `EMBED_PROVIDER=ollama` → `llama-index-embeddings-ollama`; `gemini` → `llama-index-embeddings-google-genai` + Vertex |
| LlamaIndex LLM (configured for consistency) | `llama-index-llms-google-genai` (`GoogleGenAI`, Vertex) |
| Token API | FastAPI + `livekit-api` |
| Frontend | Next.js (App Router), Tailwind CSS, `@livekit/components-react` |

---

## Setup — run locally

**Prerequisites:** Python **3.10+**, Node **18+**, a **LiveKit** project (`LIVEKIT_URL`, keys), a GCP project with Speech, TTS, and Vertex Gemini/embedding APIs enabled, and a **service-account JSON** key.

### Backend

From the repo root (where `requirements.txt`, `agent.py`, and `server.py` live):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` in the **same directory** as `agent.py` (minimal example):

```
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...

GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=true

# RAG embeddings: "ollama" (local) or "gemini" (Vertex text-embedding-004 by default via GEMINI_EMBED_MODEL)
EMBED_PROVIDER=gemini

# Voice TTS Cloud model override (LiveKit google plugin), e.g. chirp_3
GOOGLE_TTS_MODEL=chirp_3
```

If using **Ollama** for embeddings instead: start Ollama, `ollama pull nomic-embed-text`, then set `EMBED_PROVIDER=ollama`, `OLLAMA_BASE_URL=http://localhost:11434`.

**Build the knowledge index** (after adding PDFs/Markdown under `alijr_knowledge_base/`):

```bash
python rag_indexer.py
```

**Token server** (for the React app):

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

**Agent worker** (connects to LiveKit and waits for rooms/jobs):

```bash
python agent.py start
# For iterative dev only: python agent.py dev (watch/reload semantics differ)
```

### Frontend

```bash
cd alijr-frontend
npm install
```

Configure environment (e.g. `.env.local`):

```
NEXT_PUBLIC_LIVEKIT_URL=wss://your-project.livekit.cloud
```

Ensure `src/lib/token.ts` points at your token API (default `http://localhost:8000/token` POST). Run:

```bash
npm run dev
```

Open the printed local URL (usually `http://localhost:3000`), start a call, and verify the worker logs show activity.

---

## Setup — toward the web

For production-quality deployment:

- Run **uvicorn/server** behind HTTPS (or expose only on a private network behind your auth).
- **Restrict CORS** in `server.py` to your deployed frontend origins instead of localhost only.
- Run **one or more stable agent workers** (systemd, Docker, Kubernetes, or LiveKit’s recommended worker hosting).
- Prefer **matching** frontend `NEXT_PUBLIC_LIVEKIT_URL` and backend `LIVEKIT_URL`, and redeploy **`chroma_db`** or index on the machine that runs retrieval.

For deeper architecture, troubleshooting, and API notes see **`ALIJR_GUIDE.md`**.
