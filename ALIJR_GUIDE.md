# AliJR — detailed guide

This document expands on **`read.md`**: architecture, data flow, environment variables, GCP setup, ingestion behavior, troubleshooting, and deployment notes.

---

## Table of contents

1. [System architecture](#1-system-architecture)
2. [Component reference](#2-component-reference)
3. [RAG pipeline (ingestion & query)](#3-rag-pipeline-ingestion--query)
4. [Environment variables](#4-environment-variables)
5. [Google Cloud setup](#5-google-cloud-setup)
6. [Local development workflows](#6-local-development-workflows)
7. [Frontend integration](#7-frontend-integration)
8. [Operational notes](#8-operational-notes)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. System architecture

AliJR separates four concerns:

1. **Real-time voice transport** — LiveKit WebRTC rooms; the browser publishes microphone audio and receives TTS playback.
2. **Agent cognition** — A Python worker using `livekit-agents`: STT transcript → Gemini (Vertex) → optional tools → TTS streams.
3. **Knowledge retrieval** — LlamaIndex + Chroma, loaded from `./chroma_db`, exposed as a **`search_documents`** LiveKit tool.
4. **Access control surface** — A small FastAPI app mints JWT room tokens (`/token`) so the frontend never embeds LiveKit secrets.

High-level sequence:

```
Browser (Next.js + livekit-client)
    → POST /token (FastAPI + livekit-api)
    → Connect to LiveKit with JWT + NEXT_PUBLIC_LIVEKIT_URL

LiveKit dispatches worker job → agent.py entrypoint
    → STT: Google Cloud Speech
    → LLM: Gemini via Vertex (livekit-plugins-google)
    → Tool: KnowledgeBaseTools.search_documents → Chroma/LlamaIndex
    → TTS: Google Cloud Text-to-Speech
```

The **LlamaIndex** `Settings` (embeddings + configured `GoogleGenAI` chat model for consistency) mirror what indexing used for vectors; mismatched embedding backends break retrieval quality even if ingestion succeeded.

---

## 2. Component reference

### 2.1 `agent.py` (LiveKit worker)

- Loads `.env`, configures LlamaIndex `Settings` once per process.
- **`embed_provider`**: `ollama` (default) vs `gemini`: must match whatever built `chroma_db`. Vertex paths require `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT`, and optionally `GOOGLE_CLOUD_LOCATION`.
- **`KnowledgeBaseTools`**: Implements `search_documents(query, folder_category)` with metadata filter on `folder_category` unless `all`.
- **Voice pipeline**: `AgentSession(stt=google.STT, llm=google.LLM(..., vertexai=True, ...), tts=google.TTS(...), tools=[kb])`.
- **Worker tuning**: threaded job executor and single idle process defaults for predictable local laptops.

### 2.2 `rag_indexer.py`

- Walks **`alijr_knowledge_base/`** for `.pdf`, `.md`, `.markdown`, `.txt`.
- **`file_metadata`**: sets `folder_category` to the immediate subdirectory under the knowledge root (`submissions`, etc.).
- Persists **`chroma_db/`** plus LlamaIndex **docstore** for incremental **`refresh_ref_docs`**.
- **`embedding_manifest.json`**: records embedding backend + model identity; mismatches vs current env trigger a wipe + rebuild so vectors are never mixed across incompatible models.

### 2.3 `server.py` (FastAPI)

- **`POST /token`**: `{ room_name, participant_identity, participant_name? }` → `{ token, livekit_url, room, identity }`.
- Requires `LIVEKIT_*` and **verifies `GOOGLE_APPLICATION_CREDENTIALS` is set** (sanity gate; token mint itself is LiveKit-only).
- **CORS** currently allows localhost dev origins — tighten before public deployment.

### 2.4 `alijr-frontend/`

- Next.js App Router UI with `@livekit/components-react` (`LiveKitRoom`, audio, transcripts/visualizer patterns per your `page.tsx`).
- **`src/lib/token.ts`**: POSTs to `http://localhost:8000/token` — change base URL when hosting.

---

## 3. RAG pipeline (ingestion & query)

### 3.1 Ingestion

1. Documents are discovered by `SimpleDirectoryReader` (`filename_as_id=True` aids stable identities across reruns).
2. Each logical document produces nodes; embeddings are computed with the configured embed model (`ollama` or Vertex `GoogleGenAIEmbedding`).
3. Vectors land in collection **`alijr_kb`** under `./chroma_db`.

### 3.2 Incremental behavior

When `docstore.json` exists and the **embedding manifest matches**:

- Existing ref docs get **`refresh_ref_docs`**: LlamaIndex re-embeds only when underlying content/metadata changed relative to persisted state.
- Files removed from disk are **`delete_ref_doc`**’d from store + docstore.

When **`embedding_manifest.json` disagrees** with current `_resolve_embedder()` output (different provider/model/project/region), **`chroma_db` is recreated** intentionally — embedding spaces are incompatible.

### 3.3 Query behavior

Runtime retrieval uses **`as_retriever(similarity_top_k=8, filters=?)`**, concatenates snippet text (truncated excerpts), and passes the string back through the Gemini session inside LiveKit tools. Folder filters map **directly** to stored `folder_category`.

---

## 4. Environment variables

| Variable | Purpose |
|----------|---------|
| `LIVEKIT_URL` | LiveKit websocket URL (`wss://...`) returned to clients |
| `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | Sign room JWTs (`server.py`, worker bootstrap) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service-account JSON (`agent.py`, `rag_indexer` when using Vertex embeddings, enforced in `server.py`) |
| `GOOGLE_CLOUD_PROJECT` | Vertex / billing project (falls back to JSON `project_id` when omitting cautiously isn’t feasible — set explicitly for clarity) |
| `GOOGLE_CLOUD_LOCATION` | GCP region (`us-central1` default in code paths) |
| `GOOGLE_GENAI_USE_VERTEXAI` | Helps human operators + some SDK stacks; ingestion/agent codepaths already target Vertex embedding/LLM config |
| `EMBED_PROVIDER` | `ollama` or `gemini` (`rag_indexer.py` / `agent.py` must agree) |
| `OLLAMA_BASE_URL`, `OLLAMA_EMBED_MODEL` | When embedding provider is Ollama |
| `GEMINI_EMBED_MODEL` | Vertex embedding id (defaults include `text-embedding-004` in indexer/agent when using gemini branch) |
| `GOOGLE_TTS_MODEL` | LiveKit google TTS model name (`chirp_3` default in agent) |
| `LIVEKIT_AGENT_NAME` | Optional worker routing name |

**Never commit** `.env`, service-account JSON, or `./chroma_db` if they contain sensitive or large artifacts.

---

## 5. Google Cloud setup

### 5.1 Service account IAM

Grant the minimal roles consistent with Speech, Vertex Gemini, embeddings, and TTS APIs you intend to invoke. Typical starting points:

- Roles that grant **Speech-to-Text** and **Cloud Text-to-Speech** consumer access.
- **Vertex AI User** / model access as required by org policy.

### 5.2 APIs to enable (console)

Depending on SKU and model availability, enable APIs such as:

- Cloud Speech-to-Text (+ v2/streaming variants if prompted)
- Cloud Text-to-Speech
- Vertex AI API (`aiplatform.googleapis.com`)

### 5.3 Quotas & regions

Gemini Vertex models and embeddings are region-scoped — pick a region (`GOOGLE_CLOUD_LOCATION`) consistent with enabled models. STT/TTS default plugin locations (`global`) may still apply for those services; mismatches manifest as quota or `SERVICE_DISABLED`-style responses.

---

## 6. Local development workflows

### Bootstrap

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Index knowledge

```bash
python rag_indexer.py
```

Re-run whenever documents change meaningfully under `alijr_knowledge_base/`.

### Terminal layout (recommended)

1. Terminal A: `uvicorn server:app --reload --port 8000`
2. Terminal B: `python agent.py start`
3. Terminal C (frontend): `cd alijr-frontend && npm run dev`

Ensure **embedding provider** unchanged between indexer and worker between runs unless you deliberately rebuild `./chroma_db`.

---

## 7. Frontend integration

### Token contract

Frontend sends **POST `/token`** with JSON body `{ room_name, participant_identity, participant_name }`; response includes `livekit_url` so the SPA can omit `NEXT_PUBLIC_LIVEKIT_URL` if backend always returns it — your current client still prefers env fallback logic in `page.tsx`.

### HTTPS & mixed-origin

Browsers enforce secure contexts for media; production frontends behind HTTPS must use **HTTPS** token APIs or same-site deployment rules.

---

## 8. Operational notes

### 8.1 `chromadb-client` pitfall

`requirements.txt` documents: **do not install `chromadb-client`** alongside server `chromadb` — it can force HTTP-only client mode and break local `PersistentClient` paths.

### 8.2 Persistence & backups

Treat `./chroma_db` as rebuilt infrastructure: back up **`alijr_knowledge_base/`** source docs; you can regenerate vectors whenever needed.

### 8.3 Scaling ingestion

Heavy PDF corpora bottleneck on embedding throughput. Strategies:

- Use Vertex embeddings for parallelism vs laptop Ollama.
- Run indexer on CI or remote CPU with GPUs only if justified.
- Keep folder hygiene so agents filter aggressively at query-time.

---

## 9. Troubleshooting

| Symptom | Likely cause | Direction |
|---------|----------------|-----------|
| Empty / nonsense retrieval after switching embed models | Mixed vectors vs old index | Manifest mismatch triggers rebuild — or delete `./chroma_db` manually once |
| 401 / token errors joining room | Wrong LiveKit secrets or clocks | Rotate keys, validate `LIVEKIT_URL` |
| CORS preflight failures | Origin not listed | Extend `CORSMiddleware` allow_origins |
| STT/TTS permission errors | Service account IAM / disabled API | Cloud console API + roles |
| `SERVICE_DISABLED` on Google calls | Billing or API toggle | Enable service + quota |
| Agent never joins | Worker not registered / mismatched LiveKit projects | Align env between worker & token server |

For the concise version of this repo, refer to **`read.md`**.
