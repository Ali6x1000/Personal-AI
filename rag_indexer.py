"""
LlamaIndex + Local Ollama embeddings + ChromaDB ingestion for AliJR.

Run after adding documents under alijr_knowledge_base/.
Install ``llama-index-readers-file`` (listed in ``requirements.txt``): without it,
``.pdf`` files are read as raw bytes interpreted as UTF-8 and indexing produces garbage chunks.
Requires Ollama running locally (or set ``OLLAMA_BASE_URL``) with ``ollama pull <model>``.
Optional env: ``OLLAMA_EMBED_MODEL`` (default ``nomic-embed-text``), ``OLLAMA_BASE_URL``.
If you previously used another embedding backend, delete ``./chroma_db`` and re-index.

Incremental runs skip unchanged files using ``./chroma_db/index_file_state.json`` (mtime/size per
indexed document id). Embedding throughput: ``RAG_EMBED_BATCH_SIZE`` (default 24),
``RAG_EMBED_NUM_WORKERS`` (async concurrent sub-requests), ``RAG_USE_ASYNC_EMBED`` (default 1).
Vertex enforces ~20k **tokens per embedding HTTP request**: use ``RAG_EMBED_MAX_CHARS_PER_REQUEST``
(default 26000) so large batches split safely. Gemini retries/backoff:
``RAG_EMBED_RETRIES``, ``RAG_EMBED_RETRY_MIN_SEC``, ``RAG_EMBED_RETRY_MAX_SEC``,
``RAG_EMBED_RETRY_EXP_BASE``, ``RAG_EMBED_TIMEOUT_SEC``. Set ``RAG_QUIET_PYPDF=0`` for verbose PDF parser logs.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, ClassVar, Sequence

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from google.oauth2 import service_account as sa_credentials
from llama_index.core import SimpleDirectoryReader, Settings, StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.ingestion import run_transformations
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "alijr_knowledge_base"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "alijr_kb"
DOCSTORE_PATH = CHROMA_DB_DIR / "docstore.json"
EMBED_MANIFEST_PATH = CHROMA_DB_DIR / "embedding_manifest.json"
# Tracks {doc_id(path): {"mtime_ns", "size"}} for incremental runs (Chroma ``stores_text`` → no ``ref_doc_info``).
INDEX_FILE_STATE_PATH = CHROMA_DB_DIR / "index_file_state.json"

# Vertex token exchange requires OAuth scopes on the service account credential.
GCP_VERTEX_SCOPES: tuple[str, ...] = ("https://www.googleapis.com/auth/cloud-platform",)


def _maybe_configure_rag_logging() -> None:
    """Downgrade noisy upstream PDF parsers during bulk indexing."""

    if os.getenv("RAG_QUIET_PYPDF", "1").strip().lower() in ("1", "true", "yes"):
        for name in ("pypdf", "pypdf._reader", "pypdf.generic"):
            logging.getLogger(name).setLevel(logging.ERROR)


def _vertex_gemini_retry_fields() -> dict[str, Any]:
    return {
        "retries": int(os.getenv("RAG_EMBED_RETRIES", "8")),
        "timeout": int(os.getenv("RAG_EMBED_TIMEOUT_SEC", "180")),
        "retry_min_seconds": float(os.getenv("RAG_EMBED_RETRY_MIN_SEC", "2")),
        "retry_max_seconds": float(os.getenv("RAG_EMBED_RETRY_MAX_SEC", "120")),
        "retry_exponential_base": float(os.getenv("RAG_EMBED_RETRY_EXP_BASE", "2")),
    }


class VertexEmbeddingRequestBudget(BaseEmbedding):
    """Split embedding ``texts`` so each Vertex HTTP call stays below ~20k input tokens."""

    _inner: BaseEmbedding = PrivateAttr()
    _max_chars_budget: int = PrivateAttr()

    _trunc_logged: ClassVar[bool] = False

    def __init__(self, inner: BaseEmbedding, *, max_chars_per_request: int) -> None:
        mc = max(4096, int(max_chars_per_request))
        ibs_raw = getattr(inner, "embed_batch_size", DEFAULT_EMBED_BATCH_SIZE)
        ibs = max(1, min(int(ibs_raw) if ibs_raw is not None else DEFAULT_EMBED_BATCH_SIZE, 2048))
        super().__init__(
            model_name=getattr(inner, "model_name", "unknown"),
            embed_batch_size=ibs,
            num_workers=getattr(inner, "num_workers", None),
            callback_manager=getattr(inner, "callback_manager", None),
            embeddings_cache=getattr(inner, "embeddings_cache", None),
            rate_limiter=getattr(inner, "rate_limiter", None),
        )
        self._inner = inner
        self._max_chars_budget = mc

    @property
    def vertex_char_budget(self) -> int:
        return self._max_chars_budget

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._inner._get_query_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await self._inner._aget_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._inner._get_text_embedding(text)

    @classmethod
    def _truncate(cls, text: str, lim: int) -> str:
        if len(text) <= lim:
            return text
        if not cls._trunc_logged:
            logger.warning(
                "At least one embedding input exceeded RAG_EMBED_MAX_CHARS_PER_REQUEST; "
                "truncating oversized segments (prefer smaller node chunks in Settings.transformations)."
            )
            cls._trunc_logged = True
        return text[:lim]

    @classmethod
    def _partition_batches(cls, texts: Sequence[str], max_chars: int) -> list[list[str]]:
        buckets: list[list[str]] = []
        cur: list[str] = []
        run = 0
        for t in texts:
            if not t:
                t = ""
            piece = cls._truncate(str(t), max_chars)
            lp = len(piece)
            if lp > max_chars:
                buckets.append([piece])
                continue
            if cur and run + lp > max_chars:
                buckets.append(cur)
                cur = []
                run = 0
            cur.append(piece)
            run += lp
        if cur:
            buckets.append(cur)
        return buckets

    def get_text_embedding_batch(
        self,
        texts: list[str],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[list[float]]:
        max_chars = self._max_chars_budget
        parts = self._partition_batches(texts, max_chars)
        if len(parts) == 1:
            return self._inner.get_text_embedding_batch(
                parts[0], show_progress=show_progress, **kwargs
            )
        merged: list[list[float]] = []
        for chunk in parts:
            merged.extend(
                self._inner.get_text_embedding_batch(chunk, show_progress=False, **kwargs)
            )
        return merged

    async def aget_text_embedding_batch(
        self,
        texts: list[str],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[list[float]]:
        max_chars = self._max_chars_budget
        parts = self._partition_batches(texts, max_chars)
        if len(parts) == 1:
            return await self._inner.aget_text_embedding_batch(
                parts[0], show_progress=show_progress, **kwargs
            )
        merged: list[list[float]] = []
        for chunk in parts:
            merged.extend(
                await self._inner.aget_text_embedding_batch(chunk, show_progress=False, **kwargs)
            )
        return merged


def _vertex_service_account_credentials(credentials_file: str) -> sa_credentials.Credentials:
    path = Path(credentials_file).expanduser()
    return sa_credentials.Credentials.from_service_account_file(
        str(path),
        scopes=list(GCP_VERTEX_SCOPES),
    )


def _chromadb_document_ids(collection_name: str, persist_dir: Path) -> set[str]:
    """Return distinct ``document_id`` / ref-doc keys stored in Chroma (LlamaIndex metadata)."""

    import chromadb

    ids: set[str] = set()
    try:
        client = chromadb.PersistentClient(path=str(persist_dir))
        coll = client.get_collection(collection_name)
    except Exception:
        return ids
    try:
        batch = coll.get(include=["metadatas"])
    except Exception:
        return ids
    for md in batch.get("metadatas") or []:
        if not md:
            continue
        did = md.get("document_id") or md.get("ref_doc_id") or md.get("doc_id")
        if did not in (None, "", "None"):
            ids.add(str(did))
    return ids


def _file_fingerprint(path_str: str) -> dict[str, int]:
    path = Path(path_str).expanduser().resolve(strict=False)
    st = path.stat()
    return {"mtime_ns": st.st_mtime_ns, "size": st.st_size}


_PART_SUFFIX = re.compile(r"_part_\d+$")


def _fingerprint_src_path(doc) -> Path:
    """Resolve on-disk PDF/Markdown file for fingerprints (handles ``…pdf_part_N`` Llama IDs)."""

    meta = getattr(doc, "metadata", None) or {}
    fp = meta.get("file_path")
    if fp:
        return Path(str(fp)).expanduser().resolve(strict=False)
    did = str(getattr(doc, "id_", ""))
    if _PART_SUFFIX.search(did):
        return Path(_PART_SUFFIX.sub("", did)).expanduser().resolve(strict=False)
    return Path(did).expanduser().resolve(strict=False)


def _apply_embed_throughput_settings(
    embed_model: Any,
    *,
    provider: str,
) -> tuple[bool, int]:
    """Tune batching / async embedding; returns (use_async_embed, insert_batch_size)."""

    batch = int(os.environ.get("RAG_EMBED_BATCH_SIZE", "24"))
    # Large ``embed_batch_size`` + parallel workers blows Vertex's ~20k-token *per request*
    # cap; VertexBudget wrapper handles splits, smaller batches still ease memory.
    vertex_cap = int(os.getenv("RAG_VERTEX_EMBED_BATCH_CAP", "48"))
    if provider == "gemini":
        batch = max(1, min(batch, max(4, vertex_cap)))
    else:
        batch = max(1, min(batch, 256))
    workers = max(1, int(os.environ.get("RAG_EMBED_NUM_WORKERS", "4")))
    use_async = os.environ.get("RAG_USE_ASYNC_EMBED", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    insert_batch = int(os.environ.get("RAG_INSERT_BATCH_SIZE", "2048"))
    insert_batch = max(512, min(insert_batch, 8192))

    proxy = getattr(embed_model, "_inner", embed_model)

    if hasattr(proxy, "embed_batch_size"):
        try:
            proxy.embed_batch_size = batch  # type: ignore[misc]
        except Exception:
            pass
    if hasattr(proxy, "num_workers"):
        try:
            proxy.num_workers = workers  # type: ignore[misc]
        except Exception:
            pass

    if workers > 1 or batch >= 8:
        extras = ""
        mc = getattr(embed_model, "vertex_char_budget", None)
        if mc is not None:
            extras = f"; vertex_char_cap_per_http_request={mc}"
        print(
            f"Embed throughput: batch_size={batch} num_workers={workers} "
            f"use_async={use_async} insert_batch_size={insert_batch}{extras}",
            flush=True,
        )
    return use_async, insert_batch


def file_metadata(filepath: str) -> dict[str, Any]:
    """Attach file path/name and immediate subfolder under the knowledge root as folder_category."""

    path = Path(filepath).expanduser().resolve(strict=False)
    root = KNOWLEDGE_BASE_DIR.resolve()
    category = "unknown"
    try:
        rel = path.relative_to(root)
        if len(rel.parts) >= 2:
            category = rel.parts[0]
        elif len(rel.parts) == 1 and root in path.parents:
            category = root.name
    except ValueError:
        pass

    return {
        "file_path": str(path),
        "file_name": path.name,
        "folder_category": category,
    }


def _resolve_embedder() -> tuple[Any, dict[str, Any]]:
    provider = os.environ.get("EMBED_PROVIDER", "ollama").strip().lower()
    print(f"EMBED_PROVIDER: {provider}")
    if provider == "ollama":
        ollama_host = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        embed_model = OllamaEmbedding(model_name=model, base_url=ollama_host)
        descriptor = {
            "provider": "ollama",
            "model": model,
            "base_url": ollama_host,
        }
        return embed_model, descriptor

    if provider == "gemini":
        try:
            from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
        except Exception as exc:
            raise RuntimeError(
                "EMBED_PROVIDER=gemini requires `llama-index-embeddings-google-genai`."
            ) from exc

        model = os.environ.get("GEMINI_EMBED_MODEL", "text-embedding-004")
        credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_file:
            raise RuntimeError(
                "EMBED_PROVIDER=gemini requires GOOGLE_APPLICATION_CREDENTIALS."
            )
        credentials = _vertex_service_account_credentials(credentials_file)
        qp = os.environ.get("GOOGLE_CLOUD_QUOTA_PROJECT")
        if qp:
            credentials = credentials.with_quota_project(qp)
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or getattr(
            credentials, "project_id", None
        )
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not project:
            raise RuntimeError(
                "EMBED_PROVIDER=gemini requires GOOGLE_CLOUD_PROJECT "
                "(or a service-account JSON that includes project_id)."
            )
        # Prefer explicit Vertex parameters when supported by the installed package.
        # Fallback to vertexai_config for versions that only support that signature.
        retries = _vertex_gemini_retry_fields()
        max_chars_rq = int(os.getenv("RAG_EMBED_MAX_CHARS_PER_REQUEST", "26000"))
        try:
            embed_model = GoogleGenAIEmbedding(
                model_name=model,
                vertexai=True,
                project=project,
                location=location,
                **retries,
            )
        except TypeError:
            embed_model = GoogleGenAIEmbedding(
                model_name=model,
                vertexai_config={
                    "project": project,
                    "location": location,
                    "credentials": credentials,
                },
                **retries,
            )
        embed_model = VertexEmbeddingRequestBudget(embed_model, max_chars_per_request=max_chars_rq)
        # Keep descriptor stable (model + Vertex target only) so indexing env knobs
        # do not invalidate ``embedding_manifest.json`` and wipe Chroma unnecessarily.
        descriptor = {
            "provider": "gemini",
            "model": model,
            "vertex_ai": True,
            "project": project,
            "location": location,
        }
        return embed_model, descriptor

    raise RuntimeError(
        f"Unsupported EMBED_PROVIDER={provider!r}. Expected 'ollama' or 'gemini'."
    )


def main() -> None:
    load_dotenv()
    _maybe_configure_rag_logging()
    if ".pdf" not in SimpleDirectoryReader.supported_suffix_fn():
        print(
            "WARN: PDF reader unavailable (install llama-index-readers-file + pypdf). "
            "PDFs would be corrupted if indexed.",
            flush=True,
        )

    embed_model, embed_descriptor = _resolve_embedder()
    provider_key = embed_descriptor.get("provider", "").lower().strip()
    use_async_embed, insert_batch_size = _apply_embed_throughput_settings(
        embed_model,
        provider=provider_key or os.environ.get("EMBED_PROVIDER", "ollama").strip().lower(),
    )

    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

    if EMBED_MANIFEST_PATH.exists():
        try:
            existing_manifest = json.loads(EMBED_MANIFEST_PATH.read_text())
        except Exception:
            existing_manifest = None
        if existing_manifest != embed_descriptor:
            print("Embedding configuration changed; rebuilding Chroma index from scratch.")
            shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)
            CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

    reader = SimpleDirectoryReader(
        input_dir=str(KNOWLEDGE_BASE_DIR),
        recursive=True,
        filename_as_id=True,
        required_exts=[".pdf", ".md", ".markdown", ".txt"],
        exclude_hidden=True,
        file_metadata=file_metadata,
        errors="ignore",
    )
    docs = reader.load_data()
    if not docs:
        print(f"No supported files found under {KNOWLEDGE_BASE_DIR}")
        print("Add PDF/Markdown/text files to subfolders, then re-run.")
        return

    # Must match agent.py query embeddings, or retrieval quality will degrade.
    Settings.embed_model = embed_model

    vector_store = ChromaVectorStore.from_params(
        collection_name=COLLECTION_NAME,
        persist_dir=str(CHROMA_DB_DIR),
    )

    force_full = os.environ.get("RAG_FULL_REBUILD", "").lower() in ("1", "true", "yes")
    if force_full:
        print("RAG_FULL_REBUILD=1: wiping Chroma and rebuilding from scratch.")
        shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        vector_store = ChromaVectorStore.from_params(
            collection_name=COLLECTION_NAME,
            persist_dir=str(CHROMA_DB_DIR),
        )

    fingerprints: dict[str, dict[str, int]] = {}
    for doc in docs:
        fingerprints[str(doc.id_)] = _file_fingerprint(str(_fingerprint_src_path(doc)))

    chroma_refs = _chromadb_document_ids(COLLECTION_NAME, CHROMA_DB_DIR)
    current_doc_ids = {str(d.id_) for d in docs}

    prev_fp: dict[str, Any] = {}
    if INDEX_FILE_STATE_PATH.exists():
        try:
            prev_fp = json.loads(INDEX_FILE_STATE_PATH.read_text())
        except Exception:
            prev_fp = {}

    take_incremental_path = DOCSTORE_PATH.exists() and chroma_refs and docs and not force_full

    if not take_incremental_path:
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(
            documents=docs,
            storage_context=storage_ctx,
            embed_model=embed_model,
            show_progress=True,
            use_async=use_async_embed,
            insert_batch_size=insert_batch_size,
        )
        storage_ctx.persist(persist_dir=str(CHROMA_DB_DIR))
        upserted = len(docs)
        deleted = 0
        if DOCSTORE_PATH.exists() and not chroma_refs and not force_full:
            print(
                "(Note: indexer performed a full ingest (no prior Chroma vectors or pairing). "
                "Future runs should incremental-sync via index_file_state.json.)"
            )
    else:
        storage_ctx = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(CHROMA_DB_DIR),
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_ctx,
            embed_model=embed_model,
            show_progress=True,
            insert_batch_size=insert_batch_size,
        )

        deleted = 0
        for rid in sorted(chroma_refs - current_doc_ids):
            try:
                index.delete_ref_doc(rid, delete_from_docstore=False)
                deleted += 1
            except Exception as exc:
                print(f"Warn: failed deleting removed doc vectors for {rid!r}: {exc}")

        docs_to_refresh: list = []
        if not prev_fp:
            docs_to_refresh.extend(d for d in docs if str(d.id_) not in chroma_refs)
        else:
            for doc in docs:
                cid = str(doc.id_)
                if cid not in prev_fp:
                    docs_to_refresh.append(doc)
                    continue
                if prev_fp.get(cid) != fingerprints.get(cid):
                    docs_to_refresh.append(doc)

        upserted = len(docs_to_refresh)
        transforms = Settings.transformations
        staged_nodes: list = []
        for doc in docs_to_refresh:
            cid = str(doc.id_)
            if cid in chroma_refs:
                try:
                    index.delete_ref_doc(cid, delete_from_docstore=False)
                except Exception as exc:
                    print(f"Warn: pre-replace delete {cid!r}: {exc}")
            staged_nodes.extend(
                run_transformations([doc], transforms, show_progress=False)
            )

        if staged_nodes:
            index.insert_nodes(staged_nodes, show_progress=True)
            for doc in docs_to_refresh:
                index.docstore.set_document_hash(doc.id_, doc.hash)
        elif take_incremental_path:
            print("No embedding work: fingerprints and Chroma are already in sync.", flush=True)

        storage_ctx.persist(persist_dir=str(CHROMA_DB_DIR))

    INDEX_FILE_STATE_PATH.write_text(json.dumps(fingerprints, indent=2))
    EMBED_MANIFEST_PATH.write_text(json.dumps(embed_descriptor, indent=2))

    print(
        f"Indexed total={len(docs)} upserted={upserted} deleted={deleted}; "
        f"embed={embed_descriptor}; "
        f"Chroma at {CHROMA_DB_DIR} (collection={COLLECTION_NAME}); "
        f"incremental={'yes' if take_incremental_path else 'full'}."
    )


if __name__ == "__main__":
    main()