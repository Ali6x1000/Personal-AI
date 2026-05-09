"""
LlamaIndex + Local Ollama embeddings + ChromaDB ingestion for AliJR.

Run after adding documents under alijr_knowledge_base/.
Requires Ollama running locally (or set ``OLLAMA_BASE_URL``) with ``ollama pull <model>``.
Optional env: ``OLLAMA_EMBED_MODEL`` (default ``nomic-embed-text``), ``OLLAMA_BASE_URL``.
If you previously used another embedding backend, delete ``./chroma_db`` and re-index.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.oauth2 import service_account as sa_credentials
from llama_index.core import SimpleDirectoryReader, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "alijr_knowledge_base"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "alijr_kb"
DOCSTORE_PATH = CHROMA_DB_DIR / "docstore.json"
EMBED_MANIFEST_PATH = CHROMA_DB_DIR / "embedding_manifest.json"

# Vertex token exchange requires OAuth scopes on the service account credential.
GCP_VERTEX_SCOPES: tuple[str, ...] = ("https://www.googleapis.com/auth/cloud-platform",)


def _vertex_service_account_credentials(credentials_file: str) -> sa_credentials.Credentials:
    path = Path(credentials_file).expanduser()
    return sa_credentials.Credentials.from_service_account_file(
        str(path),
        scopes=list(GCP_VERTEX_SCOPES),
    )


def file_metadata(filepath: str) -> dict[str, Any]:
    """Attach immediate subfolder under the knowledge root as folder_category."""
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

    return {"folder_category": category}


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
        try:
            embed_model = GoogleGenAIEmbedding(
                model_name=model,
                vertexai=True,
                project=project,
                location=location,
            )
        except TypeError:
            embed_model = GoogleGenAIEmbedding(
                model_name=model,
                vertexai_config={
                    "project": project,
                    "location": location,
                    "credentials": credentials,
                },
            )
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
    embed_model, embed_descriptor = _resolve_embedder()

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
    # Persist docstore/index metadata alongside Chroma so we can refresh incrementally.
    if DOCSTORE_PATH.exists():
        storage_ctx = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(CHROMA_DB_DIR),
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_ctx,
            embed_model=embed_model,
        )
        existing_ref_doc_ids = set(index.ref_doc_info.keys())
    else:
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
        index = None
        existing_ref_doc_ids = set()

    current_ref_doc_ids = {doc.id_ for doc in docs}

    if index is None or not existing_ref_doc_ids:
        # First run (or no persisted doc metadata): build from scratch.
        index = VectorStoreIndex.from_documents(
            documents=docs,
            storage_context=storage_ctx,
            embed_model=embed_model,
            show_progress=True,
        )
        upserted = len(docs)
        deleted = 0
    else:
        refreshed_flags = index.refresh_ref_docs(docs)
        upserted = sum(1 for changed in refreshed_flags if changed)

        removed_ids = sorted(existing_ref_doc_ids - current_ref_doc_ids)
        for ref_doc_id in removed_ids:
            index.delete_ref_doc(ref_doc_id, delete_from_docstore=True)
        deleted = len(removed_ids)

    storage_ctx.persist(persist_dir=str(CHROMA_DB_DIR))
    EMBED_MANIFEST_PATH.write_text(json.dumps(embed_descriptor, indent=2))

    print(
        f"Indexed total={len(docs)} upserted={upserted} deleted={deleted}; "
        f"embed={embed_descriptor}; "
        f"Chroma at {CHROMA_DB_DIR} (collection={COLLECTION_NAME})."
    )


if __name__ == "__main__":
    main()