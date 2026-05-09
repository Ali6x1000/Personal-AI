"""
LlamaIndex + Local Ollama embeddings + ChromaDB ingestion for AliJR.

Run after adding documents under alijr_knowledge_base/.
Requires Ollama running locally (or set ``OLLAMA_BASE_URL``) with ``ollama pull <model>``.
Optional env: ``OLLAMA_EMBED_MODEL`` (default ``nomic-embed-text``), ``OLLAMA_BASE_URL``.
If you previously used another embedding backend, delete ``./chroma_db`` and re-index.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "alijr_knowledge_base"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "alijr_kb"


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


def main() -> None:
    load_dotenv()

    ollama_host = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model_name = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

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

    # Must match agent.py query embeddings: same model + base_url (+ instructions if set).
    embed_model = OllamaEmbedding(
        model_name=embed_model_name,
        base_url=ollama_host,
    )
    Settings.embed_model = embed_model

    vector_store = ChromaVectorStore.from_params(
        collection_name=COLLECTION_NAME,
        persist_dir=str(CHROMA_DB_DIR),
    )
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    if docs:
        VectorStoreIndex.from_documents(
            documents=docs,
            storage_context=storage_ctx,
            embed_model=embed_model,
            show_progress=True,
        )

    print(
        f"Indexed {len(docs)} document(s); "
        f"Ollama embeddings model={embed_model_name} host={ollama_host}; "
        f"Chroma at {CHROMA_DB_DIR} (collection={COLLECTION_NAME})."
    )


if __name__ == "__main__":
    main()