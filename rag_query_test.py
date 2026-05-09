"""Smoke-test Chroma retrieval; uses the same embedding config as ``rag_indexer.py`` (``EMBED_*`` env).

Run after ``rag_indexer.py``. Example::

    cd /path/to/project && source .venv/bin/activate
    python rag_query_test.py \"k-means clustering assignment\" --folder submissions --top-k 5
"""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag_indexer import CHROMA_DB_DIR, COLLECTION_NAME, _resolve_embedder

_FOLDER_CHOICES = ("all", "submissions", "research", "course_lectures", "projects")


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description="Print raw retrieved chunks from Chroma (RAG debug).")
    p.add_argument("query", help="Search query text")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument(
        "--folder",
        choices=_FOLDER_CHOICES,
        default="all",
        help="Restrict to indexer folder_category (default: no filter)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON object per node (scores, metadata, excerpt)",
    )
    args = p.parse_args()

    if not CHROMA_DB_DIR.exists():
        raise SystemExit(f"Missing {CHROMA_DB_DIR}; run rag_indexer.py first.")

    embed_model, _ = _resolve_embedder()
    Settings.embed_model = embed_model

    store = ChromaVectorStore.from_params(
        collection_name=COLLECTION_NAME,
        persist_dir=str(CHROMA_DB_DIR.resolve()),
    )
    index = VectorStoreIndex.from_vector_store(vector_store=store)

    kwargs: dict = {"similarity_top_k": max(1, args.top_k)}
    if args.folder != "all":
        kwargs["filters"] = MetadataFilters(
            filters=[MetadataFilter(key="folder_category", value=args.folder)],
        )

    nodes = index.as_retriever(**kwargs).retrieve(args.query)

    if not nodes:
        print("No chunks returned.")
        return

    if args.json:
        for n in nodes:
            md = getattr(n.node, "metadata", None) or {}
            body = (n.node.get_content(metadata_mode="none") or "").strip()
            print(
                json.dumps(
                    {"score": n.score, "metadata": md, "text_excerpt": body[:4000]},
                    indent=2,
                ),
                flush=True,
            )
        return

    for i, n in enumerate(nodes, start=1):
        md = getattr(n.node, "metadata", None) or {}
        path_hint = md.get("file_path") or md.get("filename") or md.get("document_id")
        body = (n.node.get_content(metadata_mode="none") or "").strip()
        preview = body[:1200] + ("…" if len(body) > 1200 else "")
        print(f"\n--- chunk {i} score={n.score} ---")
        print(f"path: {path_hint}")
        print(f"folder_category: {md.get('folder_category')}")
        print(preview)


if __name__ == "__main__":
    main()
