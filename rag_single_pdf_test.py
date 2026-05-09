"""Index exactly one PDF into a disposable Chroma dir and sanity-check retrieval (no touching main chroma_db).

Uses the same embedder wiring as rag_indexer (``EMBED_*`` / Vertex / Ollama from ``.env``).

Examples::

    source .venv/bin/activate && python rag_single_pdf_test.py
    python rag_single_pdf_test.py --pdf \"course_lectures/CSDS 341/Extracted_Files/SP26_A4.pdf\"
    python rag_single_pdf_test.py --query \"assignment due date\" --keep-dir /tmp/pdf-rag-debug
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag_indexer import KNOWLEDGE_BASE_DIR, file_metadata, _resolve_embedder

_COLLECTION = "alijr_single_pdf_smoke"


def _letter_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(1 for c in text if c.isalpha())
    return letters / len(text)


def _default_pdf_under_kb() -> str:
    root = KNOWLEDGE_BASE_DIR.resolve()
    pdfs = sorted(root.rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs under {root}")
    choice = pdfs[0]
    print(f"(Using first PDF by sort order: {choice.relative_to(root)})")
    return str(choice.resolve())


def _resolve_pdf_arg(path_arg: str | None) -> str:
    if not path_arg:
        return _default_pdf_under_kb()
    p = path_arg.expanduser().resolve(strict=False)
    if not p.is_file():
        under_kb = KNOWLEDGE_BASE_DIR / path_arg
        if under_kb.is_file():
            p = under_kb.resolve()
        else:
            raise SystemExit(f"PDF not found: {path_arg}")
    if p.suffix.lower() != ".pdf":
        raise SystemExit(f"Expected a .pdf file, got: {p}")
    return str(p)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Disposable single-PDF index + retrieval smoke test.")
    ap.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="PDF path (absolute or relative to cwd); default: first *.pdf under alijr_knowledge_base/",
    )
    ap.add_argument(
        "--query",
        default="introduction assignment course",
        help="Retrieval smoke query after indexing",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="similarity_top_k for retriever",
    )
    ap.add_argument(
        "--keep-dir",
        type=Path,
        default=None,
        help="If set, persist Chroma here and do not delete (for inspection)",
    )
    args = ap.parse_args()

    if ".pdf" not in SimpleDirectoryReader.supported_suffix_fn():
        raise SystemExit(
            "PDF reader unavailable. Install packages from requirements.txt "
            "(llama-index-readers-file + pypdf)."
        )

    pdf_path = _resolve_pdf_arg(args.pdf)
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

    reader = SimpleDirectoryReader(
        input_files=[pdf_path],
        filename_as_id=True,
        file_metadata=file_metadata,
        errors="raise",
        raise_on_error=True,
    )
    docs = reader.load_data(show_progress=False)
    if not docs:
        raise SystemExit("Reader returned zero documents.")

    sample = docs[0].text or ""
    lr = _letter_ratio(sample)
    print(f"Loaded {len(docs)} PDF segment(s); first segment letter-ratio={lr:.3f}")

    embed_model, _ = _resolve_embedder()
    Settings.embed_model = embed_model

    keep_flag = args.keep_dir is not None
    if keep_flag:
        work_dir = args.keep_dir.expanduser().resolve(strict=False)
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="alijr_single_pdf_chroma_"))

    print(f"Chroma staging dir: {work_dir}")

    vs = ChromaVectorStore.from_params(
        collection_name=_COLLECTION,
        persist_dir=str(work_dir),
    )
    ctx = StorageContext.from_defaults(vector_store=vs)
    VectorStoreIndex.from_documents(
        documents=list(docs),
        storage_context=ctx,
        embed_model=embed_model,
        show_progress=True,
    )
    ctx.persist(persist_dir=str(work_dir))

    reload_vs = ChromaVectorStore.from_params(
        collection_name=_COLLECTION,
        persist_dir=str(work_dir),
    )
    index = VectorStoreIndex.from_vector_store(vector_store=reload_vs)
    nodes = index.as_retriever(similarity_top_k=max(1, args.top_k)).retrieve(args.query)

    print(f"\nQuery: {args.query!r}  →  {len(nodes)} node(s)\n")
    best_ratio = 0.0
    for i, n in enumerate(nodes, start=1):
        raw = (n.node.get_content(metadata_mode="none") or "").strip()
        md = getattr(n.node, "metadata", None) or {}
        path_hint = md.get("file_path") or md.get("filename") or md.get("file_name")
        ratio = _letter_ratio(raw)
        best_ratio = max(best_ratio, ratio)
        preview = raw[:500].replace("\n", " ")
        if len(raw) > 500:
            preview += "…"
        print(f"--- chunk {i} score={n.score} letter_ratio={ratio:.3f} ---")
        print(f"path: {path_hint}")
        print(preview)
        print()

    if best_ratio < 0.12:
        raise SystemExit(
            "FAIL: retrieved text looks like binary/noise (low letter ratio). "
            "If the PDF is scanned, you need OCR; otherwise check PDFReader/pypdf."
        )

    print("PASS: retrieved chunks look like readable text.")
    if not keep_flag:
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"Removed temp dir {work_dir}")


if __name__ == "__main__":
    main()
