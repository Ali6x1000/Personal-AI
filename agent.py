"""
AliJR LiveKit voice agent: Google Cloud STT/TTS + Gemini LLM + LlamaIndex RAG on Chroma.

- RAG vectors: local Ollama (default ``OLLAMA_EMBED_MODEL=nomic-embed-text``, ``OLLAMA_BASE_URL``).
- RAG answers: Gemini via GOOGLE_API_KEY (must match indexer embedding setup above).
- google.STT / google.TTS use Google Cloud Speech & Text-to-Speech; configure ADC as needed.
"""

from __future__ import annotations

import asyncio
import logging
import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, function_tool
from livekit.agents.llm.tool_context import Toolset
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.chroma import ChromaVectorStore
from livekit.plugins import google as lk_google

load_dotenv()

logger = logging.getLogger("alijr.agent")

PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "alijr_kb"

ALIJR_SYSTEM_PROMPT = (
    "You are AliJR, an elite AI engineering assistant and academic clone. "
    "You have access to university coursework, GitHub READMEs, and ML research. "
    "Be concise, technical, and conversational."
)


class FolderCategory(str, Enum):
    submissions = "submissions"
    research = "research"
    course_lectures = "course_lectures"
    projects = "projects"
    all = "all"


_llama_index_configured = False


def _configure_llama_index() -> None:
    global _llama_index_configured
    if _llama_index_configured:
        return

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY must be set for RAG synthesis LLM (Gemini).")

    ollama_host = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model_name = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # Same embedding backend/settings as rag_indexer.py — otherwise Chroma search is wrong.
    Settings.embed_model = OllamaEmbedding(
        model_name=embed_model_name,
        base_url=ollama_host,
    )
    Settings.llm = Gemini(
        api_key=api_key,
        model="models/gemini-1.5-flash",
        temperature=0.2,
    )

    _llama_index_configured = True


class KnowledgeBaseTools(Toolset):
    """Folder-aware Chroma RAG exposed as a LiveKit toolset."""

    def __init__(self, *, chroma_dir: Path | None = None) -> None:
        self._chroma_dir = chroma_dir or CHROMA_DB_DIR
        self._index = None
        super().__init__(id="alijr_knowledge_base")

    def _load_index(self) -> VectorStoreIndex:
        _configure_llama_index()
        if self._index is None:
            if not self._chroma_dir.exists():
                raise FileNotFoundError(
                    f"Chroma persist dir missing: {self._chroma_dir}. Run rag_indexer.py first."
                )
            store = ChromaVectorStore.from_params(
                collection_name=COLLECTION_NAME,
                persist_dir=str(self._chroma_dir.resolve()),
            )
            self._index = VectorStoreIndex.from_vector_store(store)
            logger.info("Loaded Chroma index (%s, %s)", self._chroma_dir, COLLECTION_NAME)
        return self._index

    @function_tool(
        description=(
            "Search the AliJR knowledge base. Restrict to one folder or use ALL for every "
            "category (submissions, research, course_lectures, projects)."
        )
    )
    async def search_documents(self, query: str, folder_category: FolderCategory) -> str:
        index = self._load_index()

        filters: MetadataFilters | None = None
        if folder_category != FolderCategory.all:
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="folder_category",
                        value=folder_category.value,
                    )
                ]
            )

        kwargs: dict = {"similarity_top_k": 8}
        if filters is not None:
            kwargs["filters"] = filters

        query_engine = index.as_query_engine(**kwargs)

        def _run_query() -> str:
            resp = query_engine.query(query)
            return str(resp).strip()

        return await asyncio.to_thread(_run_query)


async def entrypoint(ctx: JobContext) -> None:
    kb = KnowledgeBaseTools()

    session = AgentSession(
        stt=lk_google.STT(),
        llm=lk_google.LLM(model="gemini-1.5-flash"),
        tts=lk_google.TTS(),
        tools=[kb],
        max_tool_steps=8,
    )

    agent = Agent(instructions=ALIJR_SYSTEM_PROMPT)

    await session.start(agent=agent, room=ctx.room)


def main() -> None:
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=os.getenv("LIVEKIT_AGENT_NAME", ""),
        )
    )


if __name__ == "__main__":
    main()
