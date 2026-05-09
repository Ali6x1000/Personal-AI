"""
AliJR LiveKit voice agent: Google Cloud STT/TTS + Gemini LLM + LlamaIndex RAG on Chroma.

- RAG vectors: local Ollama (default ``OLLAMA_EMBED_MODEL=nomic-embed-text``, ``OLLAMA_BASE_URL``).
- RAG retrieval embeddings can use Ollama or Gemini via ``EMBED_PROVIDER``.
- Voice LLM uses the unified ``google-genai`` client (``google.genai``) via ``livekit-plugins-google``,
  configured for **Vertex AI** with service-account credentials.
- **Vertex limitation:** Gemini on Vertex rejects mixing native ``Google Search`` tools with Python
  ``function_tool`` declarations in one request (``MULTIPLE_TOOLS … all search tools``). This agent
  uses **RAG function tools only**; add live web via a separate HTTP search tool if you need it.
- Extended thinking uses ``ThinkingConfig(thinking_budget=-1, include_thoughts=True)``; when the model
  emits thought content, the LiveKit agent pipeline surfaces assistant state for UIs that listen for it.
  Full “thinking” visualization depends on LiveKit Components + model/plugin support for thought parts.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import warnings
from enum import Enum
from pathlib import Path

from google.genai import types
from google.oauth2 import service_account

# Suppress noisy upstream deprecation/future warnings during local runs.
warnings.filterwarnings(
    "ignore",
    message=r"You are using a Python version \(3\.10\..*\) which Google will stop supporting.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*google\.generativeai.*ended.*",
    category=FutureWarning,
)

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobExecutorType,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.agents.llm.tool_context import Toolset
from livekit.agents.types import NOT_GIVEN, APIConnectOptions
from livekit.agents.voice.agent_session import SessionConnectOptions
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import QueryBundle
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from livekit.plugins import google as lk_google

load_dotenv()

logger = logging.getLogger("alijr.agent")

PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "alijr_kb"

# Vertex + google-genai token refresh expects OAuth scopes on the credential; missing scopes →
# ``invalid_scope: Invalid OAuth scope or ID token audience provided`` on JWT exchange.
GCP_VERTEX_SCOPES: tuple[str, ...] = ("https://www.googleapis.com/auth/cloud-platform",)


def _vertex_service_account_credentials(credentials_file: str) -> service_account.Credentials:
    path = Path(credentials_file).expanduser()
    creds = service_account.Credentials.from_service_account_file(
        str(path),
        scopes=list(GCP_VERTEX_SCOPES),
    )
    return creds


ALIJR_SYSTEM_PROMPT = (
    "Your name is Ali Junior (also shortened as AliJR). You are an elite AI engineering developed by Ali Nawaf"
    "assistant and personal academic twin. You speak in first person about the owner's work "
    'when relevant (e.g. "your notes", "your project"). Your voice interface should stay '
    "concise and conversational, but technically precise.\n\n"
    "Knowledge layout: private materials live in categorized folders accessible only through "
    "the search_documents tool—submissions (assignments/writeups), course_lectures, research "
    "(papers/PDFs/notes), and projects (repos/READMEs).\n\n"
    "Mandatory RAG discipline—whenever the user asks anything that might be grounded in those "
    "materials (coursework, grades, deadlines, implementations, citations, equations, diagrams, "
    "paper summaries, README details, lecture topics, labs, proofs, datasets, repos, submissions), "
    "you MUST call search_documents before answering. Prefer folder_category=submissions "
    "| course_lectures | research | projects when the user's intent maps clearly; "
    "use folder_category=all if unsure or topics span folders. Reformulate vague questions into "
    "a crisp search query.\n\n"
    "When search_documents returns excerpts, they are plain text chunks from PDFs/Markdown and may "
    "include OCR artefacts: focus on readable sentences, headings, and numbered steps; ignore "
    "isolated symbols and table junk. Each block is labeled with source path and a relevance score.\n\n"
    "Answer using the snippets returned when they contain relevant facts; summarize in your "
    "own words. If snippets are thin, off-topic, or only noise, say that clearly—do not pretend "
    "the documents covered a topic they did not—and ask one clarifying question or suggest a "
    "different search phrase.\n\n"
    "Avoid long lists, URLs, markdown headings, emoji, bullet walls, and punctuation that "
    "doesn't read aloud well."
)


def _session_connect_options() -> SessionConnectOptions:
    """Use a longer default TTS timeout than 10s (long answers + Cloud TTS streaming)."""

    tts_timeout = float(os.getenv("ALIJR_TTS_TIMEOUT_SEC", os.getenv("GOOGLE_TTS_TIMEOUT_SEC", "60")))
    # LLM retries can multiply wall time—keep roomy but bounded.
    llm_timeout = float(os.getenv("ALIJR_LLM_TIMEOUT_SEC", "120"))
    return SessionConnectOptions(
        tts_conn_options=APIConnectOptions(timeout=max(30.0, tts_timeout), max_retry=3),
        llm_conn_options=APIConnectOptions(timeout=max(30.0, llm_timeout), max_retry=3),
    )


class FolderCategory(str, Enum):
    submissions = "submissions"
    research = "research"
    course_lectures = "course_lectures"
    projects = "projects"
    all = "all"


def _snippet_alphabetic_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(1 for c in text if c.isalpha())
    return letters / len(text)


def _clean_rag_excerpt(text: str, *, max_chars: int = 1400) -> str:
    """Normalize PDF/OCR-ish text into something the voice LLM can read."""

    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = t.strip()
    if len(t) <= max_chars:
        return t
    t = t[:max_chars]
    cut = max(t.rfind(". "), t.rfind("\n"))
    if cut > max_chars // 3:
        t = t[: cut + 1].strip()
    else:
        sp = t.rfind(" ")
        if sp > max_chars // 2:
            t = t[:sp].rstrip(".,;: ") + " …"
        else:
            t = t.rstrip() + " …"
    return t


def _filter_rag_nodes(
    nodes: list,
    *,
    query_str: str,
    similarity_cutoff: float,
    max_snippets: int,
) -> tuple[list, list[str]]:
    """Apply similarity cutoff and drop unreadable/low-signal chunks."""

    notes: list[str] = []
    if not nodes:
        return [], notes

    numeric_scores = [n.score for n in nodes if n.score is not None]
    if numeric_scores:
        logger.info(
            "search_documents similarity preview (LlamaIndex: exp(-chroma_distance)): "
            "min=%.4f median=%.4f max=%.4f",
            min(numeric_scores),
            sorted(numeric_scores)[len(numeric_scores) // 2],
            max(numeric_scores),
        )

    post = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
    filtered = post.postprocess_nodes(nodes, query_bundle=QueryBundle(query_str=query_str))
    if not filtered:
        notes.append(
            "No chunks met the similarity threshold; falling back to the strongest raw matches."
        )
        filtered = sorted(nodes, key=lambda n: (n.score is not None, n.score or 0.0), reverse=True)[
            : max(max_snippets + 4, 8)
        ]

    ordered = sorted(
        filtered,
        key=lambda n: (n.score is not None, n.score or 0.0),
        reverse=True,
    )
    kept: list = []
    for n in ordered:
        raw = (n.node.get_content(metadata_mode="none") or "").strip()
        if len(raw) > 100 and _snippet_alphabetic_ratio(raw) < 0.18:
            continue
        kept.append(n)
        if len(kept) >= max_snippets:
            break

    if not kept and ordered:
        notes.append("Chunks were mostly noise by heuristics; returning top raw hits anyway.")
        kept = ordered[: min(3, len(ordered))]

    return kept, notes


def _gemini_generate_content_voice_defaults() -> types.GenerateContentConfig:
    thinking = types.ThinkingConfig(thinking_budget=-1, include_thoughts=True)
    safety = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.OFF,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.OFF,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.OFF,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.OFF,
        ),
    ]
    return types.GenerateContentConfig(
        temperature=1.0,
        top_p=1.0,
        thinking_config=thinking,
        safety_settings=safety,
    )


def _build_vertex_voice_llm(
    *,
    project: str,
    location: str,
    credentials: service_account.Credentials,
) -> lk_google.LLM:
    """LiveKit Google LLM: Vertex + ``google-genai`` request options (mirrors GenerateContentConfig)."""

    api_key_from_env = os.environ.get("GOOGLE_CLOUD_API_KEY")
    # When ``vertexai=True``, the plugin uses service-account / ADC and clears API keys internally;
    # we still read ``GOOGLE_CLOUD_API_KEY`` when set, per deployment conventions.
    api_key = api_key_from_env if api_key_from_env else NOT_GIVEN

    voice_cfg = _gemini_generate_content_voice_defaults()
    return lk_google.LLM(
        model="gemini-2.5-flash",
        api_key=api_key,
        vertexai=True,
        project=project,
        location=location,
        credentials=credentials,
        temperature=voice_cfg.temperature,
        top_p=voice_cfg.top_p,
        thinking_config=voice_cfg.thinking_config,
        safety_settings=voice_cfg.safety_settings,
    )


_llama_index_configured = False


def _configure_llama_index() -> None:
    global _llama_index_configured
    if _llama_index_configured:
        return

    credentials_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_file:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS is required for Vertex AI configuration."
        )
    if not Path(credentials_file).expanduser().exists():
        raise RuntimeError(
            f"GOOGLE_APPLICATION_CREDENTIALS points to a missing file: {credentials_file}"
        )

    credentials = _vertex_service_account_credentials(credentials_file)
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or getattr(credentials, "project_id", None)
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not project:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT is required for Vertex AI "
            "(or include project_id in the service-account JSON)."
        )

    qp = os.getenv("GOOGLE_CLOUD_QUOTA_PROJECT")
    if qp:
        credentials = credentials.with_quota_project(qp)

    embed_provider = os.getenv("EMBED_PROVIDER", "ollama").strip().lower()
    if embed_provider == "ollama":
        ollama_host = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        embed_model_name = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        Settings.embed_model = OllamaEmbedding(
            model_name=embed_model_name,
            base_url=ollama_host,
        )
    elif embed_provider == "gemini":
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

        model = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
        try:
            Settings.embed_model = GoogleGenAIEmbedding(
                model_name=model,
                vertexai=True,
                project=project,
                location=location,
            )
        except TypeError:
            Settings.embed_model = GoogleGenAIEmbedding(
                model_name=model,
                vertexai_config={
                    "project": project,
                    "location": location,
                    "credentials": credentials,
                },
            )
    else:
        raise RuntimeError(
            f"Unsupported EMBED_PROVIDER={embed_provider!r}. Expected 'ollama' or 'gemini'."
        )

    from llama_index.llms.google_genai import GoogleGenAI

    voice_cfg = _gemini_generate_content_voice_defaults()
    Settings.llm = GoogleGenAI(
        model="gemini-2.5-flash",
        generation_config=voice_cfg,
        vertexai_config={
            "project": project,
            "location": location,
            "credentials": credentials,
        },
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
            "PRIMARY knowledge search for Ali Junior. Vector search over the owner's indexed "
            "PDFs/markdown/text under submissions, research, course_lectures, and projects. "
            "Always call this for questions that could rely on those materials. "
            "folder_category=submissions | research | course_lectures | projects narrows retrieval; "
            "use all only when the scope is unclear."
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

        top_k = int(os.getenv("ALIJR_RAG_TOP_K", "16"))
        kwargs: dict = {"similarity_top_k": max(8, top_k)}
        if filters is not None:
            kwargs["filters"] = filters

        retriever = index.as_retriever(**kwargs)
        min_sim = float(os.getenv("ALIJR_RAG_MIN_SIMILARITY", "0.06"))
        max_snippets = int(os.getenv("ALIJR_RAG_MAX_SNIPPETS", "6"))

        def _run_query() -> str:
            nodes = retriever.retrieve(query)
            logger.info(
                "search_documents retrieve query=%r folder_category=%s raw_nodes=%s",
                query,
                folder_category.value,
                len(nodes),
            )
            if not nodes:
                return (
                    "No vectors matched this query in the index. "
                    "The topic may be missing from alijr_knowledge_base/ or needs re-indexing (run rag_indexer.py)."
                )

            picked, filter_notes = _filter_rag_nodes(
                nodes,
                query_str=query,
                similarity_cutoff=min_sim,
                max_snippets=max_snippets,
            )
            logger.info(
                "search_documents after filter: kept=%s (min_similarity=%s)",
                len(picked),
                min_sim,
            )

            snippets: list[str] = []
            for idx, node in enumerate(picked, start=1):
                raw = (node.node.get_content(metadata_mode="none") or "").strip()
                text = _clean_rag_excerpt(raw)
                if not text:
                    continue
                meta = getattr(node.node, "metadata", None) or {}
                path_hint = meta.get("file_path") or meta.get("filename") or meta.get("document_id")
                category = meta.get("folder_category", "")
                head = f"[{idx}]"
                if node.score is not None:
                    head += f" relevance={float(node.score):.3f}"
                if path_hint:
                    head += f" source={path_hint}"
                if category:
                    head += f" folder={category}"
                snippets.append(f"{head}\n{text}")
            if not snippets:
                return (
                    "Retrieval ran but every chunk was empty or unreadable after cleanup. "
                    "Try re-indexing or add cleaner source files."
                )

            preamble = (
                "Below are excerpted passages from the owner's indexed knowledge base "
                f"({folder_category.value}). Treat relevance scores as soft signals; prefer coherent "
                "sentences over layout debris.\n"
            )
            if filter_notes:
                preamble += "Notes: " + " ".join(filter_notes) + "\n"
            blob = preamble + "\n---\n\n".join(snippets)
            logger.debug("search_documents returning %s chars of context", len(blob))
            return blob

        return await asyncio.to_thread(_run_query)


async def entrypoint(ctx: JobContext) -> None:
    kb = KnowledgeBaseTools()
    credentials_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_file:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS is required for google.STT/google.TTS. "
            "Set it to a Google Cloud service-account JSON path."
        )
    if not Path(credentials_file).expanduser().exists():
        raise RuntimeError(
            f"GOOGLE_APPLICATION_CREDENTIALS points to a missing file: {credentials_file}"
        )
    credentials = _vertex_service_account_credentials(credentials_file)
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or getattr(credentials, "project_id", None)
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not project:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT is required for Vertex AI "
            "(or include project_id in the service-account JSON)."
        )
    qp = os.getenv("GOOGLE_CLOUD_QUOTA_PROJECT")
    if qp:
        credentials = credentials.with_quota_project(qp)
    tts_model = os.getenv("GOOGLE_TTS_MODEL", "chirp_3")

    voice_llm = _build_vertex_voice_llm(project=project, location=location, credentials=credentials)

    session = AgentSession(
        stt=lk_google.STT(credentials_file=credentials_file),
        llm=voice_llm,
        tts=lk_google.TTS(
            credentials_file=credentials_file,
            model_name=tts_model,
        ),
        conn_options=_session_connect_options(),
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
            # On local Macs, process-based workers can time out during cold startup/imports.
            job_executor_type=JobExecutorType.THREAD,
            initialize_process_timeout=60.0,
            # Keep local behavior predictable: one warm runner instead of CPU-count fan-out.
            num_idle_processes=1,
        )
    )


if __name__ == "__main__":
    main()
