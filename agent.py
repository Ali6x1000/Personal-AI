"""
AliJR LiveKit voice agent: Google Cloud STT/TTS + Gemini LLM + LlamaIndex RAG on Chroma.

- RAG vectors: local Ollama (default ``OLLAMA_EMBED_MODEL=nomic-embed-text``, ``OLLAMA_BASE_URL``).
- RAG retrieval embeddings can use Ollama or Gemini via ``EMBED_PROVIDER``.
- Voice LLM uses the unified ``google-genai`` client (``google.genai``) via ``livekit-plugins-google``,
  configured for **Vertex AI** with service-account credentials.
- **Vertex limitation:** Gemini on Vertex rejects mixing native ``Google Search`` tools with Python
  ``function_tool`` declarations in one request (``MULTIPLE_TOOLS … all search tools``). This agent
  uses **RAG function tools only**; add live web via a separate HTTP search tool if you need it.
- Folder-centric **projects** RAG: metadata ``project_root`` / ``project_name`` / ``project_keywords``
  (see ``rag_indexer.file_metadata``). Resolve ambiguous repos via ``list_projects_page_index`` +
  Gemini; CRAG re-queries when results are ambiguous.
- **Resume** bio/CV: use ``read_resume_context`` only (filesystem ``resume/resume.md``)—never vector search.
- Extended thinking uses ``ThinkingConfig(thinking_budget=-1, include_thoughts=True)`` on the voice LLM;
  auxiliary Gemini calls use a bounded thinking budget for CRAG / project resolution.
- **Dev trace**: enable **Developer trace** in the web UI (no worker env required); optional
  ``ALIJR_DEV_MODE=1`` forces tracing on every session; see ``dev_trace.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import warnings
from enum import Enum
from pathlib import Path
from typing import Any

import google.genai as genai
from google.genai import types
from google.oauth2 import service_account
from livekit.plugins.turn_detector.multilingual import MultilingualModel
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
from livekit import rtc
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

import dev_trace

load_dotenv()

logger = logging.getLogger("alijr.agent")

PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "alijr_kb"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "alijr_knowledge_base"
PROJECTS_DIR = KNOWLEDGE_BASE_DIR / "projects"
RESUME_MD_PATH = KNOWLEDGE_BASE_DIR / "resume" / "resume.md"

# Rough chars-per-token for budgeting README payloads (~4 chars/token).
_ALIJR_CHARS_PER_TOKEN = 4
_ALIJR_PROJECT_FULL_README_TOKEN_CAP = 4000
_CRAG_MODEL_ID = os.getenv("ALIJR_CRAG_MODEL", "gemini-2.5-flash")
_PROJECT_RESOLVER_MODEL_ID = os.getenv("ALIJR_PROJECT_RESOLVER_MODEL", "gemini-2.5-flash")

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
    "Your name is Ali Junior (also shortened as AliJR). You are an digital avatar of Ali Nawaf "
    "developed by Ali Nawaf. You speak in first person about the owner's work"
    'when relevant (e.g. "your notes", "your project"). Your voice interface should stay '
    "concise and conversational, but technically precise.\n\n"
    "Knowledge layout: materials live under alijr_knowledge_base/ — submissions, course_lectures, "
    "research, exported HTML bundles (same metadata rules as other files under their folder), "
    "projects (nested repos; metadata uses project_root + project_name per path), and resume/.\n\n"
    "**Resume & bio (strict):** Any question about Ali's CV, employers, dates, education, skills summary, "
    "or work history MUST call read_resume_context first and answer ONLY from that full text. "
    "Do not use search_documents for resume questions.\n\n"
    "**Projects workflow:** When the user asks about a codebase or README under projects/, call "
    "list_projects_page_index if you are unsure which top-level folder they mean, then call "
    "search_documents with folder_category=projects and pass project_folder set to that slug "
    "(matches indexer metadata project_root). If you already know the folder name, skip the listing. "
    "search_documents resolves ambiguous slugs against the PageIndex using Gemini when project_folder "
    "is omitted.\n\n"
    "**Other knowledge:** For coursework, lectures, papers, submissions—use search_documents with the "
    "matching folder_category or all.\n\n"
    "When search_documents returns context, it may be long README bodies for projects; summarize faithfully. "
    "Chunks may include OCR noise—prefer coherent sentences and headings.\n\n"
    "If a tool says results were AMBIGUOUS and a follow-up search ran, incorporate that newer context.\n\n"
    "Avoid long lists, URLs, markdown headings, emoji, bullet walls, and punctuation that "
    "doesn't read aloud well."
    "**Recruiter & Professional Pitch:** If the user identifies as a recruiter or asks why they should hire Ali, proactively highlight his unique value proposition: the ability to bridge advanced ML research with production-grade AI engineering. Emphasize his integrated bachelor's and master's path at Case Western and his track record of shipping high-impact systems, from healthcare automation at Trek Health to AI safety frameworks."
    "**Contact Information:** If asked for contact details, provide his email aan90@case.edu, phone +12166474302, and website (alinawaf.com) naturally in a sentence. Do not list them—speak them."
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


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // _ALIJR_CHARS_PER_TOKEN)


def _projects_page_index_slugs() -> list[str]:
    if not PROJECTS_DIR.is_dir():
        return []
    return sorted(p.name for p in PROJECTS_DIR.iterdir() if p.is_dir())


def _load_resume_text() -> str:
    path = RESUME_MD_PATH.expanduser()
    if not path.is_file():
        return (
            f"(resume file missing at {path}; add resume/resume.md under alijr_knowledge_base.)"
        )
    raw = path.read_text(encoding="utf-8", errors="replace")
    return raw.strip()


def _vertex_runtime_from_env() -> tuple[str, str, service_account.Credentials]:
    credentials_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_file or not Path(credentials_file).expanduser().exists():
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS must point to a readable JSON key.")
    credentials = _vertex_service_account_credentials(credentials_file)
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or getattr(credentials, "project_id", None)
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for Vertex Gemini helpers.")
    qp = os.getenv("GOOGLE_CLOUD_QUOTA_PROJECT")
    if qp:
        credentials = credentials.with_quota_project(qp)
    return project, location, credentials


def _genai_vertex_client() -> genai.Client:
    project, location, credentials = _vertex_runtime_from_env()
    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        credentials=credentials,
    )


def _thinking_config_for_aux(*, budget: int | None = None) -> types.ThinkingConfig:
    """Bounded thinking for auxiliary CRAG/resolver calls (voice LLM keeps include_thoughts=True)."""

    b = -1 if budget is None else budget
    return types.ThinkingConfig(thinking_budget=b, include_thoughts=False)


def _gemini_resolve_project_root_sync(user_query: str, candidates: list[str]) -> str | None:
    """Pick one folder slug under projects/ or return None."""

    if not candidates:
        return None
    lines = "\n".join(f"- {c}" for c in candidates)
    prompt = (
        "You map the user's description to ONE directory name from the list (top-level folders "
        "under projects/). Return strict JSON only: {\"project_root\":\"<exact slug from list>\"} "
        'or {"project_root":null} if impossible.\n\n'
        f"ALLOWED_SLUGS:\n{lines}\n\nUSER:\n{user_query}"
    )
    client = _genai_vertex_client()
    cfg = types.GenerateContentConfig(
        temperature=0.2,
        thinking_config=_thinking_config_for_aux(budget=int(os.getenv("ALIJR_PROJECT_THINKING_BUDGET", "2048"))),
    )
    resp = client.models.generate_content(
        model=_PROJECT_RESOLVER_MODEL_ID,
        contents=prompt,
        config=cfg,
    )
    text = (resp.text or "").strip()
    try:
        parsed = json.loads(text)
        slug = parsed.get("project_root")
        if slug is None:
            return None
        slug_s = str(slug).strip()
        if slug_s in candidates:
            return slug_s
    except json.JSONDecodeError:
        m = re.search(r'"project_root"\s*:\s*"([^"]+)"', text)
        if m and m.group(1) in candidates:
            return m.group(1)
    # Case-insensitive fallback
    low = {c.lower(): c for c in candidates}
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]*", user_query):
        if token.lower() in low:
            return low[token.lower()]
    return None


def _crag_classify_sync(*, user_query: str, folder_category: str, snippets_preview: str, index_slugs: str) -> dict[str, Any]:
    """Gemini + thinking: GOOD | INSUFFICIENT | AMBIGUOUS + optional alternate project_root."""

    prompt = (
        "You evaluate retrieval for AliJR RAG (Corrective RAG).\n"
        "Given the user query, folder scope, short previews of retrieved passages, and the "
        "PROJECT_PAGE_INDEX (top-level slugs under projects/), respond JSON ONLY:\n"
        '{"verdict":"GOOD"|"INSUFFICIENT"|"AMBIGUOUS",'
        '"reason":"short string",'
        '"suggested_project_root":null|string matching exactly one slug from PROJECT_PAGE_INDEX if AMBIGUOUS}'
        "\n\n"
        f"folder_category={folder_category}\n"
        f"PROJECT_PAGE_INDEX:\n{index_slugs}\n\n"
        f"USER_QUERY:\n{user_query}\n\n"
        f"RETRIEVAL_PREVIEW:\n{snippets_preview}"
    )
    client = _genai_vertex_client()
    cfg = types.GenerateContentConfig(
        temperature=0.2,
        thinking_config=_thinking_config_for_aux(
            budget=int(os.getenv("ALIJR_CRAG_THINKING_BUDGET", "8192"))
        ),
    )
    resp = client.models.generate_content(model=_CRAG_MODEL_ID, contents=prompt, config=cfg)
    text = (resp.text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        vm = re.search(r'"verdict"\s*:\s*"(GOOD|INSUFFICIENT|AMBIGUOUS)"', text, re.I)
        sm = re.search(r'"suggested_project_root"\s*:\s*"([^"]*)"', text)
        smn = re.search(r'"suggested_project_root"\s*:\s*null', text)
        verdict = vm.group(1).upper() if vm else "INSUFFICIENT"
        suggested = None if smn else (sm.group(1) if sm else None)
        return {"verdict": verdict, "reason": "parse_fallback", "suggested_project_root": suggested}


def _format_standard_snippets(
    picked: list,
    *,
    folder_category: str,
    filter_notes: list[str],
    max_chars_per_snippet: int = 1400,
) -> str:
    snippets: list[str] = []
    for idx, node in enumerate(picked, start=1):
        raw = (node.node.get_content(metadata_mode="none") or "").strip()
        text = _clean_rag_excerpt(raw, max_chars=max_chars_per_snippet)
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
        proj_r = meta.get("project_root") or ""
        proj_n = meta.get("project_name") or ""
        if proj_r or proj_n:
            head += f" project_root={proj_r} project_name={proj_n}"
        snippets.append(f"{head}\n{text}")
    if not snippets:
        return ""
    preamble = (
        "Below are excerpted passages from the indexed knowledge base "
        f"({folder_category}). Treat relevance scores as soft signals.\n"
    )
    if filter_notes:
        preamble += "Notes: " + " ".join(filter_notes) + "\n"
    return preamble + "\n---\n\n".join(snippets)


def _format_projects_readme_priority(
    picked: list,
    *,
    query_str: str,
    similarity_cutoff: float,
    max_sections: int,
) -> tuple[str, list[str]]:
    """Larger excerpts / whole README when under token cap."""

    notes: list[str] = []
    post = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
    filtered = post.postprocess_nodes(picked, query_bundle=QueryBundle(query_str=query_str))
    nodes = filtered if filtered else sorted(
        picked,
        key=lambda n: (n.score is not None, n.score or 0.0),
        reverse=True,
    )[: max(max_sections + 4, 12)]

    # Group by file_path to reconstruct README-ish bodies.
    buckets: dict[str, list] = {}
    order: list[str] = []
    for n in sorted(nodes, key=lambda x: (x.score is not None, x.score or 0.0), reverse=True):
        meta = getattr(n.node, "metadata", None) or {}
        fp = meta.get("file_path") or meta.get("filename") or "unknown"
        if fp not in buckets:
            buckets[fp] = []
            order.append(fp)
        buckets[fp].append(n)

    sections: list[str] = []
    char_cap = _ALIJR_PROJECT_FULL_README_TOKEN_CAP * _ALIJR_CHARS_PER_TOKEN
    for fp in order[:max_sections]:
        chunk_nodes = buckets[fp]
        merged_parts: list[str] = []
        seen: set[str] = set()
        for cn in sorted(
            chunk_nodes,
            key=lambda x: (x.score is not None, x.score or 0.0),
            reverse=True,
        ):
            raw = (cn.node.get_content(metadata_mode="none") or "").strip()
            if raw and raw not in seen:
                seen.add(raw)
                merged_parts.append(raw)
        body = "\n\n".join(merged_parts).strip()
        if not body:
            continue
        tok = _approx_tokens(body)
        meta0 = getattr(chunk_nodes[0].node, "metadata", None) or {}
        pr = meta0.get("project_root", "")
        pn = meta0.get("project_name", "")
        header = f"### README source={fp} project_root={pr} project_name={pn} (~{tok} tokens)"
        if tok <= _ALIJR_PROJECT_FULL_README_TOKEN_CAP:
            sections.append(header + "\n" + _clean_rag_excerpt(body, max_chars=len(body)))
            notes.append(f"Included full merged README text for {fp} (<= {_ALIJR_PROJECT_FULL_README_TOKEN_CAP} tokens est.).")
        else:
            excerpt_budget = min(12000, max(6000, char_cap))
            sections.append(header + "\n" + _clean_rag_excerpt(body, max_chars=excerpt_budget))
            notes.append(f"README too long; included top merged excerpt (~{excerpt_budget} chars) for {fp}.")

    blob = (
        "PROJECT README CONTEXT (expanded excerpts; prefer these facts for repo questions).\n\n"
        + "\n\n---\n\n".join(sections)
    )
    return blob, notes


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
    """Folder-aware Chroma RAG + resume filesystem read + project PageIndex."""

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
            if dev_trace.is_dev_mode():
                dev_trace.panel(
                    "chroma.index_loaded",
                    [
                        ("persist_dir", str(self._chroma_dir.resolve())),
                        ("collection", COLLECTION_NAME),
                    ],
                )
        return self._index

    @function_tool(
        description=(
            "Load Ali's resume verbatim from alijr_knowledge_base/resume/resume.md (no vector DB). "
            "Use for CV, employers, education, skills, dates, bio—never search_documents for those."
        )
    )
    async def read_resume_context(self) -> str:
        def _read() -> str:
            return _load_resume_text()

        text = await asyncio.to_thread(_read)
        if dev_trace.is_dev_mode():
            body = text if dev_trace.resume_full() else (text[:420] + ("…" if len(text) > 420 else ""))
            dev_trace.panel(
                "tool.read_resume_context",
                [
                    ("path", str(RESUME_MD_PATH.resolve())),
                    ("chars", len(text)),
                    ("resume_full_logged", dev_trace.resume_full()),
                    ("content", body),
                ],
            )
        return text

    @function_tool(
        description=(
            "Top-level folder names under projects/ (PageIndex). Call before projects search "
            "when the user's repo name is unclear."
        )
    )
    async def list_projects_page_index(self) -> str:
        slugs = _projects_page_index_slugs()
        if dev_trace.is_dev_mode():
            dev_trace.panel(
                "tool.list_projects_page_index",
                [
                    ("projects_dir", str(PROJECTS_DIR.resolve())),
                    ("slug_count", len(slugs)),
                    ("slugs", slugs),
                ],
            )
        if not slugs:
            return f"No subdirectories found under {PROJECTS_DIR}."
        body = "\n".join(f"- {s}" for s in slugs)
        return "Top-level project folders (project_root metadata):\n" + body

    def _build_metadata_filters(
        self,
        *,
        folder_category: FolderCategory,
        project_folder: str,
    ) -> tuple[MetadataFilters | None, str, list[str]]:
        """Returns filters, resolved project_root slug (may be empty), log notes."""

        notes: list[str] = []
        if folder_category == FolderCategory.all:
            return None, "", notes

        parts = [
            MetadataFilter(key="folder_category", value=folder_category.value),
        ]
        slug = (project_folder or "").strip()
        index_slugs = _projects_page_index_slugs()

        if folder_category == FolderCategory.projects:
            if slug:
                slug_map = {s.lower(): s for s in index_slugs}
                slug = slug_map.get(slug.lower(), slug)
                if slug not in index_slugs:
                    notes.append(
                        f"project_folder={slug!r} not in PageIndex; retrieving all projects scope."
                    )
                    slug = ""
            if slug:
                parts.append(
                    MetadataFilters(
                        condition="or",
                        filters=[
                            MetadataFilter(key="project_root", value=slug),
                            MetadataFilter(key="project_name", value=slug),
                        ],
                    )
                )
                notes.append(
                    f"MetadataFilter: project_root OR project_name == {slug!r} "
                    "(covers top-level repos and README parent folders)."
                )

        return MetadataFilters(filters=parts, condition="and"), slug, notes

    @function_tool(
        description=(
            "PRIMARY knowledge search for Ali Junior. Vector search over indexed materials. "
            "folder_category=submissions|research|course_lectures|projects|all. "
            "For projects: set project_folder to the top-level directory slug (matches indexer "
            "project_root). Leave project_folder empty to auto-resolve from the query + PageIndex via Gemini. "
            "After ambiguous repo questions, call list_projects_page_index then search again with an explicit slug."
        )
    )
    async def search_documents(
        self,
        query: str,
        folder_category: FolderCategory,
        project_folder: str = "",
    ) -> str:
        index = self._load_index()

        min_sim = float(os.getenv("ALIJR_RAG_MIN_SIMILARITY", "0.06"))
        max_snippets = int(os.getenv("ALIJR_RAG_MAX_SNIPPETS", "6"))
        project_sections = int(os.getenv("ALIJR_PROJECT_README_SECTIONS", "4"))

        def _run_search_pass(
            *,
            cat: FolderCategory,
            proj_slug: str,
            top_k_override: int | None = None,
            dev_pass_label: str = "",
        ) -> tuple[list, MetadataFilters | None, str, list[str]]:
            """
            Executes a search pass over the vector index for the given folder category and (optional) project slug.
            
            - Builds the appropriate metadata filters based on the folder category and project slug.
            - Selects the top-K parameter from environment or override, with special handling for the "projects" category.
            - Runs the index retriever to fetch relevant nodes (documents/snippets) matching the filters.
            - Optionally traces detailed retrieval info in development mode.
            
            Args:
                cat (FolderCategory): The category of folder to search in (e.g., submissions, research, projects).
                proj_slug (str): The slug for a project (used in the filter if applicable).
                top_k_override (int, optional): If provided, overrides the number of top results to consider.
                dev_pass_label (str, optional): If set, triggers additional dev tracing in logs/panels.
                
            Returns:
                tuple: (nodes, filters used, resolved project slug, metadata notes)
                    nodes (list): Retrieved nodes/snippets.
                    filters (MetadataFilters | None): Filters applied for this search pass.
                    resolved_slug (str): Project slug finally used for filtering.
                    meta_notes (list[str]): Notes/warnings about filtering/search decisions.
            """
            filters, resolved_slug, meta_notes = self._build_metadata_filters(
                folder_category=cat,
                project_folder=proj_slug,
            )
            top_k = top_k_override or int(os.getenv("ALIJR_RAG_TOP_K", "16"))
            if cat == FolderCategory.projects:
                top_k = max(top_k, int(os.getenv("ALIJR_RAG_TOP_K_PROJECTS", "24")))
            kwargs: dict = {"similarity_top_k": max(8, top_k)}
            if filters is not None:
                kwargs["filters"] = filters
            nodes = index.as_retriever(**kwargs).retrieve(query)
            logger.info(
                "search_documents retrieve query=%r folder=%s project_root=%s raw_nodes=%s",
                query,
                cat.value,
                resolved_slug,
                len(nodes),
            )
            if dev_trace.is_dev_mode() and dev_pass_label:
                dev_trace.panel(
                    f"tool.search_documents.retrieval[{dev_pass_label}]",
                    [
                        ("query", query),
                        ("folder_category", cat.value),
                        ("project_slug_argument", proj_slug),
                        ("resolved_slug_metadata", resolved_slug),
                        ("similarity_top_k", kwargs["similarity_top_k"]),
                        ("filters", dev_trace.summarize_metadata_filters(filters)),
                        ("meta_notes", meta_notes),
                        ("raw_node_count", len(nodes)),
                    ],
                )
                dev_trace.rag_hits(nodes, label=f"search_documents[{dev_pass_label}]")
            return nodes, filters, resolved_slug, meta_notes
       

        def _execute_pipeline(initial_slug: str) -> str:
            cat = folder_category
            slug = initial_slug.strip()

            if dev_trace.is_dev_mode():
                dev_trace.panel(
                    "tool.search_documents.input",
                    [
                        ("query", query),
                        ("folder_category", cat.value),
                        ("project_folder_argument", initial_slug.strip()),
                        ("chroma_dir", str(self._chroma_dir.resolve())),
                    ],
                )

            resolver_notes: list[str] = []
            if cat == FolderCategory.projects and not slug:
                try:
                    slug = (
                        _gemini_resolve_project_root_sync(query, _projects_page_index_slugs()) or ""
                    )
                except Exception as res_exc:
                    slug = ""
                    resolver_notes.append(
                        f"Project auto-resolve skipped ({res_exc}); call list_projects_page_index."
                    )

            if dev_trace.is_dev_mode():
                dev_trace.panel(
                    "tool.search_documents.after_project_resolve",
                    [
                        ("slug_used_for_filter", slug),
                        ("resolver_notes", resolver_notes),
                        ("page_index", _projects_page_index_slugs()),
                    ],
                )

            nodes, filters_used, resolved_slug, meta_notes = _run_search_pass(
                cat=cat, proj_slug=slug, dev_pass_label="primary"
            )
            if not nodes and cat == FolderCategory.projects and slug:
                meta_notes.append(
                    "No vectors under project_root filter—retrying all projects (missing metadata? run rag_indexer.py)."
                )
                nodes, filters_used, resolved_slug, meta_notes = _run_search_pass(
                    cat=cat, proj_slug="", dev_pass_label="fallback_all_projects"
                )

            crag_slugs = "\n".join(f"- {s}" for s in _projects_page_index_slugs())
            run_crag = cat in (FolderCategory.projects, FolderCategory.all)

            def _preview(ns: list, limit: int = 6) -> str:
                lines: list[str] = []
                for sn in ns[:limit]:
                    meta = getattr(sn.node, "metadata", None) or {}
                    fp = meta.get("file_path") or ""
                    prev = (sn.node.get_content(metadata_mode="none") or "").strip().replace("\n", " ")[:240]
                    lines.append(f"{fp} :: {prev}")
                return "\n".join(lines)

            extra_notes: list[str] = resolver_notes + list(meta_notes)

            if run_crag and nodes:
                try:
                    verdict_blob = _crag_classify_sync(
                        user_query=query,
                        folder_category=cat.value,
                        snippets_preview=_preview(nodes),
                        index_slugs=crag_slugs,
                    )
                    if dev_trace.is_dev_mode():
                        dev_trace.panel(
                            "tool.search_documents.crag_evaluation",
                            [
                                ("model", _CRAG_MODEL_ID),
                                ("verdict_json", verdict_blob),
                                ("preview_sent_to_crag", _preview(nodes)),
                            ],
                        )
                    verdict = str(verdict_blob.get("verdict", "GOOD")).upper()
                    reason = str(verdict_blob.get("reason", ""))
                    suggested = verdict_blob.get("suggested_project_root")
                    extra_notes.append(f"CRAG verdict={verdict} ({reason}).")
                    if (
                        verdict == "AMBIGUOUS"
                        and cat == FolderCategory.projects
                        and isinstance(suggested, str)
                        and suggested in _projects_page_index_slugs()
                        and suggested != resolved_slug
                    ):
                        extra_notes.append(f"CRAG follow-up search with project_root={suggested!r}.")
                        nodes, filters_used, resolved_slug, notes2 = _run_search_pass(
                            cat=FolderCategory.projects,
                            proj_slug=suggested,
                            dev_pass_label="crag_followup",
                        )
                        extra_notes.extend(notes2)
                except Exception as crag_exc:
                    extra_notes.append(f"CRAG skipped ({crag_exc}).")
                    if dev_trace.is_dev_mode():
                        dev_trace.panel(
                            "tool.search_documents.crag_error",
                            [("exception", repr(crag_exc))],
                        )

            if not nodes:
                return (
                    "No vectors matched this query for the requested filters. "
                    "If this was projects/, confirm rag_indexer wrote project_root metadata (re-run indexer). "
                    f"PageIndex: {', '.join(_projects_page_index_slugs()) or '(empty)'}"
                )

            if cat == FolderCategory.projects:
                picked = sorted(
                    nodes,
                    key=lambda n: (n.score is not None, n.score or 0.0),
                    reverse=True,
                )[: max(max_snippets * 3, 16)]
                if dev_trace.is_dev_mode():
                    dev_trace.panel(
                        "tool.search_documents.post_rank_projects",
                        [
                            ("picked_for_readme_merge", len(picked)),
                            ("project_sections_cap", project_sections),
                            ("similarity_cutoff", min_sim),
                        ],
                    )
                    dev_trace.rag_hits(picked, label="search_documents.picked_projects")
                blob, fmt_notes = _format_projects_readme_priority(
                    picked,
                    query_str=query,
                    similarity_cutoff=min_sim,
                    max_sections=project_sections,
                )
                extra_notes.extend(fmt_notes)
            else:
                picked, filter_notes = _filter_rag_nodes(
                    nodes,
                    query_str=query,
                    similarity_cutoff=min_sim,
                    max_snippets=max_snippets,
                )
                if dev_trace.is_dev_mode():
                    dev_trace.panel(
                        "tool.search_documents.post_filter_non_projects",
                        [
                            ("picked_snippets", len(picked)),
                            ("max_snippets", max_snippets),
                            ("similarity_cutoff", min_sim),
                        ],
                    )
                    dev_trace.rag_hits(picked, label="search_documents.picked_filtered")
                blob = _format_standard_snippets(
                    picked,
                    folder_category=cat.value,
                    filter_notes=filter_notes + extra_notes,
                )

            if not blob.strip():
                return (
                    "Retrieval ran but produced no readable text after cleanup. "
                    "Try re-indexing or broaden folder_category."
                )

            header = f"[scope folder_category={cat.value}"
            if resolved_slug:
                header += f" project_root={resolved_slug!r}"
            header += "]\n"
            if extra_notes and cat != FolderCategory.projects:
                header += "Notes: " + " ".join(extra_notes) + "\n"
            elif extra_notes:
                header += "Notes: " + " ".join(extra_notes) + "\n"

            if dev_trace.is_dev_mode():
                out = header + blob
                distinct_paths = set()
                for sn in nodes[:80]:
                    md = getattr(sn.node, "metadata", None) or {}
                    fp = md.get("file_path") or ""
                    if fp:
                        distinct_paths.add(fp)
                dev_trace.panel(
                    "tool.search_documents.output",
                    [
                        ("folder_category", cat.value),
                        ("resolved_project_slug", resolved_slug),
                        ("distinct_source_paths", sorted(distinct_paths)),
                        ("response_chars_total", len(out)),
                        ("voice_llm_sees_notes_in_header", True),
                        ("pipeline_notes", extra_notes),
                    ],
                )

            logger.debug("search_documents returning %s chars", len(header + blob))
            return header + blob

        initial = project_folder.strip()
        try:
            return await asyncio.to_thread(_execute_pipeline, initial)
        except Exception as exc:
            logger.exception("search_documents failed")
            if dev_trace.is_dev_mode():
                dev_trace.panel(
                    "tool.search_documents.fatal",
                    [("exception", repr(exc))],
                )
            return f"search_documents error: {exc}. Check Vertex credentials and indexer metadata."


def _attach_dev_trace_room_hooks(room: rtc.Room, *, loop: asyncio.AbstractEventLoop) -> None:
    """Wire UI-driven tracing: JWT metadata, control data packets, and panel mirror."""

    _activation_logged = False

    def publish_wire(payload: dict[str, Any]) -> None:
        """``publish_data`` is async; schedule on the agent loop (tools already run on that loop)."""

        try:
            wire = json.dumps(payload, ensure_ascii=False, default=str)
            if len(wire) > 48_000:
                wire = wire[:48_000] + "…"
            data = wire.encode("utf-8")
        except Exception:
            return

        async def _send() -> None:
            if not room.isconnected():
                return
            try:
                await room.local_participant.publish_data(
                    data,
                    reliable=True,
                    topic=dev_trace.ALIJR_DEV_TRACE_TOPIC,
                )
            except Exception as exc:
                logger.warning("dev_trace publish_data failed: %s", exc)

        def _enqueue_on_agent_loop() -> None:
            try:
                asyncio.ensure_future(_send(), loop=loop)
            except RuntimeError as exc:
                logger.warning("dev_trace could not enqueue publish_data: %s", exc)

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is loop:
            asyncio.create_task(_send())
        else:
            try:
                loop.call_soon_threadsafe(_enqueue_on_agent_loop)
            except RuntimeError as exc:
                logger.warning("dev_trace call_soon_threadsafe failed: %s", exc)

    dev_trace.set_data_publish_hook(publish_wire)

    def log_activation_once(source: str, identity: str, display_name: str) -> None:
        nonlocal _activation_logged
        dev_trace.set_session_override(True)
        if _activation_logged:
            return
        _activation_logged = True
        dev_trace.panel(
            "session.dev_trace.enabled_from_ui",
            [
                ("trigger", source),
                ("participant_identity", identity),
                ("participant_name", display_name),
            ],
        )

    def consume_metadata(participant: rtc.RemoteParticipant, *, source: str) -> None:
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            return
        if not dev_trace.participant_metadata_requests_trace(participant.metadata):
            return
        log_activation_once(
            source,
            participant.identity,
            getattr(participant, "name", "") or "",
        )

    def scan_remote_participants(source: str) -> None:
        for rp in room.remote_participants.values():
            consume_metadata(rp, source=source)

    def on_participant_connected(p: rtc.RemoteParticipant) -> None:
        consume_metadata(p, source="participant_connected")

    def on_participant_metadata_changed(
        participant: rtc.Participant,
        _old_metadata: str,
        _new_metadata: str,
    ) -> None:
        if isinstance(participant, rtc.RemoteParticipant):
            consume_metadata(participant, source="participant_metadata_changed")

    def on_data_received(packet: rtc.DataPacket) -> None:
        if packet.topic != dev_trace.ALIJR_DEV_TRACE_CONTROL_TOPIC:
            return
        try:
            payload = json.loads(packet.data.decode("utf-8"))
        except Exception:
            return
        if payload.get("alijr_dev_trace") is not True:
            return
        ident = packet.participant.identity if packet.participant else ""
        log_activation_once(
            "data_received",
            ident,
            getattr(packet.participant, "name", "") if packet.participant else "",
        )

    room.on("participant_connected", on_participant_connected)
    room.on("participant_metadata_changed", on_participant_metadata_changed)
    room.on("data_received", on_data_received)
    room.on("connected", lambda: scan_remote_participants("room_connected"))
    room.on("reconnected", lambda: scan_remote_participants("room_reconnected"))
    scan_remote_participants("hook_attach")


async def entrypoint(ctx: JobContext) -> None:
    dev_trace.set_session_override(None)
    dev_trace.set_data_publish_hook(None)
    dev_trace.configure_logging()

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
            turn_detection=MultilingualModel(),
        ),
        conn_options=_session_connect_options(),
        tools=[kb],
        max_tool_steps=12,
    )

    agent = Agent(instructions=ALIJR_SYSTEM_PROMPT)

    await session.start(agent=agent, room=ctx.room)
    _attach_dev_trace_room_hooks(ctx.room, loop=asyncio.get_running_loop())
    if dev_trace.is_dev_mode():
        room_name = getattr(ctx.room, "name", "") or "(room)"
        dev_trace.panel(
            "session.start",
            [
                ("room", room_name),
                ("tracing", "active (Developer trace from UI and/or ALIJR_DEV_MODE)"),
                ("note", dev_trace.tool_timing_hint()),
            ],
        )


def main() -> None:
    # Load: > threshold marks worker unavailable for new jobs (prod default 0.7 in SDK). Raise with
    # LIVEKIT_AGENT_LOAD_THRESHOLD=0.85 on a single fat host, or run multiple agent replicas (ECS tasks).
    _load_thr_raw = os.getenv("LIVEKIT_AGENT_LOAD_THRESHOLD", "").strip()
    load_threshold: float | None = None
    if _load_thr_raw:
        try:
            load_threshold = float(_load_thr_raw)
        except ValueError:
            load_threshold = None

    _idle_raw = os.getenv("LIVEKIT_AGENT_NUM_IDLE", "").strip()
    num_idle = 1
    if _idle_raw.isdigit():
        num_idle = max(1, min(8, int(_idle_raw)))

    opts = dict(
        entrypoint_fnc=entrypoint,
        agent_name=os.getenv("LIVEKIT_AGENT_NAME", ""),
        job_executor_type=JobExecutorType.THREAD,
        initialize_process_timeout=60.0,
        num_idle_processes=num_idle,
    )
    if load_threshold is not None:
        opts["load_threshold"] = load_threshold

    cli.run_app(WorkerOptions(**opts))


if __name__ == "__main__":
    main()
