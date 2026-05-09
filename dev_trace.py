"""
Developer tracing for AliJR (off by default).

1. **UI (primary):** Enable **Developer trace** in the web app — the browser publishes a short control
   packet and/or joins with token metadata ``{"alijr_dev_trace": true}``. No worker env vars required.
2. **Environment (optional):** ``ALIJR_DEV_MODE=1`` forces tracing on for every session on that worker.
   Compact panels can mirror over LiveKit **data** topic ``alijr-dev-trace``.

Optional:
- ``ALIJR_DEV_VERBOSE=1`` — longer chunk excerpts.
- ``ALIJR_DEV_RESUME_FULL=1`` — log full resume (default is a short preview for privacy).
- ``ALIJR_DEV_WIRE_MAX_CHARS`` — cap JSON field size when publishing to the UI (default 900).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable

_logger = logging.getLogger("alijr.dev")
_configured = False
_session_override: bool | None = None
_data_publish_hook: Callable[[dict[str, Any]], None] | None = None

# Must match the topic filtered in ``alijr-frontend`` DevTraceFeed.
ALIJR_DEV_TRACE_TOPIC = "alijr-dev-trace"
# Browser → agent: enable tracing without relying on JWT metadata timing.
ALIJR_DEV_TRACE_CONTROL_TOPIC = "alijr-dev-trace-control"


def set_session_override(enabled: bool | None) -> None:
    """``True`` / ``False`` forces tracing on/off; ``None`` falls back to ``ALIJR_DEV_MODE`` env."""

    global _session_override
    _session_override = enabled
    if enabled is True:
        configure_logging()


def set_data_publish_hook(fn: Callable[[dict[str, Any]], None] | None) -> None:
    """Agent registers a hook to ``Room.local_participant.publish_data`` for UI mirroring."""

    global _data_publish_hook
    _data_publish_hook = fn


def is_dev_mode() -> bool:
    if _session_override is True:
        return True
    if _session_override is False:
        return False
    return os.getenv("ALIJR_DEV_MODE", "").strip().lower() in ("1", "true", "yes", "on")


def dev_verbose() -> bool:
    return os.getenv("ALIJR_DEV_VERBOSE", "").strip().lower() in ("1", "true", "yes", "on")


def resume_full() -> bool:
    return os.getenv("ALIJR_DEV_RESUME_FULL", "").strip().lower() in ("1", "true", "yes", "on")


def configure_logging() -> None:
    global _configured
    if _configured or not is_dev_mode():
        return
    _configured = True
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)


def _truncate_wire_text(s: str, lim: int) -> str:
    if len(s) <= lim:
        return s
    return s[: lim - 1] + "…"


def _sanitize_wire_val(val: Any, max_len: int) -> Any:
    if isinstance(val, (dict, list)):
        raw = json.dumps(val, ensure_ascii=False, default=str)
        return _truncate_wire_text(raw, max_len)
    return _truncate_wire_text(str(val), max_len)


def _maybe_publish_wire(envelope: dict[str, Any]) -> None:
    if _data_publish_hook is None or not is_dev_mode():
        return
    max_len = int(os.getenv("ALIJR_DEV_WIRE_MAX_CHARS", "900"))
    try:
        payload = {"v": 1, "ts": time.time(), **envelope}
        _data_publish_hook(payload)
    except Exception:
        pass


def participant_metadata_requests_trace(metadata: str | None) -> bool:
    """Parse LiveKit participant metadata JSON from the web token."""

    if not metadata or not str(metadata).strip():
        return False
    try:
        obj = json.loads(metadata)
    except json.JSONDecodeError:
        return False
    flag = obj.get("alijr_dev_trace")
    if flag is True:
        return True
    if isinstance(obj.get("alijr"), dict) and obj["alijr"].get("dev_trace") is True:
        return True
    return str(flag).strip().lower() in ("1", "true", "yes", "on")


def _fmt_val(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        return json.dumps(val, indent=2, ensure_ascii=False, default=str)
    return str(val)


def panel(title: str, rows: list[tuple[str, Any]]) -> None:
    """Print a framed block (ASCII-only, log-safe)."""

    if not is_dev_mode():
        return
    width = 78
    bar = "=" * width
    rule = "-" * width
    out = [bar, f"  AliJR DEV  |  {title}", rule]
    for key, val in rows:
        block = _fmt_val(val)
        out.append(f"  · {key}")
        for line in block.splitlines() or [""]:
            out.append(f"      {line}")
    out.append(bar)
    _logger.info("\n".join(out))
    max_len = int(os.getenv("ALIJR_DEV_WIRE_MAX_CHARS", "900"))
    slim_rows: list[list[Any]] = []
    for key, val in rows:
        slim_rows.append([key, _sanitize_wire_val(val, max_len)])
    _maybe_publish_wire({"type": "panel", "title": title, "rows": slim_rows})


def subsection(title: str) -> None:
    if not is_dev_mode():
        return
    _logger.info("%s\n  --- %s ---\n%s", "-" * 72, title, "-" * 72)


def summarize_metadata_filters(filters: Any | None) -> str:
    if filters is None:
        return "(none — all folders)"
    try:
        from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
    except Exception:
        return repr(filters)

    def walk(obj: Any, depth: int = 0) -> str:
        pad = "  " * depth
        if isinstance(obj, MetadataFilter):
            op = getattr(obj, "operator", None)
            if op:
                return f"{pad}{obj.key} {op} {obj.value!r}"
            return f"{pad}{obj.key} == {obj.value!r}"
        if isinstance(obj, MetadataFilters):
            cond = getattr(obj, "condition", None) or "and"
            nested = "\n".join(walk(f, depth + 1) for f in obj.filters)
            return f"{pad}<{cond}>\n{nested}"
        return f"{pad}{obj!r}"

    return walk(filters)


def excerpt_limit_default() -> int:
    return 900 if dev_verbose() else 220


def rag_hits(nodes: list[Any], *, label: str, excerpt_chars: int | None = None) -> None:
    if not is_dev_mode():
        return
    lim = excerpt_chars if excerpt_chars is not None else excerpt_limit_default()
    rows: list[dict[str, Any]] = []
    show_max = int(os.getenv("ALIJR_DEV_RAG_ROWS", "24"))
    for i, sn in enumerate(nodes[:show_max], start=1):
        meta = getattr(sn.node, "metadata", None) or {}
        raw = (sn.node.get_content(metadata_mode="none") or "").strip()
        flat = " ".join(raw.split())
        excerpt = flat[:lim] + ("…" if len(flat) > lim else "")
        rows.append(
            {
                "#": i,
                "score": round(float(sn.score), 4) if sn.score is not None else None,
                "folder_category": meta.get("folder_category"),
                "project_root": meta.get("project_root") or "",
                "project_name": meta.get("project_name") or "",
                "file_name": meta.get("file_name") or "",
                "file_path": meta.get("file_path") or "",
                "excerpt": excerpt,
            }
        )
    tail = f"{len(nodes)} total nodes" if len(nodes) > show_max else f"{len(nodes)} nodes"
    panel(f"{label} ({tail})", [("retrieval_hits", rows)])


def tool_timing_hint() -> str:
    return (
        "Tracing: stderr logger ``alijr.dev``, UI mirror on data topic "
        f"``{ALIJR_DEV_TRACE_TOPIC}`` (toggle off in the web UI for production)."
    )
