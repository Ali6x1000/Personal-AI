"""
AliJR voice agent for the portfolio website.

Speech-to-speech with the Gemini Live API (authenticated with a plain Gemini API key, no
service account). Knowledge lookups go through the portfolio's /api/alijr/tools endpoint,
so the voice agent and the text chat share one retrieval implementation and one pgvector
database.

Required env: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET (injected automatically on
LiveKit Cloud), GOOGLE_API_KEY, ALIJR_SITE_URL, ALIJR_AGENT_SECRET.
"""

from __future__ import annotations

import asyncio
import logging
import os
from enum import Enum

import httpx
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    function_tool,
)
from livekit.plugins import google

load_dotenv()
load_dotenv(".env.local")

logger = logging.getLogger("alijr.voice")

AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "alijr-voice")
SITE_URL = os.getenv("ALIJR_SITE_URL", "http://localhost:3000").rstrip("/")
AGENT_SECRET = os.getenv("ALIJR_AGENT_SECRET", "")
VOICE_MODEL = os.getenv("ALIJR_VOICE_MODEL", "")
VOICE_NAME = os.getenv("ALIJR_VOICE", "Puck")
MAX_SESSION_SECONDS = int(os.getenv("ALIJR_MAX_SESSION_SECONDS", "600"))

VOICE_PROMPT = """
You are AliJR (Ali Junior), the digital twin of Ali Nawaf, built by Ali and talking with a visitor
on his portfolio website. Speak about Ali's work in first person ("I built", "my research"), but
never claim to be a human.

Tools:
- Resume, employers, dates, education, skills, contact details: call read_resume and answer only from it.
- Coding projects and repositories: call search_knowledge with folder_category "projects". If unsure
  which repo the visitor means, call list_projects first and pass the matching project_folder.
- Coursework, lectures, papers and assignments: call search_knowledge with course_lectures, research,
  submissions, or all.
- Before a tool call, say a short natural filler like "Let me check" so the visitor is not left in silence.
- Never invent facts. If the tools do not have the answer, say so and suggest emailing aan90@case.edu.

Recruiters: if someone is a recruiter or asks why they should hire Ali, highlight how he bridges
advanced ML research and production AI engineering, his integrated BS/MS at Case Western, and shipped
systems like the Trek Health RAG pipeline and his AI safety work.

Voice style: warm, upbeat, concise. Two or three sentences per turn unless asked for detail. No lists,
URLs, markdown or emoji. Speak contact details naturally: email aan90@case.edu, website alinawaf.com.
""".strip()


class FolderCategory(str, Enum):
    projects = "projects"
    research = "research"
    course_lectures = "course_lectures"
    submissions = "submissions"
    all = "all"


class SiteTools:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=SITE_URL,
            timeout=httpx.Timeout(25.0),
            headers={"Authorization": f"Bearer {AGENT_SECRET}"},
        )

    async def call(self, tool: str, **args: object) -> str:
        try:
            res = await self._client.post("/api/alijr/tools", json={"tool": tool, "args": args})
            res.raise_for_status()
            return str(res.json().get("result", ""))
        except Exception:
            logger.exception("tool %s failed", tool)
            return "The knowledge base is unavailable right now. Answer briefly from what you know."

    async def aclose(self) -> None:
        await self._client.aclose()


class AliJR(Agent):
    def __init__(self, tools: SiteTools) -> None:
        super().__init__(instructions=VOICE_PROMPT)
        self._tools = tools

    @function_tool
    async def read_resume(self, context: RunContext) -> str:
        """Load Ali's full resume: education, experience, skills and contact details."""
        return await self._tools.call("readResume")

    @function_tool
    async def list_projects(self, context: RunContext) -> str:
        """List the project folders in Ali's knowledge base."""
        return await self._tools.call("listProjects")

    @function_tool
    async def search_knowledge(
        self,
        context: RunContext,
        query: str,
        folder_category: FolderCategory,
        project_folder: str = "",
    ) -> str:
        """Semantic search over Ali's project READMEs, research, course lectures and submissions.

        Args:
            query: A focused search query.
            folder_category: Which part of the knowledge base to search.
            project_folder: Exact project folder from list_projects when searching projects.
        """
        return await self._tools.call(
            "searchKnowledge",
            query=query,
            folderCategory=folder_category.value,
            projectFolder=project_folder or None,
        )


server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    tools = SiteTools()
    ctx.add_shutdown_callback(tools.aclose)

    model_kwargs: dict = {"voice": VOICE_NAME}
    if VOICE_MODEL:
        model_kwargs["model"] = VOICE_MODEL

    session = AgentSession(llm=google.realtime.RealtimeModel(**model_kwargs))
    await session.start(agent=AliJR(tools), room=ctx.room)
    await session.generate_reply(
        instructions="Greet the visitor in one short sentence as AliJR and ask what they'd like to know about Ali."
    )

    async def enforce_time_limit() -> None:
        await asyncio.sleep(MAX_SESSION_SECONDS)
        await session.generate_reply(
            instructions="Politely say the call time limit was reached and they can keep chatting by text."
        )
        await asyncio.sleep(8)
        ctx.shutdown(reason="session time limit")

    limit = asyncio.create_task(enforce_time_limit())

    async def cancel_limit() -> None:
        limit.cancel()

    ctx.add_shutdown_callback(cancel_limit)


if __name__ == "__main__":
    cli.run_app(server)
