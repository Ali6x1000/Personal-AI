"""
FastAPI: mint LiveKit room tokens for the frontend.

Reads LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, and
GOOGLE_APPLICATION_CREDENTIALS from the environment (via python-dotenv when
present). The Google credentials check ensures the backend is configured for
Google services used by the agent.
"""

from __future__ import annotations
import json
import os
import re
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from livekit import api
from pydantic import BaseModel, Field


load_dotenv()

app = FastAPI(title="AliJR LiveKit token API")

_default_origins = "http://localhost:3000,http://127.0.0.1:3000"
_cors_origins = [x.strip() for x in os.getenv("CORS_ALLOW_ORIGINS", _default_origins).split(",") if x.strip()]
_cors_regex = os.getenv("CORS_ALLOW_ORIGIN_REGEX", "").strip()

_cors_kw: dict = dict(
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if _cors_regex:
    _cors_kw["allow_origin_regex"] = _cors_regex

app.add_middleware(CORSMiddleware, allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)


def _require_env(*keys: str) -> dict[str, str]:
    values = {}
    missing = []
    for key in keys:
        val = os.environ.get(key)
        if not val:
            missing.append(key)
        else:
            values[key] = val
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing required environment variables: {', '.join(missing)}",
        )
    return values


class TokenResponse(BaseModel):
    token: str
    livekit_url: str
    room: str
    identity: str


class MintTokenBody(BaseModel):
    room_name: str = Field(..., min_length=1, description="Target LiveKit room name.")
    participant_identity: str = Field(default="frontend-user")
    participant_name: str | None = Field(
        default=None,
        description="Display name; defaults to participant_identity.",
    )
    dev_trace: bool = Field(
        default=False,
        description="Embed alijr_dev_trace in participant metadata so the agent mirrors debug panels.",
    )


def _normalized_web_identity(display_name: str) -> str:
    """Match alijr-frontend slug convention for participant_identity."""

    slug = re.sub(r"[^a-z0-9]+", "-", display_name.lower())
    slug = re.sub(r"^-+|-+$", "", slug)
    return f"web-{slug}-{int(time.time() * 1000)}"


def _mint_access_token_response(
    *,
    room_name: str,
    participant_identity: str,
    participant_name: str | None,
    dev_trace: bool,
) -> TokenResponse:
    env = _require_env(
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "GOOGLE_APPLICATION_CREDENTIALS",
    )

    display = participant_name or participant_identity
    builder = (
        api.AccessToken(env["LIVEKIT_API_KEY"], env["LIVEKIT_API_SECRET"])
        .with_identity(participant_identity)
        .with_name(display)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
    )
    if dev_trace:
        builder = builder.with_metadata(json.dumps({"alijr_dev_trace": True}))
    token_jwt = builder.to_jwt()

    return TokenResponse(
        token=token_jwt,
        livekit_url=env["LIVEKIT_URL"],
        room=room_name,
        identity=participant_identity,
    )

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "AliJR is online"}

@app.get("/getToken", response_model=TokenResponse)
async def get_participant_token(
    name: str = Query(..., min_length=1, description="Display name for the browser participant."),
    room_name: str = Query(default="alijr-test", min_length=1),
    dev_trace: bool = Query(default=False),
) -> TokenResponse:
    """Simple GET mint for Vercel / curl; identity is generated server-side."""

    pid = _normalized_web_identity(name)
    return _mint_access_token_response(
        room_name=room_name,
        participant_identity=pid,
        participant_name=name,
        dev_trace=dev_trace,
    )


@app.post("/token", response_model=TokenResponse)
async def mint_participant_token(body: MintTokenBody) -> TokenResponse:
    return _mint_access_token_response(
        room_name=body.room_name,
        participant_identity=body.participant_identity,
        participant_name=body.participant_name,
        dev_trace=body.dev_trace,
    )
