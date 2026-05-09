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

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from livekit import api
from pydantic import BaseModel, Field


load_dotenv()

app = FastAPI(title="AliJR LiveKit token API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post("/token", response_model=TokenResponse)
async def mint_participant_token(body: MintTokenBody) -> TokenResponse:
    env = _require_env(
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "GOOGLE_APPLICATION_CREDENTIALS",
    )

    builder = (
        api.AccessToken(env["LIVEKIT_API_KEY"], env["LIVEKIT_API_SECRET"])
        .with_identity(body.participant_identity)
        .with_name(body.participant_name or body.participant_identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=body.room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
    )
    if body.dev_trace:
        builder = builder.with_metadata(json.dumps({"alijr_dev_trace": True}))
    token = builder.to_jwt()

    return TokenResponse(
        token=token,
        livekit_url=env["LIVEKIT_URL"],
        room=body.room_name,
        identity=body.participant_identity,
    )
