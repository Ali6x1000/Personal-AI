"""
FastAPI: mint LiveKit room tokens for the frontend.

Reads LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, and GOOGLE_API_KEY from the
environment (via python-dotenv when present). All four must be set before tokens
are issued so deployment mistakes surface early.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from livekit import api
from pydantic import BaseModel, Field


load_dotenv()

app = FastAPI(title="AliJR LiveKit token API")


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


@app.post("/token", response_model=TokenResponse)
async def mint_participant_token(body: MintTokenBody) -> TokenResponse:
    env = _require_env(
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "GOOGLE_API_KEY",
    )

    token = (
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
        .to_jwt()
    )

    return TokenResponse(
        token=token,
        livekit_url=env["LIVEKIT_URL"],
        room=body.room_name,
        identity=body.participant_identity,
    )
