#!/bin/bash
set -eo pipefail

# Decode GCP service-account JSON from ECS / Secrets Manager (base64).
echo "$GOOGLE_B64_CREDS" | base64 -d > /app/google_creds.json
export GOOGLE_APPLICATION_CREDENTIALS="/app/google_creds.json"

python agent.py start &
exec uvicorn server:app --host 0.0.0.0 --port 8000
