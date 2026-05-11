#!/bin/bash
set -e
# .env often contains a developer Mac path. In the image only /app/cred/*.json exists — fix before Python loads dotenv.
if [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
  unset GOOGLE_APPLICATION_CREDENTIALS
fi
if [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && [ -d /app/cred ]; then
  for f in /app/cred/*.json; do
    if [ -f "$f" ]; then
      export GOOGLE_APPLICATION_CREDENTIALS="$f"
      break
    fi
  done
fi
if [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] || [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
  echo "ERROR: GOOGLE_APPLICATION_CREDENTIALS must point to a readable JSON file under /app/cred/ (e.g. /app/cred/alijr.json)." >&2
  exit 1
fi

python agent.py start &
# Do not `exec` uvicorn — the shell must stay parent of the background agent process.
UVICORN_WORKERS="${UVICORN_WORKERS:-4}"
uvicorn server:app --host 0.0.0.0 --port 8000 --workers "${UVICORN_WORKERS}"
