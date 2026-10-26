#!/usr/bin/env bash
set -e

echo "============================================================================"
echo "  DawatDesk Voice Concierge - Haveli Grill & Courtyard (F-7 Markaz)"
echo "  Starting Custom Non-LLM Brain Backend (Port 8000) & React UI (Port 5173)"
echo "============================================================================"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

(
  cd "$ROOT_DIR/backend"
  if [ -f ".venv/bin/uvicorn" ]; then
    .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
  else
    uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
  fi
) &
BACKEND_PID=$!

(
  cd "$ROOT_DIR/frontend"
  npm run dev
) &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true" EXIT
wait
