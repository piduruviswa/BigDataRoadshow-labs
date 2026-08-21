#!/usr/bin/env bash
# Boots the MEKO Fraud Detection demo: FastAPI backend + the
# "Schwab Unified Fraud & SecOps Console" frontend, served from one process.
set -euo pipefail
cd "$(dirname "$0")/backend"

if [ ! -d .venv ]; then
  echo "Creating virtualenv..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt

echo ""
echo "MEKO Fraud Detection Platform"
echo "  Console:  http://localhost:8000/"
echo "  API docs: http://localhost:8000/docs"
echo ""
exec uvicorn demo_app:app --reload --port 8000
