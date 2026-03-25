#!/bin/bash
# Start the Reliability Analysis GUI
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Starting backend..."
cd "$REPO_DIR/gui/backend"
"$REPO_DIR/.venv/bin/uvicorn" main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

echo "Starting frontend..."
cd "$REPO_DIR/gui/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
