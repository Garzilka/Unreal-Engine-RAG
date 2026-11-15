#!/usr/bin/env bash
set -e

# --- НАСТРОЙКИ ---
SERVER_NAME="${1:-UE_RAG_MCP}"
BASE_NAME="${2:-RAG_NAME}"
INPUT_DIR="${3:-/path/to/your/ue_project}"
PORT="${4:-8766}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SCRIPT_DIR}/venv/bin/python3"
SERVER_SCRIPT="${SCRIPT_DIR}/RunMCP.py"

if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

# Переходим в папку, где лежат индексы
cd "$SCRIPT_DIR"

echo "[*] Запуск MCP-сервера '$SERVER_NAME' (индекс: $BASE_NAME) на порту $PORT..."
exec "$PYTHON_BIN" "$SERVER_SCRIPT" \
    --server-name "$SERVER_NAME" \
    --base-name "$BASE_NAME" \
    --input-dir "$INPUT_DIR" \
    --port "$PORT"
