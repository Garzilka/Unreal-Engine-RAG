#!/usr/bin/env bash
set -e

# --- НАСТРОЙКИ ПО УМОЛЧАНИЮ ---
# Можно переопределить при запуске: ./update_index.sh "/path/to/repo" "MyProject"
DIR="${1:-/path/to/your/ue_project}"
NAME="${2:-RAG_NAME}"

# Автоматически определяем директорию, где лежит сам скрипт
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SCRIPT_DIR}/venv/bin/python3"
BUILDER_SCRIPT="${SCRIPT_DIR}/RagMCP.py"

# Если виртуальное окружение не найдено, используем системный python3
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

if [ ! -d "$DIR" ]; then
    echo "[!] Ошибка: директория с исходниками '$DIR' не найдена."
    exit 1
fi

# Обновляем репозиторий, если это git-папка
if [ -d "$DIR/.git" ]; then
    echo "[*] Обновляем код в $DIR..."
    git -C "$DIR" pull
fi

# Переходим в папку со скриптами и индексами
cd "$SCRIPT_DIR"

echo "[*] Очищаем старые файлы индекса для '$NAME'..."
rm -f "${NAME}_faiss.index" "${NAME}_chunks.jsonl" "${NAME}_raw.jsonl"

echo "[*] Строим новый RAG-индекс: $NAME..."
"$PYTHON_BIN" "$BUILDER_SCRIPT" --input-dir "$DIR" --output-name "$NAME"

echo "[OK] Индекс '$NAME' успешно обновлён."
