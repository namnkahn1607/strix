#!/bin/bash
set -euo pipefail

# Directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
MODEL_DIR="$PROJECT_ROOT/model"
PREPROCESS_SCRIPT="$SCRIPT_DIR/pp_codealpaca.py"
FETCH_TOKENIZER_SCRIPT="$SCRIPT_DIR/fetch_tokenizer.sh"
TOKENIZER_PATH="$MODEL_DIR/tokenizer.json"

# Dataset configurations
REPO="sahil2801/CodeAlpaca-20k"
FILE_NAME="code_alpaca_20k.json"
PROCESSED_NAME="codealpaca-20k.jsonl"
DOWNLOAD_URL="https://huggingface.co/datasets/$REPO/resolve/main/$FILE_NAME"

RAW_PATH="$DATA_DIR/$FILE_NAME"
PROCESSED_PATH="$DATA_DIR/$PROCESSED_NAME"

# Check Python script existence
if [[ ! -f "$PREPROCESS_SCRIPT" ]]; then
    echo "ERROR: pp_dolly.py not found at $PREPROCESS_SCRIPT" >&2
    exit 1
fi

# Check dictionary fetching script existence
if [[ ! -f "$FETCH_TOKENIZER_SCRIPT" ]]; then
    echo "ERROR: fetch_tokenizer.sh not found at $FETCH_TOKENIZER_SCRIPT" >&2
    exit 1
fi

# Check if Python3 is available
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found in PATH." >&2
    exit 1
fi

# Check if package 'tokenizers' is available
if ! python3 -c "import tokenizers" &>/dev/null; then
    echo "ERROR: Python package 'tokenizers' is not installed." >&2
    echo "       Run: pip install tokenizers" >&2
    exit 1
fi

mkdir -p "$DATA_DIR"

# Step 1: Ensure 'tokenizer.json' is present
if [[ ! -f "$TOKENIZER_PATH" ]]; then
    echo "[1/4] tokenizer.json not found — fetching..."
    bash "$FETCH_TOKENIZER_SCRIPT"
else
    echo "[1/4] tokenizer.json already present, skipping download."
fi

# Step 2: Download dataset
echo "[2/4] Downloading $FILE_NAME..."
curl -fSL --progress-bar -o "$RAW_PATH" "$DOWNLOAD_URL"

# Step 3: Validating integrity
echo "[3/4] Verifying JSON integrity..."
if python3 -c "import json; json.load(open('$RAW_PATH'))" 2>/dev/null; then
    echo "  Integrity OK."
else
    echo "ERROR: $FILE_NAME is not valid JSON - file corrupted or truncated." >&2
    rm -f "$RAW_PATH"
    exit 1
fi

# Step 4: Dataset preprocessing
echo "[4/4] Preprocessing dataset..."
python3 "$PREPROCESS_SCRIPT" "$RAW_PATH" "$PROCESSED_PATH" "$TOKENIZER_PATH"

rm -rf "$RAW_PATH"

echo ""
echo "Done. Output file has been written to $PROCESSED_PATH"
