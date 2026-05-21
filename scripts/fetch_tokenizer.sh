#!/bin/bash
set -euo pipefail

# Directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_ROOT/model"
TOKENIZER_PATH="$MODEL_DIR/tokenizer.json"
TOKENIZER_URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"

# Python available check
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found in PATH." >&2
    exit 1
fi

mkdir -p "$MODEL_DIR"

# Step 1: Download dictionary and configuration file
echo "[1/2] Downloading tokenizer.json..."
curl -fSL --progress-bar -o "$TOKENIZER_PATH" "$TOKENIZER_URL"

# Step 2: JSON verification
echo "[2/2] Verifying JSON integrity..."
if python3 -c "import json; json.load(open('$TOKENIZER_PATH'))" 2>/dev/null; then
    echo "  Integrity OK: 'tokenizer.json'."
else
    echo "ERROR: tokenizer.json is not valid JSON - file corrupted or truncated." >&2
    rm -f "$TOKENIZER_PATH"
    exit 1
fi

echo ""
echo "Done. 'tokenizer.json' saved to $TOKENIZER_PATH"
