#!/bin/bash
set -euo pipefail

# Directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_ROOT/model"
CONFIG_PATH="$MODEL_DIR/tokenizer_config.json"
TOKENIZER_PATH="$MODEL_DIR/tokenizer.json"
VOCAB_PATH="$MODEL_DIR/vocab.txt"
CONFIG_URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json"
TOKENIZER_URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
VOCAB_URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt"

# Python available check
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found in PATH." >&2
    exit 1
fi

mkdir -p "$MODEL_DIR"

# Step 1: Download dictionary and configuration file
echo "[1/2] Downloading tokenizer.json..."
curl -fSL --progress-bar -o "$CONFIG_PATH" "$CONFIG_URL"
curl -fSL --progress-bar -o "$TOKENIZER_PATH" "$TOKENIZER_URL"
curl -fSL --progress-bar -o "$VOCAB_PATH" "$VOCAB_URL"

# Step 2: JSON verification
echo "[2/2] Verifying JSON integrity..."

# tokenizer_config.json
if python3 -c "import json; json.load(open('$CONFIG_PATH'))" 2>/dev/null; then
    echo "  Integrity OK: 'tokenizer_config.json'."
else
    echo "ERROR: tokenizer_config.json is not valid JSON - file corrupted or truncated." >&2
    rm -f "$CONFIG_PATH"
    exit 1
fi

# tokenizer.json
if python3 -c "import json; json.load(open('$TOKENIZER_PATH'))" 2>/dev/null; then
    echo "  Integrity OK: 'tokenizer.json'."
else
    echo "ERROR: tokenizer.json is not valid JSON - file corrupted or truncated." >&2
    rm -f "$TOKENIZER_PATH"
    exit 1
fi

echo ""
echo "Done." 
echo "'tokenizer.json', 'tokenizer_config.json' and 'vocab.txt' saved to $TOKENIZER_PATH"
