#!/bin/bash
set -euo pipefail

# Directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
DATA_DIR="$PROJECT_ROOT/data"
MODEL_DIR="$PROJECT_ROOT/model"
PREPROCESS_SCRIPT="$SCRIPT_DIR/pp_dolly.py"
FETCH_TOKENIZER_SCRIPT="$SCRIPT_DIR/fetch_tokenizer.sh"
TOKENIZER_PATH="$MODEL_DIR/tokenizer.json"

# Dataset configurations
DATA_NAME="databricks-dolly-15k.jsonl"
PROCESSED_NAME="dolly-15k.jsonl"
DATA_URL="https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
EXPECTED_SHA256="2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"

RAW_PATH="$DATA_DIR/$DATA_NAME"
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
echo "[2/4] Downloading $DATA_NAME..."
curl -fSL --progress-bar -o "$RAW_PATH" "$DATA_URL"

# Step 3: Verify checksum
echo "[3/4] Verifying SHA-256 checksum..."
ACTUAL_SHA256=$(sha256sum "$RAW_PATH" | awk '{ print $1 }')

if [ "$EXPECTED_SHA256" != "$ACTUAL_SHA256" ]; then
    echo "ERROR: Checksum mismatch - file may be corrupted or tampered." >&2
    echo "  Expected : $EXPECTED_SHA256" >&2
    echo "  Got      : $ACTUAL_SHA256" >&2
    rm -f "$RAW_PATH"
    exit 1
fi

echo "Checksum OK."

# Step 4: Dataset preprocessing
echo "[4/4] Preprocessing dataset..."
python3 "$PREPROCESS_SCRIPT" "$RAW_PATH" "$PROCESSED_PATH" "$TOKENIZER_PATH"

rm -rf "$RAW_PATH"

echo ""
echo "Done. Output file has been written to $PROCESSED_PATH"
