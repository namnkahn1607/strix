#!/usr/bin/env bash
################################################################################
# This script fetches the tokenizing dictionary of all-MiniLM-L6-v2, then create
# a configuration file for 'transformers' lib.
#
# Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# Usage: bash scripts/fetch/tokenizer.sh
# Output:
#   strix/model/tokenizer.json
#   strix/model/config.json
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)" # strix/
cd "$REPO_ROOT"

MODEL_DIR="$REPO_ROOT/model"
mkdir -p "$MODEL_DIR"

DICTIONARY="$MODEL_DIR/tokenizer.json"

log() { echo "[INFO] $*" >&2; }
warn() { echo "[WARN] $*" >&2; }
die() { echo "[ERROR] $*" >&2; exit 1; }

command -v python3 >/dev/null 2>&1 || die "python3 not found."

# ==============================================================================
# Step 1. Download the JSON dictionary
# ==============================================================================
URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"

log "Downloading the dictionary."
curl -fsSL --progress-bar -o "$DICTIONARY" "$URL"

# ==============================================================================
# Step 2. JSON verification
# ==============================================================================
if python3 -c "import json; json.load(open('$DICTIONARY'))" >/dev/null; then
    log "Integrity OK: 'tokenizer.json'."
else
    rm -f "$DICTIONARY"
    die "'tokenizer.json' is not valid JSON - file corrupted or truncated."
fi

# ==============================================================================
# Step 3. Generate configuration file
# ==============================================================================
CONFIG_FILE="$MODEL_DIR/config.json"
cat << 'EOF' > "$CONFIG_FILE"
{"model_type": "bert"}
EOF

log "Done. All saved to strix/model/"
