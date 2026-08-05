#!/usr/bin/env bash
################################################################################
# This script downloads the quantitized UINT8 ONNX binary of all-MiniLM-L6-v2,
# optimized using AVX2 intrinsics.
#
# Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# Usage: bash scripts/fetch/transformer.sh
# Output: strix/model/transformer.onnx
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)" # strix/
cd "$REPO_ROOT"

MODEL_DIR="$REPO_ROOT/model"
mkdir -p "$MODEL_DIR"

TRANSFORMER="$MODEL_DIR/transformer.onnx"

log() { echo "[INFO] $*" >&2; }
warn() { echo "[WARN] $*" >&2; }
die() { echo "[ERROR] $*" >&2; exit 1; }

# ==============================================================================
# Step 1. Download the target ONNX binary
# ==============================================================================
URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model_quint8_avx2.onnx"

log "Downloading the binary. Might take a while."
curl -fsSL --progress-bar -o "$TRANSFORMER" "$URL"

# ==============================================================================
# Step 2. Checksum verification
# ==============================================================================
EXPECTED_SHA256="b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21"
ACTUAL_SHA256=$(sha256sum "$TRANSFORMER" | awk '{ print $1 }')

if [ "$EXPECTED_SHA256" != "$ACTUAL_SHA256" ]; then
    rm -f "$TRANSFORMER"
    die "Checksum mismatch! File corrupted." >&2
fi

log "Done. Model download and verified. Saved to strix/model/"
