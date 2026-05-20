#!/bin/bash
set -euo pipefail

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_ROOT/model"
MODEL_NAME="transformer.onnx"
MODEL_URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model_quint8_avx2.onnx"
EXPECTED_SHA256="b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21"

mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR" || exit 1

# Step 1: Download inference model ONNX file
echo "[1/2] Downloading $MODEL_NAME..."
curl -fSL -o "$MODEL_DIR/$MODEL_NAME" "$MODEL_URL"

# Step 2: Checsum verification
echo "[2/2] Verifying checksum..."
ACTUAL_SHA256=$(sha256sum "$MODEL_DIR/$MODEL_NAME" | awk '{ print $1 }')
 
if [ "$EXPECTED_SHA256" != "$ACTUAL_SHA256" ]; then
    echo "ERROR: Checksum mismatch! File corrupted." >&2
    rm -f "$MODEL_DIR/$MODEL_NAME"
    exit 1
fi

echo ""
echo "Done. Model downloaded and verified succesfully"
