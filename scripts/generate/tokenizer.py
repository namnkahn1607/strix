#!/usr/bin/env python3
"""
Generate BERT Tokenizer ONNX graph from tokenizer.json.
 
Inputs  : model/tokenizer.json + config.json
Outputs : model/tokenizer.onnx
 
Usage:
    uv run python3 generate/tokenizer.py
"""

import subprocess
from pathlib import Path

import onnx
from transformers import AutoTokenizer
from onnxruntime_extensions import gen_processing_models

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(subprocess.run(
    ["git", "-C", str(SCRIPT_DIR), "rev-parse", "--show-toplevel"],
    capture_output=True, text=True, check=True,
).stdout.strip())

assert REPO_ROOT, "Cannot resolve repository root."

MODEL_DIR = REPO_ROOT / "model"
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

tok_model, _ = gen_processing_models(tokenizer, pre_kwargs={})
assert tok_model is not None, (
    "gen_processing_models returned None - "
    "verify that strix/tokenizer.json is a valid HF tokenizer."
)

TOKENIZER = MODEL_DIR / "tokenizer.onnx"
onnx.save(tok_model, str(TOKENIZER))

print(f"Saved  : {TOKENIZER}")
print("Inputs :", [i.name for i in tok_model.graph.input])
print("Outputs:", [o.name for o in tok_model.graph.output])
