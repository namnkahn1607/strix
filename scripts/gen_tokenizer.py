#!/usr/bin/env python3
"""
Generate BERT Tokenizer ONNX graph from tokenizer.json.
 
Reads  : model/tokenizer.json  (+ tokenizer_config.json if present)
Outputs: model/tokenizer.onnx
 
Usage:
    python3 scripts/gen_tokenizer.py
"""

from pathlib import Path

import onnx
from transformers import AutoTokenizer
from onnxruntime_extensions import gen_processing_models

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = ROOT / "model"
OUTPUT_ONNX = TOKENIZER_DIR / "tokenizer.onnx"

tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

tok_model, _ = gen_processing_models(
    tokenizer,
    pre_kwargs={},
)

assert tok_model is not None, (
    "gen_processing_models returned None - "
    "verify that model/tokenizer.json is a valid HuggingFace tokenizer."
)

onnx.save(tok_model, str(OUTPUT_ONNX))
print(f"Saved  : {OUTPUT_ONNX}")
print("Inputs :", [i.name for i in tok_model.graph.input])
print("Outputs:", [o.name for o in tok_model.graph.output])
