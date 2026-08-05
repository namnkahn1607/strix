#!/usr/bin/env python3
"""
Preprocess (reformat) raw sahil2801/CodeAlpaca-20k dataset.
Source: https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k

Input schema  : JSON array of { output, instruction, input }
Output schema : JSONL of { prompt, payload }]

Tokenizer: loaded from strix/model/tokenizer.json (all-MiniLM-L6-v2).

Prompt construction & filtering:
  - prompt  = instruction + "\n" + input  (if input is non-empty)
            = instruction                 (otherwise)
  - payload = output

NOTE:
  NO fallback chain: `input` is a primary operand (code snippet, function
  signature, constraints), not optional context. Therefore discard if 'prompt'
  token exceeds MAX_PROMPT_TOKENS.

  The only 'soft' case is when input is empty to begin with - then instruction
  alone is the full prompt.

Usage:
    uv run python3 datasets/codealpaca-20k/preprocess.py <input-path> <output-path>
"""

import json
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import MAX_BOUND_TOKENS
from _preprocess import count_tokens, entrypoint
from tokenizers import Tokenizer

def iter_raw_records(input_path: Path) -> Iterator[dict]:
    with input_path.open("r", encoding="utf-8") as fin:
        records = json.load(fin)

    if not isinstance(records, list):
        raise ValueError(
            f"Expected a JSON array at top level, got {type(records).__name__}"
        )

    yield from records

def build_prompt(instruction: str, input_: str) -> str:
    instruction = instruction.strip()
    input_ = input_.strip()

    if input_:
        return f"{instruction}\n{input_}"

    return instruction

def process_record(record: dict,
                   tokenizer: Tokenizer) -> tuple[dict | None, str | None]:
    instruction: str = record.get("instruction", "")
    input_: str = record.get("input", "")
    output_: str = record.get("output", "")

    if not output_.strip():
        return None, "empty_output"

    prompt = build_prompt(instruction, input_)
    if not prompt:
        return None, "empty_prompt"

    if count_tokens(tokenizer, prompt) > MAX_BOUND_TOKENS:
        return {"prompt": prompt, "payload": output_}, "overlimit"

    return {"prompt": prompt, "payload": output_}, None

if __name__ == "__main__":
    sys.exit(entrypoint(iter_raw_records, process_record, description=__doc__))
