#!/usr/bin/env python3
"""
Preprocess (reformat) raw databricks/databricks-dolly-15k dataset.
Source: https://huggingface.co/datasets/databricks/databricks-dolly-15k

Input schema  : { instruction, context, response, category }
Output schema : { prompt, payload }

Tokenizer: loaded from strix/model/tokenizer.json (all-MiniLM-L6-v2). 

Prompt construction & filtering:
  1. Try prompt = instruction + " " + context (stripped)
  2. If len(tokens) <= 256 -> emit (payload = response)
  3. Else try prompt = instruction only
  4. If len(tokens) <= 256 -> emit (payload = response, context dropped)
  5. Else discard the record entirely

Usage:
    uv run python3 datasets/dolly-15k/preprocess.py \\ 
        <input-path.raw> <output-path.jsonl>
"""

import json
import logging
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import MAX_PROMPT_TOKENS
from _preprocess import count_tokens, entrypoint
from tokenizers import Tokenizer

log = logging.getLogger(__name__)

def iter_raw_records(input_path: Path) -> Iterator[dict]:
    with input_path.open("r", encoding="utf-8") as fin:
        for lineno, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning(
                    "Malformed JSON at line %d - skipping. (%s)", lineno, exc,
                )

def build_prompt(instruction: str, context: str) -> str:
    instruction = instruction.strip()
    context = context.strip()

    if context:
        return f"{instruction} {context}"
    
    return instruction

def process_record(record: dict, tokenizer: Tokenizer) -> tuple[dict | None, str | None]:
    instruction: str = record.get("instruction", "")
    context: str = record.get("context", "")
    response: str = record.get("response", "")

    if not response.strip():
        return None, "empty_response"
    
    # Attempt 1: Full prompt
    full_prompt = build_prompt(instruction, context)
    if count_tokens(tokenizer, full_prompt) <= MAX_PROMPT_TOKENS:
        return {"prompt": full_prompt, "payload": response}, None
    
    # Attempt 2: Instruction only
    instr_only = instruction.strip()
    if instr_only and count_tokens(tokenizer, instr_only) <= MAX_PROMPT_TOKENS:
        return {"prompt": instr_only, "payload": response}, None
    
    return {"prompt": full_prompt, "payload": response}, "overlimit"

if __name__ == "__main__":
    sys.exit(entrypoint(iter_raw_records, process_record, description=__doc__))
