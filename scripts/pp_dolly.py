#!/usr/bin/env python3
"""
Preprocess (reformat) databricks-dolly-15k.jsonl
 
Input schema:  { instruction, context, response, category }
Output schema: { prompt, payload }
 
Prompt construction & filtering logic:
  1. Try prompt = instruction + " " + context (stripped)
  2. If len(tokens) <= 256 -> emit
  3. Else try prompt = instruction only
  4. If len(tokens) <= 256 -> emit (payload = response, context dropped)
  5. Else discard the record entirely

Tokenizer: loaded from a local tokenizer.json (sentence-transformers/all-MiniLM-L6-v2).
 
Usage:
    python3 pp_dolly.py <input.jsonl> <output.jsonl> <tokenizer.json>
"""

import sys
import json
import argparse
import logging
from pathlib import Path

from tokenizers import Tokenizer

MAX_TOKENS = 256

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    log.info("Loading Tokenizer from %s", tokenizer_path)
    tok = Tokenizer.from_file(str(tokenizer_path))
    tok.no_padding()
    tok.no_truncation()

    return tok

def count_tokens(tokenizer: Tokenizer, text: str) -> int:
    encoding = tokenizer.encode(text)
    return len(encoding.ids)

def build_prompt(instruction: str, context: str) -> str:
    instruction = instruction.strip()
    context = context.strip()

    if context:
        return f"{instruction} {context}"
    
    return instruction

def process_record(record: dict, tokenizer: Tokenizer) -> dict | None:
    instruction: str = record.get("instruction", "")
    context: str = record.get("context", "")
    response: str = record.get("response", "")

    if not response.strip():
        return None
    
    # Attempt 1: Full prompt
    full_prompt = build_prompt(instruction, context)
    if count_tokens(tokenizer, full_prompt) <= MAX_TOKENS:
        return {"prompt": full_prompt, "payload": response}
    
    # Attempt 2: Instruction only
    instr_only = instruction.strip()
    if instr_only and count_tokens(tokenizer, instr_only) <= MAX_TOKENS:
        return {"prompt": instr_only, "payload": response}
    
    return None

def preprocess(input_path: Path, output_path: Path, tokenizer_path: Path):
    tokenizer = load_tokenizer(tokenizer_path)
    total = kept = dropped_overflow = dropped_empty = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        input_path.open("r", encoding="utf-8") as fin,
        output_path.open("w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Malformed JSON at record %d - skipping. (%s)", total, exc)
                dropped_empty += 1
                continue

            result = process_record(record, tokenizer)
            if result is None:
                if not record.get("response", "").strip():
                    dropped_empty += 1
                else:
                    dropped_overflow += 1

                continue

            fout.write(json.dumps(result, ensure_ascii=True) + "\n")
            kept += 1

    log.info("-" * 50)
    log.info("Total records     : %d", total)
    log.info("Kept              : %d  (%.1f%%)", kept, 100 * kept / max(total, 1))
    log.info("Dropped (overflow): %d", dropped_overflow)
    log.info("Dropped (empty)   : %d", dropped_empty)
    log.info("Output            : %s", output_path.resolve())

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Path to raw JSONL file")
    parser.add_argument("output",    type=Path, help="Path for processed output JSONL file")
    parser.add_argument("tokenizer", type=Path, help="Path to tokenizer.json (all-MiniLM-L6-v2)")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
 
    if not args.input.exists():
        log.error("Input file not found: %s", args.input)
        sys.exit(1)
 
    if not args.tokenizer.exists():
        log.error("'tokenizer.json' not found: %s", args.tokenizer)
        sys.exit(1)
 
    preprocess(args.input, args.output, args.tokenizer)
