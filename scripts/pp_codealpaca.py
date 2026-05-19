#!/usr/bin/env python3
"""
Preprocess (reformat) code_alpaca_20k.json
 
Input schema:  JSON array of { output, instruction, input }
Output schema: JSONL of { prompt, payload }
 
Prompt construction & filtering logic:
  - prompt  = instruction + "\n" + input  (if input is non-empty)
            = instruction                 (if input is empty)
  - payload = output
 
  NO fallback chain: `input` is a primary operand (code snippet, function signature, constraint), not optional context.
  Therefore: if len(tokens) > 256 -> discard.
 
  The only 'soft' case is when input is empty to begin with - then instruction alone is the full prompt, not a degraded fallback.
 
Tokenizer: loaded from a local tokenizer.json (sentence-transformers/all-MiniLM-L6-v2).
 
Usage:
    python3 pp_codealpaca.py <input.json> <output.jsonl> <tokenizer.json>
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
    log.info("Loading tokenizer from: %s", tokenizer_path)
    tok = Tokenizer.from_file(str(tokenizer_path))
    tok.no_padding()
    tok.no_truncation()
    return tok
 
def count_tokens(tokenizer: Tokenizer, text: str) -> int:
    encoding = tokenizer.encode(text)
    return len(encoding.ids)

def build_prompt(instruction: str, input_: str) -> str:
    instruction = instruction.strip()
    input_ = input_.strip()
    
    if input_:
        return f"{instruction}\n{input_}"
    
    return instruction

def process_record(record: dict, tokenizer: Tokenizer) -> dict | None:
    instruction: str = record.get("instruction", "")
    input_: str = record.get("input", "")
    output_: str = record.get("output", "")

    if not output_.strip():
        return None
    
    prompt = build_prompt(instruction, input_)

    if not prompt:
        return None
    
    if count_tokens(tokenizer, prompt) > MAX_TOKENS:
        return None
    
    return {"prompt": prompt, "payload": output_}

def preprocess(input_path: Path, output_path: Path, tokenizer_path: Path):
    tokenizer = load_tokenizer(tokenizer_path)
    
    log.info("Loading JSON array from: %s", input_path)
    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        log.error("Expected a JSON array at top level, got %s", type(records).__name__)
        sys.exit(1)

    total = len(records)
    kept = dropped_overflow = dropped_empty = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for record in records:
            result = process_record(record, tokenizer)

            if result is None:
                if not record.get("output", "").strip():
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input",     type=Path, help="Path to raw CodeAlpaca JSON file")
    parser.add_argument("output",    type=Path, help="Path for processed output JSONL file")
    parser.add_argument("tokenizer", type=Path, help="Path to tokenizer.json (all-MiniLM-L6-v2)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if not args.input.exists():
        log.error("Input file not found: %s", args.input)
        sys.exit(1)
 
    if not args.tokenizer.exists():
        log.error("tokenizer.json not found: %s", args.tokenizer)
        sys.exit(1)

    preprocess(args.input, args.output, args.tokenizer)
