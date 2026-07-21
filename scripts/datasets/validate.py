#!/usr/bin/env python3
"""
Validate processed JSONL dataset format.

Checks performed on every line:
  1. No blank lines
  2. Valid JSON
  3. Top-level value is an object (dictionary), not array/scalar
  4. Exactly the keys {"prompt", "payload"} - no missing, no extras
  5. Both values are non-empty strings (not null, not "", not whitespace-only)
  6. 'prompt' token count does not exceed MAX_PROMPT_TOKENS

A record failing any check is reported (file:line + reason) and counted as
a violation. Returns 0 on SUCCESS, 1 on FAILURE.

Summary statistics reported at the end:
  - Total records
  - 'prompt' length  : min / max / mean (in tokens)
  - 'prompt' length  : min / max / mean (in characters)
  - 'payload' length : min / max / mean (in characters)

NOTE:
  - This script uses the dictionary at strix/model/tokenizer.json. Make sure
    the file is available. If not, run: 'bash fetch/tokenizer.sh' first.

Usage:
    uv run python3 datasets/validate.py <file1.jsonl> [file2.jsonl ...]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from _common import MAX_PROMPT_TOKENS, assert_in_venv, default_tokenizer_path

# This script uses third-party library. 
# MUST ensure it is run inside a virtual environment.
assert_in_venv()

from tokenizers import Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROMPT_FIELD = "prompt"
PAYLOAD_FIELD = "payload"
EXPECTED_KEYS = {PROMPT_FIELD, PAYLOAD_FIELD}

def verify_file(path: Path, tokenizer: Tokenizer) -> bool:
    total = violations = over_limit = 0
    prompt_toks: list[int] = []
    prompt_bytes: list[int] = []
    payload_bytes: list[int] = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            if raw.strip() == "":
                print(f"{path}:{lineno} - Blank line", file=sys.stderr)
                violations += 1
                continue

            total += 1

            try:
                record = json.loads(raw)
            except json.JSONDecodeError as err:
                print(f"{path}:{lineno} - Invalid JSON: {err}", file=sys.stderr)
                violations += 1
                continue

            if not isinstance(record, dict):
                print(
                    f"{path}:{lineno} - Expected object, got {type(record).__name__}",
                    file=sys.stderr,
                )
                violations += 1
                continue

            actual_keys = set(record.keys())

            missing = EXPECTED_KEYS - actual_keys
            if missing:
                print(
                    f"{path}:{lineno} - Missing keys {sorted(missing)}",
                    file=sys.stderr,
                )
                violations += 1
                continue

            extra = actual_keys - EXPECTED_KEYS
            if extra:
                print(
                    f"{path}:{lineno} - Unexpected keys {sorted(extra)}",
                    file=sys.stderr,
                )
                violations += 1
                continue

            record_ok = True
            for key in (PROMPT_FIELD, PAYLOAD_FIELD):
                val = record[key]

                if not isinstance(val, str):
                    print(
                        f"{path}:{lineno} - '{key}' is {type(val).__name__}, expected str",
                        file=sys.stderr,
                    )
                    record_ok = False
                    break

                if not val.strip():
                    print(
                        f"{path}:{lineno} '{key}' is empty or whitespace-only",
                        file=sys.stderr,
                    )
                    record_ok = False
                    break
            
            if not record_ok:
                violations += 1
                continue

            tok_count = len(tokenizer.encode(record[PROMPT_FIELD]).ids)
            if tok_count > MAX_PROMPT_TOKENS:
                over_limit += 1

            prompt_toks.append(tok_count)
            prompt_bytes.append(len(record[PROMPT_FIELD]))
            payload_bytes.append(len(record[PAYLOAD_FIELD]))

    passed = total - violations

    log.info("-" * 52)
    log.info("File              : %s", path)
    log.info("Total records     : %d", total)
    log.info("Passed            : %d", passed)
    log.info("Violations        : %d", violations)
    if over_limit:
        log.info("Over token limit  : %d  (kept, informational only)", over_limit)

    if prompt_bytes:
        log.info(
            "prompt  len (tokens): min=%-5d  max=%-5d  mean=%.0f  (limit=%d)",
            min(prompt_toks),
            max(prompt_toks),
            sum(prompt_toks) / len(prompt_toks),
            MAX_PROMPT_TOKENS,
        )

        log.info(
            "prompt  len (chars) : min=%-5d  max=%-5d  mean=%.0f",
            min(prompt_bytes),
            max(prompt_bytes),
            sum(prompt_bytes) / len(prompt_bytes),
        )
 
        log.info(
            "payload len (chars) : min=%-5d  max=%-5d  mean=%.0f",
            min(payload_bytes),
            max(payload_bytes),
            sum(payload_bytes) / len(payload_bytes),
        )

    return violations == 0

def load_default_tokenizer() -> Tokenizer:
    tokenizer_path = default_tokenizer_path()
    if not tokenizer_path.exists():
        log.error(
            "Tokenizer not found: %s - run 'bash fetch/tokenizer.sh' first.",
            tokenizer_path,
        )
        sys.exit(1)

    log.info("Using tokenizer: %s", tokenizer_path)
    tok = Tokenizer.from_file(str(tokenizer_path))
    tok.no_padding()
    tok.no_truncation()
    return tok

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "files", nargs="+", type=Path,
        help="One or more processed JSONL files to verify",
    )

    return parser.parse_args()

def main() -> int:
    args = parse_args()
    tokenizer = load_default_tokenizer()

    all_passed = True
    for path in args.files:
        if not path.exists():
            log.error("File not found: %s", path)
            all_passed = False
            continue

        result = verify_file(path, tokenizer)
        all_passed = all_passed and result

    if all_passed:
        log.info("=" * 52)
        log.info("All files passed validation.")
        return 0
    else:
        log.error("=" * 52)
        log.error("Validation FAILED.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
