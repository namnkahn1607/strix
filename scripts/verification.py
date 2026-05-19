#!/usr/bin/env python3
"""
Verify processed JSONL dataset files
 
Checks performed on every line:
  1. Valid JSON
  2. Top-level value is an object (dict), not array/scalar
  3. Exactly the keys {"prompt", "payload"} - no missing, no extras
  4. Both values are non-empty strings (not null, not "", not whitespace-only)
 
Summary statistics reported at the end:
  - Total records
  - prompt  length: min / max / mean (in characters)
  - payload length: min / max / mean (in characters)
 
Exit codes:
  0: all checks passed
  1: one or more violations found (details printed to stderr)
 
Usage:
    python3 verification.py <file1.jsonl> [file2.jsonl ...]
"""

import sys
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROMPT_FIELD = "prompt"
PAYLOAD_FIELD = "payload"
EXPECTED_KEYS = {PROMPT_FIELD, PAYLOAD_FIELD}

def verify_file(path: Path) -> bool:
    total = violations = 0
    prompt_lengths: list[int] = []
    payload_lengths: list[int] = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            if raw.strip() == "":
                print(
                    f"{path}:{lineno}: ERROR: blank line",
                    file=sys.stderr,
                )
                
                violations += 1
                continue

            total += 1

            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(
                    f"{path}:{lineno}: ERROR: invalid JSON - {exc}",
                    file=sys.stderr,
                )

                violations += 1
                continue

            if not isinstance(record, dict):
                print(
                    f"{path}:{lineno}: ERROR: expected object, got {type(record).__name__}",
                    file=sys.stderr,
                )

                violations += 1
                continue

            actual_keys = set(record.keys())
            missing = EXPECTED_KEYS - actual_keys
            extra = actual_keys - EXPECTED_KEYS

            if missing:
                print(
                    f"{path}:{lineno}: ERROR: missing keys {sorted(missing)}",
                    file=sys.stderr,
                )

                violations += 1
                continue

            if extra:
                print(
                    f"{path}:{lineno}: ERROR: unexpected keys {sorted(extra)}",
                    file=sys.stderr,
                )

                violations += 1
                continue

            for key in (PROMPT_FIELD, PAYLOAD_FIELD):
                val = record[key]

                if not isinstance(val, str):
                    print(
                        f"{path}:{lineno}: ERROR: '{key}' is {type(val).__name__}, expected str",
                        file=sys.stderr,
                    )

                    violations += 1
                    break

                if not val.strip():
                    print(
                        f"{path}:{lineno}: ERROR: '{key}' is empty or whitespace-only",
                        file=sys.stderr,
                    )

                    violations += 1
                    break
            else:
                prompt_lengths.append(len(record[PROMPT_FIELD]))
                payload_lengths.append(len(record[PAYLOAD_FIELD]))

    passed = total - violations

    log.info("-" * 52)
    log.info("File              : %s", path)
    log.info("Total records     : %d", total)
    log.info("Passed            : %d", passed)
    log.info("Violations        : %d", violations)

    if prompt_lengths:
        log.info(
            "prompt  len (chars): min=%-5d  max=%-5d  mean=%.0f",
            min(prompt_lengths),
            max(prompt_lengths),
            sum(prompt_lengths) / len(prompt_lengths),
        )

        log.info(
            "payload len (chars): min=%-5d  max=%-5d  mean=%.0f",
            min(payload_lengths),
            max(payload_lengths),
            sum(payload_lengths) / len(payload_lengths),
        )

    return violations == 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="One or more processed JSONL files to verify")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    all_passed = True
    for path in args.files:
        if not path.exists():
            log.error("File not found: %s", path)
            all_passed = False
            continue

        result = verify_file(path)
        all_passed = all_passed and result

    if all_passed:
        log.info("═" * 52)
        log.info("All files passed verification.")
        sys.exit(0)
    else:
        log.error("═" * 52)
        log.error("Verification FAILED - see errors above.")
        sys.exit(1)
