"""
Shared driver for per-dataset preprocessor script.

Each dataset's preprocessor supplies exactly 2 things:
    iter_raw_records(input_path) -> Iterator[dict]
        Parses the RAW format. This is where JSON-array vs JSONL vs anything
        else a future dataset uses actually diverges - kept out of the shared
        driver on purpose, since it's the one part that's genuinely per-format,
        not per-dataset-logic.
 
    process_record(record, tokenizer) -> tuple[dict | None, str | None]
        The real per-record decision. Return (kept_dict, None) to keep a
        record, or (None, "some_reason") to drop it with a labeled reason -
        the driver counts and reports whatever reason strings come back, so
        stats are always accurate to what actually happened, never inferred
        after the fact from the output shape.
 
        build_prompt(...) is NOT a driver concern - its signature differs per
        dataset (different raw field names), so it lives entirely inside each
        dataset's own process_record.
 
...then calls: entrypoint(iter_raw_records, process_record, description=__doc__)
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Iterator

from _common import assert_in_venv, default_tokenizer_path, repo_root

# MUST run before any third-party import.
assert_in_venv()

from tokenizers import Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ProcessFn = Callable[[dict, Tokenizer], "tuple[dict | None, str | None]"]
IterFn = Callable[[Path], Iterator[dict]]

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

def count_tokens(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text).ids)

def run(input_path: Path, output_path: Path, 
        iter_raw_records: IterFn, process_record: ProcessFn):
    tokenizer = load_default_tokenizer()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total = kept = 0
    drop_reasons: Counter[str] = Counter()
    flagged: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8") as fout:
        for record in iter_raw_records(input_path):
            total += 1

            result, tag = process_record(record, tokenizer)
            if result is None:
                drop_reasons[tag or "unspecified"] += 1
                continue
            if tag:
                flagged[tag] += 1

            fout.write(json.dumps(result, ensure_ascii=True) + "\n")
            kept += 1

    log.info("-" * 50)
    log.info("Total records : %d", total)
    log.info("Kept          : %d  (%.1f%%)", kept, 100 * kept / max(total, 1))
    for reason, count in drop_reasons.most_common():
        log.info("Dropped (%-12s): %d", reason, count)
    for tag, count in flagged.most_common():
        log.info("Flagged (%-12s): %d  (kept, not excluded)", tag, count)
    log.info("Output        : %s", output_path.resolve())

def parse_args(description: str | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input", type=Path,
        help="Path to the raw dataset file"
    )

    return parser.parse_args()
    
def entrypoint(iter_raw_records: IterFn, process_record: ProcessFn, 
               description: str | None = None) -> int:
    args = parse_args(description)

    input_path = args.input.resolve()
    if not input_path.exists():
        log.error("Raw input dataset not found: %s", args.input)
        return 1
    if not input_path.is_file():
        log.error("Input path must be a file, not a directory: %s", args.input)
        return 1

    dataset_name = input_path.stem
    output_path = repo_root() / "data" / f"{dataset_name}.jsonl"
    run(input_path, output_path, iter_raw_records, process_record)
    return 0 
