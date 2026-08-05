#!/usr/bin/env python3
"""
Quick token counter for a single input text.

Usage:
    uv run python3 datasets/count.py "TEXT" [options]

Options:
    -x, --exclude-special  Exclude special control tokens (e.g. [CLS], [SEP]...)
                           added to input text during tokenization.
    -p, --is-repeatable    Check whether "TEXT" is safe to use as a repeatable unit
                           for building synthetic strings, i.e. whether joining N
                           copies of "TEXT" (with whitespace in between) always yield
                           exactly N tokens.

NOTE:
  - If "TEXT" is confirmed (by -p) as a valid repeat unit: When actually build the
    synthetic string, join repetitions with whitespace(s) in between. WordPiece's
    pre-tokenizer only inserts implicit word boundaries at whitespace and punctuation
    characters. Concatenating an alphanumeric pattern directly produces ONE unbroken
    word that can exceed the tokenizer's max_input_chars_per_word limit (100 by
    default), silently collapsing the entire chunk into a single [UNK].
"""

import argparse
import sys

from _common import assert_in_venv, default_tokenizer_path, MAX_BOUND_TOKENS

# This script uses third-party library.
# MUST ensure it is run inside a virtual environment.
assert_in_venv()

from tokenizers import Tokenizer

# Specical control tokens: [CLS] and [SEP].
NUM_SPECIAL_TOKENS = 2


def load_default_tokenizer() -> Tokenizer:
    tokenizer_path = default_tokenizer_path()
    if not tokenizer_path.exists():
        print(
            f"Tokenizer not found: {tokenizer_path} -",
            "run 'bash fetch/tokenizer.sh' first.",
        )
        sys.exit(1)

    tok = Tokenizer.from_file(str(tokenizer_path))
    tok.no_padding()
    tok.no_truncation()

    _probe = tok.encode("")
    assert len(_probe) == NUM_SPECIAL_TOKENS, (
        f"Expected tokenizer to always add exactly 2 special tokens "
        f"([CLS]/[SEP]) per encode, got {len(_probe)} for an empty string. "
    )

    return tok


def check_is_repeatable(tokenizer: Tokenizer, pattern: str) -> tuple[bool, int]:
    for n in range(1, MAX_BOUND_TOKENS + 1):
        sample = " ".join([pattern] * n)
        count = len(tokenizer.encode(sample).ids) - NUM_SPECIAL_TOKENS
        if count != n:
            return False, n

    return True, -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "text", type=str,
        help="Input text to measure number of tokens"
    )
    parser.add_argument(
        "-x", "--exclude-special", action="store_true",
        help="Exclude special control tokens",
    )
    parser.add_argument(
        "-p", "--is-repeatable", action="store_true",
        help="Check if TEXT is safe as a whitespace-joined repeat unit",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    tokenizer = load_default_tokenizer()
    count = len(tokenizer.encode(args.text).ids)

    excl_count = count - NUM_SPECIAL_TOKENS
    if excl_count < 0:
        # Shouldn't happen with valid input.
        print(
            "Negative number of tokens after excluding special tokens!",
            file=sys.stderr,
        )
        return 1

    if args.is_repeatable:
        confirmed, first_break = check_is_repeatable(tokenizer, args.text)
        if confirmed:
            print(
                f"Confirmed as a repeatable unit (1:1 up to {MAX_BOUND_TOKENS} reps)."
            )
            print(
                "NOTE: Join repetitions with whitespace in between when building "
                "synthetic string - direct concatenation can silently collapse "
                "into a single [UNK]."
            )
        else:
            print(
                f"NOT a repeat unit. First break at N={first_break} reps."
            )

    if args.exclude_special:
        output = f"Number of tokens: {excl_count} - excluding special tokens."
    else:
        output = f"Number of tokens: {count} - including special tokens."

    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
