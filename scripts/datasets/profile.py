#!/usr/bin/env python3
"""
Empirical byte-length threshold profiler for BERT WordPiece tokenizer.

NOTE:
  - This profiler uses the dictionary at strix/model/tokenizer.json. Make sure
    the file is available. If not, run: 'bash fetch/tokenizer.sh' first.
  - It also expect valid JSONL datasets generated using datasets/build.py.

Usage:
    uv run python3 datasets/profiler.py <file1.jsonl> [file2.jsonl ...] [options]

Options:
    -p, --plot  Enable visual plotting. Default output: scripts/out/.
                If multiple datasets are included, default visual plot.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _common import assert_in_venv, default_tokenizer_path, MAX_BOUND_TOKENS

# This script uses third-party library.
# MUST ensure it is run inside a virtual environment.
assert_in_venv()

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from tokenizers import Tokenizer

# 256 minus [CLS] and [SEP] at each endpoint.
EFFECTIVE_TOKENS = MAX_BOUND_TOKENS - 2
PROMPT_FIELD = "prompt"
TRANSFORMER_NAME="all-MiniLM-L6-v2"

def load_default_tokenizer() -> Tokenizer:
    tokenizer_path = default_tokenizer_path()
    if not tokenizer_path.exists():
        print(
            f"Tokenizer not found: {tokenizer_path} -",
            "run 'bash fetch/tokenizer.sh' first.",
        )
        sys.exit(1)

    print(f"Using tokenizer: {tokenizer_path}")
    tok = Tokenizer.from_file(str(tokenizer_path))
    tok.no_padding()
    tok.no_truncation()

    _probe = tok.encode("")
    assert len(_probe) == 2, (
        f"Expected tokenizer to always add exactly 2 special tokens "
        f"([CLS]/[SEP]) per encode, got {len(_probe)} for an empty string. "
        f"EFFECTIVE_TOKENS calculation is no longer valid."
    )

    print(f"Effective tokens: {EFFECTIVE_TOKENS}")
    return tok

# ==============================================================================
# Level 1 - Theoretical vocab bounds
# ==============================================================================
def _is_special(token: str) -> bool:
    return token.startswith("[") and token.endswith("]")

def lvl1_vocab_bounds(tokenizer: Tokenizer) -> dict:
    vocab: dict[str, int] = tokenizer.get_vocab()

    token_bytes: list[tuple[str, int]] = []
    for token in vocab:
        if _is_special(token):
            continue

        surface = token.lstrip("#")
        byte_len = len(surface.encode("utf-8"))
        token_bytes.append((token, byte_len))

    token_bytes.sort(key=lambda x: x[1])

    shortest = token_bytes[:10]
    longest = token_bytes[-10:]
    min_bytes = token_bytes[0][1]
    max_bytes = token_bytes[-1][1]

    print("\n" + "=" * 60)
    print("LEVEL 1 - Theoretical Bounds")
    print("=" * 60)
    print(f"Vocab size (excl. special tokens): {len(token_bytes):,}")

    print(f"\nTop 10 shortest tokens (bytes):")
    for tok, bl in shortest:
        print(f"  {tok!r:30s}  {bl} byte(s)")

    print(f"\nTop 10 longest tokens (bytes):")
    for tok, bl in reversed(longest):
        print(f"  {tok!r:30s}  {bl} byte(s)")

    print(f"\nAbsolute lower bound: {EFFECTIVE_TOKENS} x {min_bytes} = "
          f"{EFFECTIVE_TOKENS * min_bytes} bytes")
    print(f"Absolute upper bound: {EFFECTIVE_TOKENS} x {max_bytes} = "
          f"{EFFECTIVE_TOKENS * max_bytes} bytes")

    return {"min_bytes": min_bytes, "max_bytes": max_bytes,
            "lower_bound": EFFECTIVE_TOKENS * min_bytes,
            "upper_bound": EFFECTIVE_TOKENS * max_bytes}

# ==============================================================================
# Level 2 - Empirical distribution
# ==============================================================================
def load_prompts(paths: list[Path]) -> list[str]:
    prompts = []
    for path in paths:
        with open(path, encoding="utf-8") as fin:
            for lineno, line in enumerate(fin, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        f"{path.name}:{lineno} - Skipping malformed JSON: {exc}",
                        file=sys.stderr,
                    )
                    continue

                if PROMPT_FIELD not in obj:
                    print(
                        f"{path.name}:{lineno} - Missing '{PROMPT_FIELD}' field, skipping",
                        file=sys.stderr,
                    )
                    continue

                prompts.append(obj[PROMPT_FIELD])

    return prompts

def lvl2_empirical(tokenizer: Tokenizer, prompts: list[str]) -> dict:
    byte_lengths = []
    token_counts = []

    print(f"\n{'=' * 60}")
    print("LEVEL 2 - Empirical Distribution")
    print(f"{'=' * 60}")
    print(f"Tokenizing {len(prompts):,} prompts …", end="", flush=True)

    for prompt in prompts:
        enc = tokenizer.encode(prompt)
        # Tokenizer encoding adds [CLS] and [SEP]. Subtract them.
        payload_toks = len(enc) - 2
        token_counts.append(payload_toks)

        byte_len = len(prompt.encode("utf-8"))
        byte_lengths.append(byte_len)

    print(" done.")

    byte_lengths = np.array(byte_lengths)
    token_counts = np.array(token_counts)

    at_limit_mask = token_counts == EFFECTIVE_TOKENS
    overlimit_mask = token_counts > EFFECTIVE_TOKENS
    valid_mask = token_counts <= EFFECTIVE_TOKENS

    print(f"\nTotal prompts          : {len(prompts):,}")
    print(f"Over-limit (>{EFFECTIVE_TOKENS} tok)  : {overlimit_mask.sum():,} "
          f"({100 * overlimit_mask.mean():.2f}%)")
    print(f"At-limit  (=={EFFECTIVE_TOKENS} tok)  : {at_limit_mask.sum():,} "
          f"({100 * at_limit_mask.mean():.2f}%)")

    if valid_mask.sum() > 0:
        valid_bytes = byte_lengths[valid_mask]
        print(f"\nByte distribution for VALID prompts (token_count <= {EFFECTIVE_TOKENS}):")
        print(f"  min    : {valid_bytes.min()}")
        print(f"  median : {np.median(valid_bytes):.0f}")
        print(f"  mean   : {valid_bytes.mean():.1f}")
        print(f"  p95    : {np.percentile(valid_bytes, 95):.0f}")
        print(f"  p99    : {np.percentile(valid_bytes, 99):.0f}")
        print(f"  p99.9  : {np.percentile(valid_bytes, 99.9):.0f}")
        print(f"  max    : {valid_bytes.max()}  <- this is the threshold FP=0 (see Level 3)")

    return {
        "byte_lengths": byte_lengths,
        "token_counts": token_counts,
        "at_limit_mask": at_limit_mask,
        "overlimit_mask": overlimit_mask,
        "valid_mask": valid_mask,
    }

# ==============================================================================
# Level 3 - Byte length threshold analysis
# ==============================================================================
ILLUSTRATIVE_THRESHOLDS = [
    512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096
]

def lvl3_threshold(data: dict) -> int:
    byte_lengths = data["byte_lengths"]
    token_counts = data["token_counts"]
    valid_mask = data["valid_mask"]

    print(f"\n{'=' * 60}")
    print("LEVEL 3 - Threshold Analysis")
    print(f"{'=' * 60}")

    if valid_mask.sum() == 0:
        print(
            f"No valid prompts (with token_count <= {EFFECTIVE_TOKENS}) in the dataset. "
            "Cannot calculate threshold."
        )
        sys.exit(1)

    threshold = int(byte_lengths[valid_mask].max())
    fp_mask = (token_counts <= EFFECTIVE_TOKENS) & (byte_lengths > threshold)
    fn_mask = (token_counts > EFFECTIVE_TOKENS) & (byte_lengths <= threshold)
    assert fp_mask.sum() == 0, "BUG: threshold = max(valid_bytes) MUST always give FP = 0."

    print(f"\nThreshold = max(byte_length | token_count <= {EFFECTIVE_TOKENS}) "
          f"= {threshold:,} bytes")
    print(f"  False Positive: {fp_mask.sum():,}")
    print(f"  False Negative: {fn_mask.sum():,}  "
          f"({100 * fn_mask.sum() / len(byte_lengths):.2f}%, acceptable)")

    print(f"\nQuantifies FP/FN at several other points:")
    print(f"{'Threshold':>10}  {'FP count':>10}  {'FN count':>10}")
    print("-" * 36)
    for thresh in ILLUSTRATIVE_THRESHOLDS:
        fp = ((token_counts <= EFFECTIVE_TOKENS) & (byte_lengths > thresh)).sum()
        fn = ((token_counts > EFFECTIVE_TOKENS) & (byte_lengths <= thresh)).sum()
        marker = "  <- recommended" if thresh == threshold else ""
        print(f"{thresh:>10,}  {fp:>10,}  {fn:>10,}{marker}")

    return threshold

# ==============================================================================
# Plotting
# ==============================================================================
def plot_results(data: dict, recommended: int, out_path: Path) -> None:
    byte_lengths    = data["byte_lengths"]
    token_counts    = data["token_counts"]
    valid_mask      = data["valid_mask"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Token Profiler (all-MiniLM-L6-v2)", fontsize=13, fontweight="bold")

    # Plot 1: Byte length distribution for valid prompts.
    ax = axes[0]
    if valid_mask.sum() > 0:
        ax.hist(byte_lengths[valid_mask], bins=40, color="#4C72B0", edgecolor="white")
        ax.axvline(recommended, color="red", linestyle="--",
                   label=f"Threshold = {recommended}")
        ax.set_title(f"Byte length | token_count <= {EFFECTIVE_TOKENS}")
        ax.set_xlabel("Bytes")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, f"No valid prompts in dataset",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Byte length | token_count <= {EFFECTIVE_TOKENS}")

    # Plot 2: Scatter (byte_length vs token_count), sampled.
    ax = axes[1]
    sample_size = min(5000, len(byte_lengths))
    idx = np.random.choice(len(byte_lengths), sample_size, replace=False)
    colors = np.where(token_counts[idx] > EFFECTIVE_TOKENS, "#DD4444", "#4C72B0")
    ax.scatter(byte_lengths[idx], token_counts[idx],
               c=colors, alpha=0.3, s=6)
    ax.axvline(recommended, color="red", linestyle="--",
               label=f"Threshold = {recommended}", linewidth=1.2)
    ax.axhline(EFFECTIVE_TOKENS, color="orange", linestyle="--",
               label=f"Token limit = {EFFECTIVE_TOKENS}", linewidth=1.0)
    ax.set_title(f"Scatter: bytes vs tokens (n={sample_size:,})")
    ax.set_xlabel("Byte length")
    ax.set_ylabel("Token count")
    ax.legend(fontsize=8)

    # Plot 3: FP / FN rate curve across thresholds.
    ax = axes[2]
    fp_rates, fn_rates = [], []
    lo = max(0, int(byte_lengths.min()) - 50)
    hi = int(byte_lengths.max()) + 50
    step = max(1, (hi - lo) // 60)
    thresholds = list(range(lo, hi, step))
    total = len(byte_lengths)
    EFFECTIVE_LIMIT = EFFECTIVE_TOKENS
    for t in thresholds:
        fp = ((token_counts <= EFFECTIVE_LIMIT) & (byte_lengths > t)).sum() / total
        fn = ((token_counts > EFFECTIVE_LIMIT) & (byte_lengths <= t)).sum() / total
        fp_rates.append(fp * 100)
        fn_rates.append(fn * 100)

    ax.plot(thresholds, fp_rates, label="False Positive %", color="#DD4444")
    ax.plot(thresholds, fn_rates, label="False Negative %", color="#F5A623")
    ax.axvline(recommended, color="red", linestyle="--",
               label=f"Recommended = {recommended}", linewidth=1.2)
    ax.set_title("FP / FN rate vs threshold")
    ax.set_xlabel("Byte threshold")
    ax.set_ylabel("Rate (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f%%"))
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved -> {out_path}")

# ==============================================================================
# Main
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "datasets", nargs="+", type=Path,
        help="One or more paths to dataset files.")
    parser.add_argument(
        "-p", "--plot", action="store_true",
        help="Distribution plot output path (default: strix/scripts/out/)",
    )

    return parser.parse_args()

def main() -> int:
    args = parse_args()
    here = Path(__file__).resolve().parent

    dataset_paths = []
    for path in args.datasets:
        path = path.resolve()
        if not path.exists():
            print(f"File not found: {path}")
            return 1

        # Validate the dataset format.
        result = subprocess.run(
            [sys.executable, str(here / "validate.py"), str(path)],
        )
        if result.returncode != 0:
            print(
                f"Invalid dataset format: {path}",
                "Are you sure it is generated using datasets/build.py?"
            )
            return 1

        dataset_paths.append(path)

    # All datasets are valid. Proceed to profiling.
    tokenizer = load_default_tokenizer()
    lvl1_vocab_bounds(tokenizer)

    prompts = load_prompts(dataset_paths)
    if not prompts:
        # Should not happen. Success validation YET no valid prompts?
        print(
            "No valid prompts found in provided datasets.\n",
            "Recommend manual proofreading the datasets first, then check for",
            "programming bugs in this script (e.g. load_prompt() function).",
            file=sys.stderr,
        )
        return 1

    data = lvl2_empirical(tokenizer, prompts)
    recommended = lvl3_threshold(data)

    if args.plot:
        plot_name = "token_profiling.png"
        if len(dataset_paths) == 1:
            dataset_name = dataset_paths[0].stem
            plot_name = f"{dataset_name}.png"

        plot_out = here.parent / "out" / plot_name
        plot_out.parent.mkdir(parents=True, exist_ok=True)
        plot_results(data, recommended, plot_out)

    print(f"\n{'=' * 60}")
    print(f"FINAL RECOMMENDATION")
    print(f"{'=' * 60}")
    print(f"HEURISTIC_BYTE_THRESHOLD = {recommended}")
    print(f"Rationale: FP rate = 0 on profiling dataset (not actual traffic).")
    return 0

if __name__ == "__main__":
    sys.exit(main())
