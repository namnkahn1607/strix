#!/usr/bin/env python3
"""
Empirical byte-length threshold profiler for BERT WordPiece tokenizer

Usage (run from project root):
    python3 wordpiece.py <file1.jsonl> [file2.jsonl ...]

NOTE: 
  - All .jsonl files as arguments are expected to be located in data/,
  so you don't need to specify 'data/file.jsonl' (or else the script
  will throw an error).
  - This profiling is based on all-MiniLM-L6-v2's dictionary ONLY,
  and also expects preprocessed JSONL.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tokenizers import Tokenizer

EFFECTIVE_TOKENS = 254 # 256 minus [CLS] and [SEP] at each endpoint
PROMPT_FIELD = "prompt"

### Path resolution ###

def project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def resolve_dataset_path(arg: str, root: Path) -> Path:
    p = root / "data" / arg
    if p.exists():
        return p
    
    raise FileNotFoundError(f"Dataset not found: {p}")

def resolve_tokenizer_path(root: Path) -> Path:
    p = root / "model" / "tokenizer.json"
    if p.exists():
        return p
    
    raise FileNotFoundError(f"Tokenizer not found: {p}")

### Level 1 - Theoretical vocab bounds ###

def is_special(token: str) -> bool:
    return token.startswith("[") and token.endswith("]")

def lvl1_vocab_bounds(tokenizer: Tokenizer) -> dict:
    vocab: dict[str, int] = tokenizer.get_vocab()

    token_bytes: list[tuple[str, int]] = []
    for token in vocab:
        if is_special(token):
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
    print("LEVEL 1 - Theoretical Bounds (from vocab)")
    print("=" * 60)
    print(f"Vocab size (excl. special tokens): {len(token_bytes):,}")

    print(f"\nTop 10 shortest tokens (bytes):")
    for tok, bl in shortest:
        print(f"  {tok!r:30s}  {bl} byte(s)")
    
    print(f"\nTop 10 longest tokens (bytes):")
    for tok, bl in reversed(longest):
        print(f"  {tok!r:30s}  {bl} byte(s)")
    
    print(f"\nAbsolute lower bound: {EFFECTIVE_TOKENS} × {min_bytes} = "
          f"{EFFECTIVE_TOKENS * min_bytes} bytes")
    print(f"Absolute upper bound: {EFFECTIVE_TOKENS} × {max_bytes} = "
          f"{EFFECTIVE_TOKENS * max_bytes} bytes")
 
    return {"min_bytes": min_bytes, "max_bytes": max_bytes,
            "lower_bound": EFFECTIVE_TOKENS * min_bytes,
            "upper_bound": EFFECTIVE_TOKENS * max_bytes}

### Level 2 - Empirical distribution ###

def load_prompts(paths: list[Path]) -> list[str]:
    prompts = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  [WARN] {path.name}:{lineno} - skipping malformed JSON: {e}",
                          file=sys.stderr)
                    continue

                if PROMPT_FIELD not in obj:
                    print(f"  [WARN] {path.name}:{lineno} - missing '{PROMPT_FIELD}' field, skipping",
                          file=sys.stderr)
                    continue

                prompts.append(obj[PROMPT_FIELD])

    return prompts

def lvl2_empirical(tokenizer: Tokenizer, prompts: list[str]) -> dict:
    EFFECTIVE_LIMIT = 254
    byte_lengths = []
    token_counts = []

    tokenizer.no_truncation()
    tokenizer.no_padding()

    print(f"\n{'=' * 60}")
    print("LEVEL 2 - Empirical Distribution")
    print(f"{'=' * 60}")
    print(f"Tokenizing {len(prompts):,} prompts …", end="", flush=True)

    for prompt in prompts:
        enc = tokenizer.encode(prompt)
        # encode() adds [CLS] and [SEP]. Subtract them for payload count
        payload_tokens = len(enc.ids) - 2
        byte_len = len(prompt.encode("utf-8"))
        byte_lengths.append(byte_len)
        token_counts.append(payload_tokens)

    print(" done.")

    byte_lengths = np.array(byte_lengths)
    token_counts = np.array(token_counts)
 
    at_limit_mask = token_counts == EFFECTIVE_LIMIT
    over_limit_mask = token_counts > EFFECTIVE_LIMIT
 
    print(f"\nTotal prompts          : {len(prompts):,}")
    print(f"Over-limit (>254 tok)  : {over_limit_mask.sum():,} "
          f"({100 * over_limit_mask.mean():.2f}%)")
    print(f"At-limit  (==254 tok)  : {at_limit_mask.sum():,} "
          f"({100 * at_limit_mask.mean():.2f}%)")
 
    if at_limit_mask.sum() > 0:
        at_limit_bytes = byte_lengths[at_limit_mask]
        print(f"\nByte distribution for prompts with exactly 254 tokens:")
        print(f"  min    : {at_limit_bytes.min()}")
        print(f"  median : {np.median(at_limit_bytes):.0f}")
        print(f"  mean   : {at_limit_bytes.mean():.1f}")
        print(f"  p95    : {np.percentile(at_limit_bytes, 95):.0f}")
        print(f"  p99    : {np.percentile(at_limit_bytes, 99):.0f}")
        print(f"  p99.9  : {np.percentile(at_limit_bytes, 99.9):.0f}")
        print(f"  max    : {at_limit_bytes.max()}")
 
    return {
        "byte_lengths": byte_lengths,
        "token_counts": token_counts,
        "at_limit_mask": at_limit_mask,
        "over_limit_mask": over_limit_mask,
    }

### Level 3 - Threshold analysis ###

CANDIDATE_THRESHOLDS = [512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072]

def level3_threshold(data: dict) -> int:
    byte_lengths   = data["byte_lengths"]
    token_counts   = data["token_counts"]
    EFFECTIVE_LIMIT = 254
 
    print(f"\n{'=' * 60}")
    print("LEVEL 3 - Threshold Analysis")
    print(f"{'=' * 60}")
    print(f"\n{'Threshold':>10}  {'FP rate':>10}  {'FN rate':>10}  "
          f"{'FP count':>10}  {'FN count':>10}")
    print("-" * 55)
 
    best_threshold = None
    for thresh in CANDIDATE_THRESHOLDS:
        # False Positive: token_count <= 254 BUT byte_length > threshold
        # -> valid prompt incorrectly forwarded to LLM
        fp_mask = (token_counts <= EFFECTIVE_LIMIT) & (byte_lengths > thresh)

        # False Negative: token_count > 254 BUT byte_length <= threshold
        # -> over-limit prompt NOT caught by heuristic
        fn_mask = (token_counts > EFFECTIVE_LIMIT) & (byte_lengths <= thresh)
 
        fp_rate = fp_mask.sum() / len(byte_lengths)
        fn_rate = fn_mask.sum() / len(byte_lengths)
 
        print(f"{thresh:>10,}  {fp_rate:>10.4%}  {fn_rate:>10.4%}  "
              f"{fp_mask.sum():>10,}  {fn_mask.sum():>10,}")
 
        # Recommend first threshold where FP rate < 0.01%
        if best_threshold is None and fp_rate < 0.0001:
            best_threshold = thresh
 
    if best_threshold is None:
        best_threshold = CANDIDATE_THRESHOLDS[-1]
 
    print(f"\n-> Recommended threshold: {best_threshold} bytes")
    print(f"  (first candidate with FP rate < 0.01%)")
    return best_threshold

### Ploting ###

def plot_results(data: dict, recommended: int, out_path: Path) -> None:
    byte_lengths    = data["byte_lengths"]
    token_counts    = data["token_counts"]
    at_limit_mask   = data["at_limit_mask"]
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Token Profiler (all-MiniLM-L6-v2)", fontsize=13, fontweight="bold")
 
    # Plot 1: Byte length distribution for at-limit prompts
    ax = axes[0]
    if at_limit_mask.sum() > 0:
        ax.hist(byte_lengths[at_limit_mask], bins=40, color="#4C72B0", edgecolor="white")
        ax.axvline(recommended, color="red", linestyle="--",
                   label=f"Threshold = {recommended}")
        ax.set_title("Byte length | token_count == 254")
        ax.set_xlabel("Bytes")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No prompts at exactly\n254 tokens in dataset",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Byte length | token_count == 254")
 
    # Plot 2: Scatter (byte_length vs token_count), sampled
    ax = axes[1]
    sample_size = min(5000, len(byte_lengths))
    idx = np.random.choice(len(byte_lengths), sample_size, replace=False)
    colors = np.where(token_counts[idx] > 254, "#DD4444", "#4C72B0")
    ax.scatter(byte_lengths[idx], token_counts[idx],
               c=colors, alpha=0.3, s=6)
    ax.axvline(recommended, color="red", linestyle="--",
               label=f"Threshold = {recommended}", linewidth=1.2)
    ax.axhline(254, color="orange", linestyle="--",
               label="Token limit = 254", linewidth=1.0)
    ax.set_title(f"Scatter: bytes vs tokens (n={sample_size:,})")
    ax.set_xlabel("Byte length")
    ax.set_ylabel("Token count")
    ax.legend(fontsize=8)
 
    # Plot 3: FP / FN rate curve across thresholds
    ax = axes[2]
    fp_rates, fn_rates = [], []
    thresholds = list(range(300, 3200, 50))
    total = len(byte_lengths)
    EFFECTIVE_LIMIT = 254
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

### Entry point ###

def main() -> None:
    parser = argparse.ArgumentParser(description="Empirical byte-threshold profiler for BERT tokenizer.")
    parser.add_argument(
        "datasets",
        nargs="+",
        metavar="DATASET",
        help="One or more JSONL dataset files (name, relative, or absolute path).",
    )
    parser.add_argument(
        "--plot",
        metavar="OUTPUT.png",
        default=None,
        help="Save distribution plots to this file (default: auto-named beside script).",
    )
    
    args = parser.parse_args()
    root = project_root()
    print(f"Project root : {root}")

    tok_path = resolve_tokenizer_path(root)
    print(f"Tokenizer    : {tok_path}")
    tokenizer = Tokenizer.from_file(str(tok_path))

    dataset_paths = []
    for arg in args.datasets:
        try:
            p = resolve_dataset_path(arg, root)
            dataset_paths.append(p)
            print(f"Dataset      : {p}")
        except FileNotFoundError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            sys.exit(1)

    lvl1_vocab_bounds(tokenizer)
 
    prompts = load_prompts(dataset_paths)
    if not prompts:
        print("[ERROR] No valid prompts found in provided datasets.", file=sys.stderr)
        sys.exit(1)
 
    data = lvl2_empirical(tokenizer, prompts)
    recommended = level3_threshold(data)
 
    plot_out = Path(args.plot) if args.plot else \
        Path(__file__).resolve().parent / "token_profiling.png"
    plot_results(data, recommended, plot_out)
 
    print(f"\n{'=' * 60}")
    print(f"FINAL RECOMMENDATION")
    print(f"{'=' * 60}")
    print(f"  HEURISTIC_BYTE_THRESHOLD = {recommended}")
    print(f"  Set this in your HTTP Gateway config.")
    print(f"  Rationale: FP rate < 0.01% on your actual prompt distribution.")

if __name__ == "__main__":
    main()
