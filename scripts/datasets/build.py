#!/usr/bin/env python3
"""
Canonical pipeline for one dataset: fetch -> preprocess -> validate.
Destination: strix/data/.

NOTE:
  - This script uses the dictionary at strix/model/tokenizer.json. Make sure
    the file is available. If not, run: 'bash fetch/tokenizer.sh' first.

Usage:
    uv run python3 datasets/build.py <dataset_dir> [options]

Options:
    -k, --keepraw  Never remove the raw dataset after a successful build.
    -p, --profile  Enable post-build profiling. Plot is written to scripts/out/.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from _common import assert_in_venv, load_manifest, raw_dataset_path, processed_dataset_path

# This script uses third-party library.
# MUST ensure it is run inside a virtual environment.
assert_in_venv()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "dataset_dir", type=Path,
        help="Path to dataset directory containing manifest and preprocessor",
    )
    parser.add_argument(
        "-k", "--keepraw", action="store_true",
        help="Don't delete the raw dataset after a successful build",
    )
    parser.add_argument(
        "-p", "--profile", action="store_true",
        help="Perform post-build empirical byte-length threshold profiling"
    )

    return parser.parse_args()

def main() -> int:
    args = parse_args()

    here = Path(__file__).resolve().parent

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.is_dir():
        print(
            f"Expected a dataset directory as argument. Got {dataset_dir}.",
            file=sys.stderr,
        )
        return 1

    info_toml = dataset_dir / "info.toml"
    preprocessor = dataset_dir / "preprocess.py"
    for required in (info_toml, preprocessor):
        if not required.exists():
            print(f"{required} not found.", file=sys.stderr)
            return 1

    manifest = load_manifest(info_toml)
    raw_path = raw_dataset_path(manifest)
    processed_path = processed_dataset_path(manifest)

    # 1. fetch
    print(f"Step 1. Fetch -> {raw_path}")
    subprocess.run(
        [sys.executable, str(here / "fetch.py"), str(info_toml)], check=True,
    )

    # 2. preprocess
    print(f"Step 2. Preprocess -> {processed_path}")
    subprocess.run(
        [sys.executable, str(preprocessor), str(raw_path)], check=True
    )

    # 3. validate
    print(f"Step 3. Validate {processed_path}")
    result = subprocess.run(
        [sys.executable, str(here / "validate.py"), str(processed_path)],
    )
    if result.returncode != 0:
        print(
            "Validation failed. "
            f"Make sure preprocess.py in {dataset_dir} is doing right."
        )
        return result.returncode

    if not args.keepraw:
        raw_path.unlink(missing_ok=True)

    if args.profile:
        print(f"Step 4. Profiling {processed_path}")
        subprocess.run(
            [sys.executable, str(here / "profile.py"), str(processed_path),
             "--plot"], check=True
        )

    print(f"OK: {processed_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
