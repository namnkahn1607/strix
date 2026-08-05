#!/usr/bin/env python3
"""
Fetch raw dataset from HuggingFace Hub, pinned to an exact commit.
Destination: ~/.cache/strix/datasets/

'info.toml' format:
    repo_id    = "<dataset name>"              # e.g. databricks/databricks-dolly-15k)
    revision   = "<full 40-char commit hash>"  # NOT a branch/tag
    file_path  = "<path to dataset>"           # Path to the file WITHIN the repository
    local_name = "<local dataset name>"        # Local unified dataset naming

Usage:
    uv run python3 datasets/fetch.py <info.toml>
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

from _common import assert_in_venv, load_manifest, raw_dataset_path

# This script uses third-party library.
# MUST ensure it is run inside a virtual environment.
assert_in_venv()

from huggingface_hub import hf_hub_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "info_toml", type=Path,
        help="Path to the TOML dataset manifest."
    )

    return parser.parse_args()

def main() -> int:
    args = parse_args()

    info_toml = args.info_toml.resolve()
    if not info_toml.exists():
        log.error("File not found: %s", info_toml)
        return 1
    if not info_toml.is_file() or info_toml.suffix.lower() != ".toml":
        log.error("Not a valid input file: %s. Must be a TOML.", info_toml)
        return 1

    manifest = load_manifest(info_toml)

    output_path = raw_dataset_path(manifest)
    if output_path.exists():
        log.info("Already cached: %s. Skip fetch.", output_path)
        return 0

    log.info("Fetching %s @ %s", manifest.repo_id, manifest.revision)
    downloaded = Path(hf_hub_download(
        repo_id=manifest.repo_id,
        revision=manifest.revision,
        filename=manifest.file_path,
        repo_type="dataset",
    ))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded, output_path)
    log.info("Wrote raw dataset -> %s", output_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())
