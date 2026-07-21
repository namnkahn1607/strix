#!/usr/bin/env python3
"""
Fetch raw dataset from HuggingFace Hub, pinned to an exact commit in info.toml.
Default destination directory to ~/.cache/strix/<dataset name>/.

'info.toml' format:
    repo_id   = "<dataset name>" (e.g. databricks/databricks-dolly-15k)
    revision  = "<full 40-char commit hash>" # NOT a branch/tag
    file_path = "<path to dataset>" # Path to the file WITHIN the repository.

Usage:
    uv run python3 datasets/fetch.py datasets/<dataset-name>/info.toml
    uv run python3 datasets/fetch.py datasets/<dataset-name>/info.toml \\
        --output <output-path.raw>
"""

import argparse
import logging
import sys
import tomllib
from pathlib import Path

from _common import assert_in_venv, default_cache_dir

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

REPO_ID_FIELD = "repo_id"
REVISION_FIELD = "revision"
FILE_PATH_FIELD = "file_path"
REQUIRED_KEYS = {REPO_ID_FIELD, REVISION_FIELD, FILE_PATH_FIELD}

def _looks_like_commit_hash(revision: str) -> bool:
    return len(revision) == 40 and all(c in "0123456789abcdef" for c in revision.lower())

def load_manifest(info_toml: Path) -> dict:
    with info_toml.open("rb") as f:
        manifest = tomllib.load(f)
    
    missing = REQUIRED_KEYS - set(manifest.keys())
    if missing:
        log.error("%s missing required key(s): %s", info_toml, sorted(missing))
        sys.exit(1)

    revision = manifest[REVISION_FIELD]
    if not isinstance(revision, str) or not _looks_like_commit_hash(revision):
        log.error(
            "'revision' at %s must be a full 40-character commit hash (got: %r)",
            info_toml, revision,
        )
        sys.exit(1)

    return manifest

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("info_toml", type=Path, help="Path to <dataset-dir>/info.toml")
    parser.add_argument(
        "--output", type=Path, default=None, 
        help="Output path for the raw dataset. Default: ~/.cache/strix/<dataset-dir>/"
    )

    args = parser.parse_args()
    info_toml = args.info_toml.resolve()
    if not info_toml.exists():
        log.error("File not found: %s", info_toml)
        return 1
    
    manifest = load_manifest(info_toml)
    name = info_toml.parent.name # The dataset name

    output_path = args.output.resolve() if args.output else default_cache_dir() / f"{name}.raw"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Fetching %s @ %s", manifest[REPO_ID_FIELD], manifest[REVISION_FIELD])
    downloaded = Path(hf_hub_download(
        repo_id=manifest[REPO_ID_FIELD],
        revision=manifest[REVISION_FIELD],
        filename=manifest[FILE_PATH_FIELD],
        repo_type="dataset",
    ))

    output_path.write_bytes(downloaded.read_bytes())
    log.info("Wrote raw dataset -> %s", output_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())
