"""
Shared helpers for dataset-processing scripts.
"""

import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

# Invariant enforced by all-MiniLM-L6-v2.
MAX_PROMPT_TOKENS = 256

def assert_in_venv():
    """
    Fail loud with a clear, actionable message if not running inside a 
    virtual environment. MUST be called before any third-party import.
    """
    if sys.prefix == sys.base_prefix:
        sys.exit(
            "This script uses third-party libraries, so it MUST be run inside a venv.\n"
            "Fix: Move your cwd to strix/scripts/ && uv run python3 <script> ...\n"
        )
    
def repo_root() -> Path:
    """Resolve repository root from the script's location."""
    script_dir = Path(__file__).resolve().parent
    result = subprocess.run(
        ["git", "-C", str(script_dir), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    )
    root = result.stdout.strip()
    assert root, "Cannot resolve repository root. Is this script inside a git checkout?"
    return Path(root)

def default_tokenizer_path() -> Path:
    """Default path to tokenizer.json."""
    return repo_root() / "model" / "tokenizer.json"

def default_cache_dir() -> Path:
    """Raw fetched datasets caching directory."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "strix" / "datasets"

REPO_ID_FIELD = "repo_id"
REVISION_FIELD = "revision"
FILE_PATH_FIELD = "file_path"
LOCAL_NAME_FIELD = "local_name"
REQUIRED_MANIFEST_KEYS = {
    REPO_ID_FIELD, REVISION_FIELD, FILE_PATH_FIELD, LOCAL_NAME_FIELD,
}

# No whitespace, single forward slash '/'
_REPO_ID_REGEX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")

# No whitespace
_LOCAL_NAME_REGEX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

@dataclass(frozen=True)
class Manifest:
    repo_id: str
    revision: str
    file_path: str
    local_name: str

    @property
    def raw_ext(self) -> str:
        """Original dataset extension on HuggingFace (e.g. '.json')."""
        return Path(self.file_path).suffix

def _commit_hash_lookalike(revision: str) -> bool:
    return len(revision) == 40 and all(c in "0123456789abcdef" for c in revision.lower())

def load_manifest(info_toml: Path) -> Manifest:
    """Reads manifest file from specified path and sanitizes its values."""
    with info_toml.open("rb") as fin:
        raw = tomllib.load(fin)

    missing = REQUIRED_MANIFEST_KEYS - set(raw.keys())
    if missing:
        sys.exit(f"{info_toml} - Missing required key(s): {sorted(missing)}")

    repo_id = raw[REPO_ID_FIELD]
    if not isinstance(repo_id, str) or not _REPO_ID_REGEX.fullmatch(repo_id):
        sys.exit(
            f"{info_toml} - Invalid '{REPO_ID_FIELD}': {repo_id!r} "
            f"Expected format \"owner/dataset_name\", no whitespace and a single '/'."
        )

    revision = raw[REVISION_FIELD]
    if not isinstance(revision, str) or not _commit_hash_lookalike(revision):
        sys.exit(
            f"{info_toml} - Invalid '{REVISION_FIELD}': {revision!r} "
            f"Expected a 40-character commit hash, not a branch/tag name."
        )

    file_path = raw[FILE_PATH_FIELD]
    if not isinstance(file_path, str) or not file_path.strip():
        sys.exit(f"{info_toml} - '{FILE_PATH_FIELD}' must be a non-empty string.")

    local_name = raw[LOCAL_NAME_FIELD]
    if not isinstance(local_name, str) or not _LOCAL_NAME_REGEX.fullmatch(local_name):
        sys.exit(
            f"{info_toml} - Invalid '{LOCAL_NAME_FIELD}': {local_name!r} "
            f"Only [A-Za-z0-9_-], no whitespace, no dot."
        )

    return Manifest(
        repo_id=repo_id, revision=revision,
        file_path=file_path, local_name=local_name,
    )

def raw_dataset_path(manifest: Manifest) -> Path:
    """Default raw dataset path resolver."""
    return default_cache_dir() / f"{manifest.local_name}{manifest.raw_ext}"
