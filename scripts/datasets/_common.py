"""
Shared helpers for scripts across datasets/.
"""

import os
import subprocess
import sys
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
    """Default tokenizer.json directory."""
    return repo_root() / "model" / "tokenizer.json"

def default_cache_dir() -> Path:
    """Raw fetched datasets caching directory."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "strix" / "datasets"
