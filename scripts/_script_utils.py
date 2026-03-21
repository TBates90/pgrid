"""Shared helpers for pgrid CLI scripts."""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env(key: str, fallback: str | None = None) -> str | None:
    """Read *key* from ``.env`` at the project root (no dependencies)."""
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
    return os.environ.get(key, fallback)
