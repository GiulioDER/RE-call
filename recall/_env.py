"""Minimal, dependency-free `.env` loader for local dev.

Reads ``KEY=VALUE`` lines from a `.env` file into ``os.environ`` WITHOUT overriding variables that
are already set. The `.env` file is gitignored — it is for local secrets (e.g. VOYAGE_API_KEY),
never committed. Entry points call ``load_dotenv()`` so those keys are picked up.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
