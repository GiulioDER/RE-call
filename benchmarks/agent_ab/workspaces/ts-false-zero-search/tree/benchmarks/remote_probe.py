"""Probe the worker host's corpus."""

import subprocess

CORPUS = "/home/sentiment/recall-repos/memory"


def count_files() -> int:
    out = subprocess.run(["ssh", "worker", f"ls {CORPUS} | wc -l"], capture_output=True)
    return int(out.stdout or 0)
