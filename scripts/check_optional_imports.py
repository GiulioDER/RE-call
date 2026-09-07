"""Verify optional dependency profiles in isolated install environments.

The CI matrix installs one profile per job.  This checker then imports the package surface that
profile promises, including the guarded adapter path for the deliberately empty ``llamaindex``
compatibility marker.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROFILE_IMPORTS: dict[str, tuple[str, ...]] = {
    "core": ("recall", "recall_agent", "recall_mcp"),
    "agent-ab": ("scipy",),
    "analysis": (
        "grimp",
        "importlinter",
        "vulture",
        "deptry",
        "pytest_randomly",
        "pytest_benchmark",
    ),
    "bench": ("mem0", "openai", "pyarrow", "numpy", "tiktoken"),
    "fastembed": ("fastembed", "onnxruntime", "tokenizers", "recall.embeddings"),
    "mcp": ("mcp", "anyio", "psycopg_pool", "jwt", "httpx2", "recall_mcp.server"),
    "agent": ("claude_agent_sdk", "recall_agent._sdk"),
    "dev": (
        "pytest",
        "pytest_timeout",
        "pytest_cov",
        "hypothesis",
        "xdist",
        "dotenv",
        "mcp",
        "psycopg_pool",
        "jwt",
        "langchain_core",
        "numpy",
        "tiktoken",
    ),
    "documents": (
        "pypdf",
        "pdfplumber",
        "docx",
        "openpyxl",
        "pptx",
        "xlrd",
        "bs4",
        "oxmsg",
    ),
    "entail": ("sentence_transformers",),
    "eval": ("matplotlib", "numpy"),
    "extract": ("openai",),
    "finetune": ("sentence_transformers", "datasets", "accelerate", "numpy"),
    "openai": ("openai",),
    "aws": ("boto3", "redis"),
    "pool": ("psycopg_pool",),
    "rerank": ("sentence_transformers",),
    "s3": ("boto3",),
    "sparse": ("transformers", "torch"),
    "voyage": ("voyageai",),
    "desktop": (
        "PySide6",
        "keyring",
        "mcp",
        "fastembed",
        "recall.desktop.ui",
        "recall.desktop.runtime",
    ),
    "langchain": ("langchain_core", "recall.integrations.langchain"),
    "llamaindex": (),
}


def _import_all(modules: Iterable[str]) -> list[str]:
    failures: list[str] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:  # BROAD-CATCH: error-translation
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    return failures


def verify_profile(profile: str) -> list[str]:
    """Return import contract failures for one independently installed profile."""

    if profile not in PROFILE_IMPORTS:
        return [f"unknown profile {profile!r}; choose one of {', '.join(PROFILE_IMPORTS)}"]

    failures = _import_all(PROFILE_IMPORTS[profile])
    if profile == "agent" and not failures:
        try:
            sdk = importlib.import_module("recall_agent._sdk")
            sdk._import_sdk()
        except Exception as exc:  # BROAD-CATCH: error-translation
            failures.append(f"recall_agent._sdk._import_sdk: {type(exc).__name__}: {exc}")

    if profile == "llamaindex":
        try:
            importlib.import_module("recall.integrations.llamaindex")
        except ModuleNotFoundError as exc:
            message = str(exc)
            if "llama-index-core" not in message:
                failures.append(
                    "recall.integrations.llamaindex raised an unexpected missing dependency: "
                    f"{message}"
                )
        except Exception as exc:  # BROAD-CATCH: error-translation
            failures.append(
                "recall.integrations.llamaindex: " f"{type(exc).__name__}: {exc}"
            )

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--profile", choices=tuple(PROFILE_IMPORTS))
    group.add_argument("--all", action="store_true", help="verify every declared profile")
    args = parser.parse_args(argv)
    profiles = tuple(PROFILE_IMPORTS) if args.all else (args.profile,)
    failed = False
    for profile in profiles:
        failures = verify_profile(profile)
        if failures:
            failed = True
            print(f"Optional import profile {profile!r} FAILED:")
            print("\n".join(f"  {failure}" for failure in failures))
        elif profile == "llamaindex":
            print("Optional import profile 'llamaindex' passed: compatibility guard is active")
        else:
            print(f"Optional import profile {profile!r} passed")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
