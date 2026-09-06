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
    "fastembed": ("fastembed", "onnxruntime", "tokenizers", "recall.embeddings"),
    "mcp": ("mcp", "anyio", "psycopg_pool", "jwt", "httpx2", "recall_mcp.server"),
    "agent": ("claude_agent_sdk", "recall_agent._sdk"),
    "desktop": ("PySide6", "keyring", "recall.desktop.ui", "recall.desktop.runtime"),
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
    parser.add_argument("--profile", required=True, choices=tuple(PROFILE_IMPORTS))
    args = parser.parse_args(argv)
    failures = verify_profile(args.profile)
    if failures:
        print(f"Optional import profile {args.profile!r} FAILED:")
        print("\n".join(f"  {failure}" for failure in failures))
        return 1
    if args.profile == "llamaindex":
        print("Optional import profile 'llamaindex' passed: compatibility guard is active")
    else:
        print(f"Optional import profile {args.profile!r} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
