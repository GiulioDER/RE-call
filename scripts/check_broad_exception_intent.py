"""Require an explicit policy on every broad ``except Exception`` handler.

Broad catches are sometimes the correct boundary for optional providers, hooks, cleanup, and
security adapters.  They are also easy to copy into a new path without deciding what an exception
means there.  The marker is deliberately on the handler itself so a reviewer can audit the policy
without looking up a registry elsewhere.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

POLICIES = ("fail-open", "fail-closed", "cleanup-only", "error-translation")
_MARKER = re.compile(r"#\s*BROAD-CATCH:\s*([a-z-]+)\b", re.IGNORECASE)
DEFAULT_ROOTS = (
    "recall",
    "recall_mcp",
    "recall_agent",
    "recall_hooks",
    "recall_interop",
    "recall_consistency",
)


def _paths(arguments: list[str]) -> list[Path]:
    roots = [Path(argument) for argument in arguments] if arguments else [Path(root) for root in DEFAULT_ROOTS]
    return sorted(
        path
        for root in roots
        for path in (root.rglob("*.py") if root.is_dir() else (root,))
    )


def find_violations(arguments: list[str] | None = None) -> list[str]:
    violations: list[str] = []
    for path in _paths(arguments or []):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError) as exc:
            violations.append(f"{path}: {type(exc).__name__}: {exc}")
            continue

        lines = source.splitlines()
        for handler in ast.walk(tree):
            if not isinstance(handler, ast.ExceptHandler):
                continue
            if not isinstance(handler.type, ast.Name) or handler.type.id != "Exception":
                continue
            line = lines[handler.lineno - 1]
            matches = _MARKER.findall(line)
            if len(matches) != 1:
                violations.append(
                    f"{path}:{handler.lineno}: add exactly one '# BROAD-CATCH: "
                    f"<{'|'.join(POLICIES)}>' marker to this handler"
                )
                continue
            policy = matches[0].lower()
            if policy not in POLICIES:
                violations.append(
                    f"{path}:{handler.lineno}: unknown broad catch policy {policy!r}; "
                    f"choose one of {', '.join(POLICIES)}"
                )
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="production files or directories to inspect")
    args = parser.parse_args(argv)
    violations = find_violations(args.paths)
    if violations:
        print("Broad exception intent violations:")
        print("\n".join(violations))
        return 1
    print(f"Broad exception intent passed for {len(_paths(args.paths))} production files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
