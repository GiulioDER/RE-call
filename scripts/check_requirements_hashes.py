#!/usr/bin/env python3
"""Check that every exported requirement is exact pinned and hash constrained."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


_REQUIREMENT = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)(?:\s*;[^\\]*)?(?:\s*\\)?$"
)
_HASH = re.compile(r"^--hash=sha256:[0-9a-f]{64}(?:\s*\\)?$")


def audit_requirements(path: Path) -> tuple[int, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    errors: list[str] = []
    packages = 0
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if (
            line
            and not line.startswith("#")
            and not line.startswith("--")
            and not raw_line[:1].isspace()
        ):
            match = _REQUIREMENT.match(line)
            if match is None:
                errors.append(
                    f"{path}:{index + 1}: requirement is not an exact == pin: {line}"
                )
                continue
        else:
            match = _REQUIREMENT.match(line)
        if match is None:
            continue
        packages += 1
        hashes: list[str] = []
        cursor = index + 1
        while cursor < len(lines):
            continuation = lines[cursor].strip()
            if continuation.startswith("--hash="):
                if _HASH.match(continuation) is None:
                    errors.append(f"{path}:{cursor + 1}: invalid hash for {match.group(1)}")
                else:
                    hashes.append(continuation)
                cursor += 1
                continue
            break
        if not hashes:
            errors.append(f"{path}:{index + 1}: {match.group(1)} has no sha256 hash")
    if packages == 0:
        errors.append(f"{path}: no exact pinned requirements found")
    return packages, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    packages, errors = audit_requirements(args.path)
    if errors:
        for error in errors:
            print(f"::error::{error}", file=sys.stderr)
        return 1
    print(f"verified {packages} exact requirements with sha256 hashes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
