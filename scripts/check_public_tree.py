"""Refuse infrastructure and private-memory content in files this public repository would publish.

GitHub's secret scanning looks for credential formats. It does not look for what leaked here in
2026-08/09 (docs/results/REDACTION-2026-09-26.md): host addresses, SSH host-key fingerprints,
hosting server IDs, personal mailboxes, and whole memos from the private memory store captured as
retrieval-candidate text in committed experiment traces. This check does.

Rules (each finding names its rule; the value is printed masked, so a CI log on a public
repository does not republish what it caught):

* ``ipv4``: an IPv4 address outside loopback, the unspecified address, RFC 1918, link-local,
  TEST-NET documentation ranges and netmasks. Tailscale's shared CGNAT range is NOT exempt.
* ``ssh-fingerprint``, ``private-key``: an OpenSSH SHA256 fingerprint or a private key block.
* ``server-id``: a hosting provider server identifier (``vmi`` + digits).
* ``personal-mail``: an address at a consumer mail domain.
* ``memo-path``: a path into another private project's memory (``sentiment-agent/...``) that is not
  a redaction pseudonym.
* ``memo-body``: the bold Why and How-to-apply section headings every stored memo carries, or a
  memo's ``originSessionId`` with a session UUID.

Exceptions live in ``scripts/public_tree_allowlist.txt`` as exact values with a reason, never as
patterns or whole files. Lock files are skipped for ``ipv4`` only (four-part version numbers).

    python scripts/check_public_tree.py            # every tracked file
    python scripts/check_public_tree.py --staged   # files staged for commit (pre-commit use)
    python scripts/check_public_tree.py FILE ...   # the named files
"""

from __future__ import annotations

import argparse
import ipaddress
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = Path(__file__).with_name("public_tree_allowlist.txt")

IPV4 = re.compile(r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\w.])")
RULES: dict[str, re.Pattern[str]] = {
    "ssh-fingerprint": re.compile(r"SHA256:[A-Za-z0-9+/]{43}"),
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"),
    "server-id": re.compile(r"\bvmi\d{5,}\b"),
    "personal-mail": re.compile(
        r"\b[A-Za-z0-9._%+-]+@(?:gmail|googlemail|hotmail|outlook|live|yahoo|icloud|me|proton|protonmail)\.[a-z]{2,}\b",
        re.IGNORECASE,
    ),
    "memo-path": re.compile(r"sentiment-agent/(?!redacted-)[A-Za-z0-9_][A-Za-z0-9_.-]*"),
    "memo-body": re.compile(
        r"\*\*Why:\*\*|\*\*How to apply:\*\*|originSessionId:\s*[0-9a-f]{8}-[0-9a-f]{4}-", re.IGNORECASE
    ),
}
EXEMPT_NETWORKS = tuple(
    ipaddress.ip_network(net)
    for net in (
        "0.0.0.0/32", "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
        "169.254.0.0/16", "192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24",
    )
)
LOCK_SUFFIXES = (".lock",)
SKIP_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz", ".whl", ".parquet")


def load_allowlist(path: Path = ALLOWLIST) -> set[str]:
    values = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                values.add(line.split("\t", 1)[0].strip())
    return values


def mask(value: str) -> str:
    return f"{value[:4]}... ({len(value)} chars)"


def ipv4_is_exempt(value: str) -> bool:
    address = ipaddress.ip_address(value)
    if value.startswith("255."):
        return True  # a netmask, not a host
    return any(address in net for net in EXEMPT_NETWORKS)


def findings(path: str, text: str, allowed: set[str]) -> Iterator[tuple[int, str, str]]:
    skip_ipv4 = path.endswith(LOCK_SUFFIXES)
    for number, line in enumerate(text.splitlines(), start=1):
        if not skip_ipv4:
            for match in IPV4.finditer(line):
                value = match.group(0)
                if value not in allowed and not ipv4_is_exempt(value):
                    yield number, "ipv4", value
        for rule, pattern in RULES.items():
            for match in pattern.finditer(line):
                if match.group(0) not in allowed:
                    yield number, rule, match.group(0)


def tracked(staged: bool) -> list[str]:
    command = (
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"] if staged else ["git", "ls-files"]
    )
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    return [line for line in result.stdout.splitlines() if line]


def scan(
    paths: Iterable[str], allowed: set[str], base: Path = ROOT
) -> tuple[list[tuple[str, int, str, str]], int, list[str]]:
    """Findings, the number of files actually read, and the named files that do not exist.

    A file that cannot be decoded as text (binary) is skipped; a file that does not exist is
    reported, because a check that quietly reads nothing reports clean for anything.
    """
    out = []
    read = 0
    missing = []
    for path in paths:
        if path.endswith(SKIP_SUFFIXES):
            continue
        target = base / path
        if not target.is_file():
            missing.append(path)
            continue
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        read += 1
        out.extend((path, number, rule, value) for number, rule, value in findings(path, text, allowed))
    return out, read, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()
    # Named files are read where the caller stands; tracked and staged files from the repository root.
    paths, base = (args.files, Path.cwd()) if args.files else (tracked(args.staged), ROOT)
    found, read, missing = scan(paths, load_allowlist(), base)
    for path in missing:
        print(f"{path}: does not exist", file=sys.stderr)
    for path, number, rule, value in found:
        print(f"{path}:{number}: {rule}: {mask(value)}")
    if missing and args.files:
        print(f"{len(missing)} named file(s) do not exist; nothing about them was checked.", file=sys.stderr)
        return 2
    if found:
        print(
            f"\n{len(found)} finding(s). This repository is public. Remove the value, take it from the "
            "environment, or, for a value that is genuinely not sensitive, add it with a reason to "
            "scripts/public_tree_allowlist.txt.",
            file=sys.stderr,
        )
        return 1
    print(f"public tree clean: {read} file(s) read")
    return 0


if __name__ == "__main__":
    sys.exit(main())
