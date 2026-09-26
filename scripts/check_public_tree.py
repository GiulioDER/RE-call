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

* ``home-user``: a real account name in a home-directory path (``C:\\Users\\NAME``,
  ``/c/Users/NAME``, ``/home/NAME``, or a Claude Code project slug ``C--Users-NAME-``). Generic
  placeholders (``user``, ``runner``, ``alice``...) and anything in angle brackets pass.
* ``private-term``: a term from the maintainer's private deny list: host aliases, domains, private
  project and account names. The list is NOT in this repository, since publishing it would
  publish what it protects: it is read from the file named by ``RECALL_PUBLIC_TREE_DENYLIST``, one
  term per line, matched case-insensitively. A finding prints the term's index, never the term.
  Without the variable this rule is not checked, and the output says so.
* ``local-rules-file``: a tracked ``CLAUDE.local.md``, a maintainer's personal rules file.

Exceptions live in ``scripts/public_tree_allowlist.txt`` as exact values with a reason, never as
patterns or whole files. Lock files are skipped for ``ipv4`` only (four-part version numbers).
A ``.gz`` file is decompressed and checked; a file that is not text is listed as not checked, so a
clean result never hides what was not read.

``home-user`` and ``private-term`` are RATCHETED rather than zero-tolerance, because the tree
still held hundreds of such mentions when they were added (2026-09-26) in files due to leave it.
``scripts/public_tree_baseline.json`` records, per file, how many of each it had; a file over its
count, or a file with none recorded, fails exactly like any other finding. ``--write-baseline``
rewrites the record from the current tree, and its diff is the review: a count may fall, and a
rise has to be argued for in the pull request that makes it.

    python scripts/check_public_tree.py            # every tracked file
    python scripts/check_public_tree.py --staged   # files staged for commit (pre-commit use)
    python scripts/check_public_tree.py FILE ...   # the named files
"""

from __future__ import annotations

import argparse
import gzip
import ipaddress
import json
import os
import re
import subprocess
import sys
import zlib
from collections.abc import Iterable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = Path(__file__).with_name("public_tree_allowlist.txt")
BASELINE = Path(__file__).with_name("public_tree_baseline.json")
DENYLIST_ENV = "RECALL_PUBLIC_TREE_DENYLIST"
#: Rules whose existing hits are grandfathered per file by BASELINE; every other rule is absolute.
RATCHETED = ("home-user", "private-term")
LOCAL_RULES_FILE = "CLAUDE.local.md"

#: An account name in a home-directory path, in the four spellings this tree has carried.
HOME_USER = re.compile(
    r"(?i)(?:\b[a-z]:[\\/]{1,2}users[\\/]{1,2}|/[a-z]/users/|/home/|\b[a-z]--users-)"
    r"(?P<name>[A-Za-z][A-Za-z0-9_.]+)"  # two characters at least: `/home/u/` is a placeholder
)
#: Names that stand for "some user" rather than for anybody. Compared lowercased.
PLACEHOLDER_USERS = frozenset({
    "user", "users", "username", "name", "you", "me", "someone", "somebody", "example", "runner",
    "runneradmin", "redacted", "alice", "bob", "carol", "jane", "john", "dev", "developer", "admin",
    "public", "default", "shared", "ubuntu", "appuser", "app", "recall", "test", "tester", "given",
})

# Bounded by anything but a digit or a dot before it, and anything but a word character or a dot
# followed by a digit after it: an address that ends a sentence (so a full stop follows it) or is
# glued to a name (``host_`` before it) is still an address, while a longer dotted number is not.
IPV4 = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\w|\.\d)")
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
SKIP_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".whl", ".parquet")
#: Compressed text is decompressed and checked like any other file: committed ``.gz`` traces are
#: the same kind of file the 2026-09 leak was in.
GZIP_SUFFIX = ".gz"
#: Decompressed bytes read from one ``.gz`` file. The largest committed trace is 2.1 MB (measured
#: 2026-09-26 over all 42); a file over this is a finding, never a silent pass.
GZIP_MAX_BYTES = 64 * 1024 * 1024


def load_allowlist(path: Path = ALLOWLIST) -> set[str]:
    values = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                values.add(line.split("\t", 1)[0].strip())
    return values


def load_denylist(path: str | None) -> list[str]:
    """Private terms, lowercased, from a file outside the tree; empty when none is configured."""
    if not path:
        return []
    terms = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        term = line.strip().lower()
        if term and not term.startswith("#"):
            terms.append(term)
    return terms


def load_baseline(path: Path = BASELINE) -> dict[str, dict[str, int]]:
    if not path.exists():
        return {}
    data: dict[str, dict[str, int]] = json.loads(path.read_text(encoding="utf-8"))
    return data


def mask(value: str) -> str:
    return f"{value[:4]}... ({len(value)} chars)"


def ipv4_is_exempt(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:  # an octet with a leading zero: refused, and reported masked like any other
        return False
    if value.startswith("255."):
        return True  # a netmask, not a host
    return any(address in net for net in EXEMPT_NETWORKS)


def findings(
    path: str, text: str, allowed: set[str], terms: Iterable[str] = ()
) -> Iterator[tuple[int, str, str]]:
    skip_ipv4 = path.endswith(LOCK_SUFFIXES)
    terms = list(terms)
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
        for match in HOME_USER.finditer(line):
            name = match.group("name").rstrip(".").lower()
            if name not in PLACEHOLDER_USERS and match.group(0) not in allowed:
                yield number, "home-user", match.group(0)
        lowered = line.lower()
        for index, term in enumerate(terms):
            # The value reported is the term's index: the term itself must not reach a public log.
            for _ in range(lowered.count(term)):
                yield number, "private-term", f"#{index}"


def tracked(staged: bool) -> list[str]:
    command = (
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"] if staged else ["git", "ls-files"]
    )
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    return [line for line in result.stdout.splitlines() if line]


def scan(
    paths: Iterable[str], allowed: set[str], base: Path = ROOT, terms: Iterable[str] = ()
) -> tuple[list[tuple[str, int, str, str]], int, list[str], list[str]]:
    """Findings, the number of files read, the named files that do not exist, and those not read.

    A ``.gz`` file is decompressed and read. A file that is not text (a known binary suffix, or
    bytes that do not decode as UTF-8, compressed or not) is not read and is returned as skipped,
    so the caller can say what was not checked; a file that does not exist is reported, because a
    check that quietly reads nothing reports clean for anything.
    """
    out: list[tuple[str, int, str, str]] = []
    read = 0
    missing: list[str] = []
    skipped: list[str] = []
    terms = list(terms)
    for path in paths:
        if Path(path).name == LOCAL_RULES_FILE:
            out.append((path, 0, "local-rules-file", path))
        if path.endswith(SKIP_SUFFIXES):
            skipped.append(path)
            continue
        target = base / path
        if not target.is_file():
            missing.append(path)
            continue
        try:
            if path.endswith(GZIP_SUFFIX):
                # Streamed and bounded, so a crafted archive cannot exhaust the runner's memory.
                with gzip.open(target, "rb") as handle:
                    raw = handle.read(GZIP_MAX_BYTES + 1)
                if len(raw) > GZIP_MAX_BYTES:
                    out.append((path, 0, "gzip-over-limit", path))  # fail closed: unread is unchecked
                    continue
            else:
                raw = target.read_bytes()
            text = raw.decode("utf-8")
        except (UnicodeDecodeError, OSError, EOFError, zlib.error):
            skipped.append(path)
            continue
        read += 1
        out.extend(
            (path, number, rule, value) for number, rule, value in findings(path, text, allowed, terms)
        )
    return out, read, missing, skipped


def counts(found: Iterable[tuple[str, int, str, str]]) -> dict[str, dict[str, int]]:
    """Per file, how many findings of each ratcheted rule it holds."""
    table: dict[str, dict[str, int]] = {}
    for path, _, rule, _ in found:
        if rule in RATCHETED:
            table.setdefault(path, {}).setdefault(rule, 0)
            table[path][rule] += 1
    return table


def apply_baseline(
    found: list[tuple[str, int, str, str]], baseline: dict[str, dict[str, int]]
) -> tuple[list[tuple[str, int, str, str]], int, int]:
    """Findings still to report, how many were grandfathered, and how far the baseline could fall.

    A ratcheted rule's findings in a file are all grandfathered while the file holds no more of them
    than its recorded count, and all reported once it holds more, so the report shows every line a
    reviewer needs to find the new one. Every other rule is reported whatever the baseline says.
    """
    now = counts(found)
    over = {
        (path, rule)
        for path, rules in now.items()
        for rule, n in rules.items()
        if n > baseline.get(path, {}).get(rule, 0)
    }
    report = [f for f in found if f[2] not in RATCHETED or (f[0], f[2]) in over]
    grandfathered = len(found) - len(report)
    slack = sum(
        max(0, recorded - now.get(path, {}).get(rule, 0))
        for path, rules in baseline.items()
        for rule, recorded in rules.items()
    )
    return report, grandfathered, slack


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--staged", action="store_true")
    parser.add_argument(
        "--write-baseline", action="store_true",
        help="record the current per-file counts of the ratcheted rules (whole tree only)",
    )
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()
    terms = load_denylist(os.environ.get(DENYLIST_ENV))
    # Named files are read where the caller stands; tracked and staged files from the repository root.
    paths, base = (args.files, Path.cwd()) if args.files else (tracked(args.staged), ROOT)
    found, read, missing, skipped = scan(paths, load_allowlist(), base, terms)
    if args.write_baseline:
        if args.files or args.staged:
            print("--write-baseline records the whole tree; drop --staged and file names.", file=sys.stderr)
            return 2
        if not terms:
            print(f"--write-baseline needs {DENYLIST_ENV}, or private-term counts are lost.", file=sys.stderr)
            return 2
        table = counts(found)
        BASELINE.write_text(json.dumps(dict(sorted(table.items())), indent=1) + "\n", encoding="utf-8")
        print(f"wrote {BASELINE.name}: {sum(sum(r.values()) for r in table.values())} finding(s) in {len(table)} file(s)")
        return 0
    baseline = load_baseline()
    if not terms:
        # Unchecked is not absent: without the list, recorded private-term counts say nothing.
        baseline = {p: {r: n for r, n in rs.items() if r != "private-term"} for p, rs in baseline.items()}
    found, grandfathered, slack = apply_baseline(found, baseline)
    for path in missing:
        print(f"{path}: does not exist", file=sys.stderr)
    for path in skipped:
        print(f"{path}: not text, not checked", file=sys.stderr)
    if not terms:
        print(f"private-term: not checked, {DENYLIST_ENV} is not set", file=sys.stderr)
    if grandfathered:
        note = f"; the baseline could fall by {slack}" if slack else ""
        print(f"{grandfathered} ratcheted finding(s) within {BASELINE.name}{note}", file=sys.stderr)
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
    print(f"public tree clean: {read} file(s) read, {len(skipped)} not text and not read")
    return 0


if __name__ == "__main__":
    sys.exit(main())
