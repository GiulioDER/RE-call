#!/usr/bin/env python3
"""Derive the pip-audit exceptions that a dependency's version cap makes unavoidable.

`--cap pillow:fastembed` ignores a pillow advisory only when ALL of these hold:

- pillow and fastembed are both pinned in the exported lock;
- OSV lists the advisory against the pinned pillow version;
- the advisory names at least one fixed pillow version;
- every fixed version is rejected by every pillow requirement the pinned fastembed declares,
  whatever its environment marker.

So an advisory with no fix, or with a fix the cap allows, still fails the audit. The exception
retires itself: once fastembed allows a fixed pillow, nothing is ignored and the audit fails until
the lock moves to it; once the lock pins a fixed pillow, OSV lists nothing against it.

Prints the `--ignore-vuln` arguments on stdout and the reason for each decision on stderr. Any
network or parse failure exits nonzero, so the audit fails rather than running with a guessed
exception list.

Distinct marker-specific versions of either selected package are refused before fetching:
one global audit exception cannot describe several dependency-version policies. Repeated
identical pins and variants of unrelated packages remain supported.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)")
# The ids end up as shell words in the audit step, so anything else is refused, not quoted.
_ADVISORY_ID = re.compile(r"^[A-Z][A-Z0-9]*-[A-Za-z0-9][A-Za-z0-9._:-]*$")
_TIMEOUT = 30


@dataclass(frozen=True)
class Decision:
    advisory: str
    ids: tuple[str, ...]
    fixed: tuple[str, ...]
    ignored: bool
    reason: str


def locked_versions(
    lock_text: str, *, require_unique: frozenset[str] = frozenset()
) -> dict[str, Version]:
    pins: dict[str, Version] = {}
    for line in lock_text.splitlines():
        match = _PIN.match(line.strip())
        if match and not line[:1].isspace():
            name = canonicalize_name(match.group(1))
            version = Version(match.group(2))
            if name in require_unique and name in pins and pins[name] != version:
                raise ValueError(f"multiple locked versions for {name}: {pins[name]} and {version}")
            pins[name] = version
    return pins


def _fixed_versions(advisory: dict, package: str) -> list[Version]:
    fixed: list[Version] = []
    for affected in advisory.get("affected", []):
        pkg = affected.get("package", {})
        if pkg.get("ecosystem") != "PyPI" or canonicalize_name(pkg.get("name", "")) != package:
            continue
        for rng in affected.get("ranges", []):
            if rng.get("type") != "ECOSYSTEM":
                continue
            fixed.extend(Version(e["fixed"]) for e in rng.get("events", []) if "fixed" in e)
    return sorted(set(fixed))


def _checked_ids(advisory: dict) -> tuple[str, ...]:
    ids = (advisory["id"], *advisory.get("aliases", []))
    for value in ids:
        if not _ADVISORY_ID.match(value):
            raise ValueError(f"refusing advisory id {value!r}: not a plain advisory identifier")
    return tuple(ids)


def capped_decisions(
    lock_text: str,
    capped: str,
    capper: str,
    fetch_requires_dist: Callable[[str, str], Iterable[str]],
    fetch_advisories: Callable[[str, str], Iterable[dict]],
) -> list[Decision]:
    capped, capper = canonicalize_name(capped), canonicalize_name(capper)
    pins = locked_versions(lock_text, require_unique=frozenset({capped, capper}))
    pinned, capper_pin = pins.get(capped), pins.get(capper)
    if pinned is None or capper_pin is None:
        return []
    caps = [
        req
        for req in map(Requirement, fetch_requires_dist(capper, str(capper_pin)))
        if canonicalize_name(req.name) == capped
    ]
    decisions = []
    for advisory in fetch_advisories(capped, str(pinned)):
        ids = _checked_ids(advisory)
        fixed = [v for v in _fixed_versions(advisory, capped) if v > pinned]
        if not caps:
            ignored, reason = False, f"{capper} {capper_pin} declares no {capped} requirement"
        elif not fixed:
            ignored, reason = False, "no fixed version is published"
        else:
            allowed = [
                str(v) for v in fixed if any(req.specifier.contains(v, prereleases=True) for req in caps)
            ]
            if allowed:
                ignored, reason = False, f"{capper} {capper_pin} allows the fix {', '.join(allowed)}"
            else:
                specs = "; ".join(sorted({str(req) for req in caps}))
                ignored, reason = True, f"every fix is forbidden by {capper} {capper_pin}: {specs}"
        decisions.append(
            Decision(advisory["id"], ids, tuple(str(v) for v in fixed), ignored, reason)
        )
    return decisions


def _get_json(url: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:  # noqa: S310 (fixed https hosts)
        return json.load(response)


def pypi_requires_dist(name: str, version: str) -> list[str]:
    info = _get_json(f"https://pypi.org/pypi/{name}/{version}/json")["info"]
    return info.get("requires_dist") or []


def osv_advisories(name: str, version: str) -> list[dict]:
    query: dict = {"package": {"name": name, "ecosystem": "PyPI"}, "version": version}
    advisories: list[dict] = []
    while True:
        page = _get_json("https://api.osv.dev/v1/query", query)
        advisories.extend(page.get("vulns", []))
        if not page.get("next_page_token"):
            return advisories
        query["page_token"] = page["next_page_token"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("requirements", type=Path)
    parser.add_argument(
        "--cap",
        action="append",
        default=[],
        metavar="CAPPED:CAPPER",
        help="e.g. pillow:fastembed; repeatable",
    )
    args = parser.parse_args(argv)
    lock_text = args.requirements.read_text(encoding="utf-8")
    words: list[str] = []
    for pair in args.cap:
        capped, sep, capper = pair.partition(":")
        if not sep or not capped or not capper:
            parser.error(f"--cap expects CAPPED:CAPPER, got {pair!r}")
        decisions = capped_decisions(lock_text, capped, capper, pypi_requires_dist, osv_advisories)
        print(f"{pair}: {len(decisions)} advisory(ies) against the pinned {capped}", file=sys.stderr)
        for d in decisions:
            verdict = "IGNORE" if d.ignored else "KEEP  "
            print(f"  {verdict} {d.advisory} {' '.join(d.ids[1:])}: {d.reason}", file=sys.stderr)
            if d.ignored:
                words.extend(w for i in d.ids for w in ("--ignore-vuln", i))
    print(" ".join(words))
    return 0


if __name__ == "__main__":
    sys.exit(main())
