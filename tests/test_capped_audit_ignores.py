"""Offline tests for scripts/capped_audit_ignores.py, the audit job's derived pillow exceptions.

Both fetchers are stubbed, so nothing here touches PyPI or OSV.

Red proof, 2026-10-21, each test run against one mutation of `capped_decisions` (or the named
helper) and seen to fail in its own assertion, then green on the restored source:

- test_ignores_an_advisory_whose_only_fix_the_cap_forbids: `ignored, reason = True` changed to
  `False` in the forbidden branch; fails `assert [d.ignored ...] == [True]`.
- test_keeps_an_advisory_whose_fix_the_cap_allows: `if allowed:` changed to `if False:`; fails
  `assert not decision.ignored`.
- test_retires_once_the_cap_lifts: same mutation as the previous one; fails `assert not ...`.
- test_keeps_an_advisory_with_no_published_fix: `elif not fixed:` changed to `elif False:`; the
  empty `fixed` list then makes `allowed` empty and the advisory is ignored; fails `assert not`.
- test_one_marker_variant_allowing_the_fix_keeps_it: `any(` changed to `all(` in the `allowed`
  comprehension; fails `assert not decision.ignored`.
- test_a_requirement_without_a_specifier_forbids_nothing: same mutation as the previous one, on
  the bare `pillow` requirement; fails `assert not`.
- test_nothing_is_ignored_when_the_capper_is_not_locked: `or capper_pin is None` removed; the
  script then fetches fastembed "None" and the stub's `pytest.fail` fires.
- test_refuses_an_advisory_id_that_is_not_a_plain_identifier: the `raise ValueError` in
  `_checked_ids` changed to `pass`; fails with `DID NOT RAISE`.
- test_ignores_only_fixes_newer_than_the_pinned_version: `if v > pinned` removed; the
  already-satisfied 12.0.0 fix is then allowed and the advisory kept; fails `assert d.ignored`.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from scripts.capped_audit_ignores import capped_decisions

LOCK = """\
fastembed==0.9.0 \\
    --hash=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
pillow==12.3.0 ; python_full_version >= '3.11' \\
    --hash=sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    # via fastembed
"""

FASTEMBED_CAP = [
    'pillow<13.0,>=10.3.0; python_version >= "3.10" and python_version < "3.13"',
    'pillow<13.0,>=12.0.0; python_version >= "3.14"',
    "numpy>=1.21",
]


def advisory(ghsa: str, cve: str, *fixed: str) -> dict:
    events = [{"introduced": "0"}, *({"fixed": f} for f in fixed)]
    return {
        "id": ghsa,
        "aliases": [cve],
        "affected": [
            {
                "package": {"name": "Pillow", "ecosystem": "PyPI"},
                "ranges": [
                    {"type": "GIT", "events": [{"introduced": "0"}, {"fixed": "abc123"}]},
                    {"type": "ECOSYSTEM", "events": events},
                ],
            }
        ],
    }


def stubs(requires: list[str], advisories: list[dict]):
    def fetch_requires_dist(name: str, version: str) -> Iterable[str]:
        assert (name, version) == ("fastembed", "0.9.0")
        return requires

    def fetch_advisories(name: str, version: str) -> Iterable[dict]:
        assert (name, version) == ("pillow", "12.3.0")
        return advisories

    return fetch_requires_dist, fetch_advisories


def decide(requires: list[str], advisories: list[dict], lock: str = LOCK):
    return capped_decisions(lock, "pillow", "fastembed", *stubs(requires, advisories))


def test_ignores_an_advisory_whose_only_fix_the_cap_forbids() -> None:
    decisions = decide(FASTEMBED_CAP, [advisory("GHSA-aaaa-bbbb-cccc", "CVE-2026-1", "13.0.0")])

    assert [d.ignored for d in decisions] == [True]
    assert decisions[0].ids == ("GHSA-aaaa-bbbb-cccc", "CVE-2026-1")


def test_keeps_an_advisory_whose_fix_the_cap_allows() -> None:
    [decision] = decide(FASTEMBED_CAP, [advisory("GHSA-aaaa-bbbb-cccc", "CVE-2026-1", "12.4.0")])

    assert not decision.ignored
    assert "allows the fix 12.4.0" in decision.reason


def test_retires_once_the_cap_lifts() -> None:
    lifted = ['pillow<14,>=10.3.0; python_version >= "3.10"']

    [decision] = decide(lifted, [advisory("GHSA-aaaa-bbbb-cccc", "CVE-2026-1", "13.0.0")])

    assert not decision.ignored


def test_keeps_an_advisory_with_no_published_fix() -> None:
    [decision] = decide(FASTEMBED_CAP, [advisory("GHSA-aaaa-bbbb-cccc", "CVE-2026-1")])

    assert not decision.ignored
    assert decision.reason == "no fixed version is published"


def test_one_marker_variant_allowing_the_fix_keeps_it() -> None:
    mixed = [*FASTEMBED_CAP, 'pillow>=12.0.0; python_version == "3.13"']

    [decision] = decide(mixed, [advisory("GHSA-aaaa-bbbb-cccc", "CVE-2026-1", "13.0.0")])

    assert not decision.ignored


def test_a_requirement_without_a_specifier_forbids_nothing() -> None:
    [decision] = decide([*FASTEMBED_CAP, "pillow"], [advisory("GHSA-a-b-c", "CVE-2026-1", "13.0.0")])

    assert not decision.ignored


def test_nothing_is_ignored_when_the_capper_is_not_locked() -> None:
    def unexpected(*_args: str) -> list:
        pytest.fail("nothing should be fetched when fastembed is not in the lock")

    lock = "pillow==12.3.0 \\\n    --hash=sha256:" + "b" * 64 + "\n"

    assert capped_decisions(lock, "pillow", "fastembed", unexpected, unexpected) == []


def test_refuses_an_advisory_id_that_is_not_a_plain_identifier() -> None:
    hostile = advisory("GHSA-aaaa-bbbb-cccc", "CVE-2026-1 --ignore-vuln X; true", "13.0.0")

    with pytest.raises(ValueError, match="refusing advisory id"):
        decide(FASTEMBED_CAP, [hostile])


def test_ignores_only_fixes_newer_than_the_pinned_version() -> None:
    # An old branch fix below the pin is already satisfied; only 13.0.0 can fix 12.3.0.
    [d] = decide(FASTEMBED_CAP, [advisory("GHSA-aaaa-bbbb-cccc", "CVE-2026-1", "12.0.0", "13.0.0")])

    assert d.ignored
    assert d.fixed == ("13.0.0",)
