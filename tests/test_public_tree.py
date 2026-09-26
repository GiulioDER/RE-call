"""The public-tree check (``scripts/check_public_tree.py``) that keeps infrastructure and private
memory content out of this public repository (docs/results/REDACTION-2026-09-26.md).

Red proof, 2026-09-26, each mutation to ``scripts/check_public_tree.py`` alone, then restored:
- adding Tailscale's CGNAT network to ``EXEMPT_NETWORKS`` (exempt Tailscale addresses):
  ``test_a_tailscale_address_is_refused`` failed on ``== ["ipv4"]``.
- removing ``(?!redacted-)`` from the ``memo-path`` rule (refuse pseudonyms too):
  ``test_another_projects_memo_path_is_refused_but_its_pseudonym_passes`` failed on ``== []``.
- ``mask`` returning the value unchanged:
  ``test_a_finding_is_printed_masked`` failed on ``not in``.
Values in this file are built by concatenation so the file itself passes the check.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "check_public_tree", Path(__file__).parents[1] / "scripts" / "check_public_tree.py"
)
assert SPEC and SPEC.loader
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def rules(text: str, path: str = "results/x.json", allowed: frozenset[str] = frozenset()) -> list[str]:
    return [rule for _, rule, _ in check.findings(path, text, set(allowed))]


TAILSCALE = "100." + "91.7.9"
PUBLIC = "194." + "163.1.1"


def test_a_tailscale_address_is_refused() -> None:
    assert rules(f"servers live at {TAILSCALE}") == ["ipv4"]


def test_a_public_address_is_refused() -> None:
    assert rules(f"ssh user@{PUBLIC}") == ["ipv4"]


def test_loopback_private_documentation_and_netmask_ranges_pass() -> None:
    text = "127.0.0.1 0.0.0.0 10.0.0.7 172.16.4.2 192.168.1.1 192.0.2.10 203.0.113.9 255.255.255.0"
    assert rules(text) == []


def test_a_lock_file_skips_the_address_rule_only() -> None:
    assert rules(f"version = {PUBLIC}", path="uv.lock") == []
    assert rules("SHA256:" + "A" * 43, path="uv.lock") == ["ssh-fingerprint"]


def test_fingerprints_keys_server_ids_and_personal_mail_are_refused() -> None:
    assert rules("SHA256:" + "a" * 43) == ["ssh-fingerprint"]
    assert rules("-----BEGIN OPENSSH " + "PRIVATE KEY-----") == ["private-key"]
    assert rules("host " + "vmi" + "1234567") == ["server-id"]
    assert rules("mail someone" + "@gmail.com") == ["personal-mail"]


def test_another_projects_memo_path_is_refused_but_its_pseudonym_passes() -> None:
    assert rules("source " + "sentiment-agent/" + "project_trading_notes.md") == ["memo-path"]
    assert rules("source " + "sentiment-agent/" + "redacted-0123456789ab") == []


def test_memo_body_headings_and_session_ids_are_refused() -> None:
    assert rules("**" + "Why:" + "** because") == ["memo-body"]
    assert rules("**" + "How to apply:" + "** always") == ["memo-body"]
    assert rules("originSessionId: " + "9d32dfb3-b630-4435") == ["memo-body"]
    assert rules("originSessionId: codex-session-id") == []


def test_an_allowlisted_exact_value_passes_and_nothing_else_does() -> None:
    assert rules("see section 3.1.3.7", allowed=frozenset({"3.1.3.7"})) == []
    assert rules("see section " + "3.1.3." + "8", allowed=frozenset({"3.1.3.7"})) == ["ipv4"]


def test_a_finding_is_printed_masked() -> None:
    printed = check.mask(TAILSCALE)
    assert TAILSCALE not in printed
    assert printed.startswith(TAILSCALE[:4])
