"""The public-tree check (``scripts/check_public_tree.py``) that keeps infrastructure and private
memory content out of this public repository (docs/results/REDACTION-2026-09-26.md).

Red proof, 2026-09-26, each mutation to ``scripts/check_public_tree.py`` alone, then restored:
- adding Tailscale's CGNAT network to ``EXEMPT_NETWORKS`` (exempt Tailscale addresses):
  ``test_a_tailscale_address_is_refused`` failed on ``== ["ipv4"]``.
- removing ``(?!redacted-)`` from the ``memo-path`` rule (refuse pseudonyms too):
  ``test_another_projects_memo_path_is_refused_but_its_pseudonym_passes`` failed on ``== []``.
- ``mask`` returning the value unchanged:
  ``test_a_finding_is_printed_masked`` failed on ``not in``.
- ``if not target.is_file(): missing.append(path); continue`` replaced by ``continue`` alone (skip
  missing files silently, the defect found 2026-09-26 when a scan from the wrong root read
  nothing and reported clean): ``test_a_named_file_that_does_not_exist_is_reported`` failed on
  ``== ["nope.json"]``.

Follow-up, 2026-09-26, from the security review of #777 (same method; the failure of each is in the
pull request):
- the old ``IPV4`` bounds, ``(?<![\\w.])`` and ``(?![\\w.])``, under which an address ending a
  sentence or glued to a name passed:
  ``test_an_address_ending_a_sentence_or_glued_to_a_name_is_refused``.
- ``ipv4_is_exempt`` without its ``except ValueError`` (a leading-zero octet crashed the checker and
  the traceback printed the value whole): ``test_a_leading_zero_address_is_a_finding_not_a_crash``.
- ``".gz"`` back in ``SKIP_SUFFIXES`` (compressed traces never read):
  ``test_a_gzip_trace_is_decompressed_and_checked``.
- the decode branch's ``skipped.append(path)`` removed (a file that is not text vanished from the
  report): ``test_a_file_that_is_not_text_is_reported_as_not_checked``.

Values in this file are built by concatenation so the file itself passes the check. The fixture
addresses are a Tailscale-range address and a public resolver address; neither is, or shares a
network with, any value this repository ever leaked.
"""

from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "check_public_tree", Path(__file__).parents[1] / "scripts" / "check_public_tree.py"
)
assert SPEC and SPEC.loader
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def rules(text: str, path: str = "results/x.json", allowed: frozenset[str] = frozenset()) -> list[str]:
    return [rule for _, rule, _ in check.findings(path, text, set(allowed))]


TAILSCALE = "100." + "64.0.9"
PUBLIC = "8.8." + "4.4"


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


def test_a_named_file_that_does_not_exist_is_reported(tmp_path: Path) -> None:
    (tmp_path / "here.json").write_text("{}", encoding="utf-8")
    found, read, missing, _skipped = check.scan(["here.json", "nope.json"], set(), tmp_path)
    assert found == []
    assert read == 1
    assert missing == ["nope.json"]


def test_an_address_ending_a_sentence_or_glued_to_a_name_is_refused() -> None:
    assert rules(f"the service answers at {TAILSCALE}.") == ["ipv4"]
    assert rules(f"host_{TAILSCALE} is up") == ["ipv4"]
    assert rules("section 1.2.3.4.5 of the spec") == []  # a longer dotted number is not an address


def test_a_leading_zero_address_is_a_finding_not_a_crash() -> None:
    value = TAILSCALE[:-1] + "09"
    assert rules(f"at {value}") == ["ipv4"]


def test_a_gzip_trace_is_decompressed_and_checked(tmp_path: Path) -> None:
    (tmp_path / "trace.jsonl.gz").write_bytes(gzip.compress(f"host {TAILSCALE}\n".encode()))
    found, read, missing, skipped = check.scan(["trace.jsonl.gz"], set(), tmp_path)
    assert [rule for _, _, rule, _ in found] == ["ipv4"]
    assert (read, missing, skipped) == (1, [], [])


def test_a_gzip_over_the_limit_is_a_finding_and_is_never_read_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OpenRouter review of #782, HIGH: an unbounded ``gzip.decompress`` could be exhausted by a
    crafted archive. Red proof: reading the whole file (no cap check) failed on ``== ["gzip-over-limit"]``."""
    monkeypatch.setattr(check, "GZIP_MAX_BYTES", 16)
    (tmp_path / "big.gz").write_bytes(gzip.compress(b"x" * 1_000))
    found, read, missing, skipped = check.scan(["big.gz"], set(), tmp_path)
    assert [rule for _, _, rule, _ in found] == ["gzip-over-limit"]
    assert (read, missing, skipped) == (0, [], [])


def test_a_corrupt_gzip_is_not_checked_rather_than_a_crash(tmp_path: Path) -> None:
    """OpenRouter review of #782, LOW: a corrupt deflate stream raises ``zlib.error``, which is not an
    ``OSError``. Red proof: ``zlib.error`` removed from the ``except`` failed with ``zlib.error``."""
    (tmp_path / "bad.gz").write_bytes(b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03" + b"\xff" * 20)
    found, read, missing, skipped = check.scan(["bad.gz"], set(), tmp_path)
    assert (found, read, missing, skipped) == ([], 0, [], ["bad.gz"])


def test_a_file_that_is_not_text_is_reported_as_not_checked(tmp_path: Path) -> None:
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00\x81")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG")
    found, read, missing, skipped = check.scan(["blob.bin", "logo.png"], set(), tmp_path)
    assert (found, read, missing) == ([], 0, [])
    assert sorted(skipped) == ["blob.bin", "logo.png"]


# ---- rules added 2026-09-26: account names, the private deny list, the local rules file, the ratchet

#: Built by concatenation so this file does not trip the rule it tests.
REAL_NAME = "mal" + "lory"


def test_a_real_account_name_in_a_home_path_is_refused_and_placeholders_pass() -> None:
    """Red proof: with the ``HOME_USER`` loop in ``findings`` removed, the four spellings fail on
    ``== ["home-user"]``; with the ``PLACEHOLDER_USERS`` test removed, the placeholder line fails
    on ``== []``."""
    for text in (
        "C:" + chr(92) + "Users" + chr(92) + REAL_NAME + chr(92) + "repo",
        "/c/Users/" + REAL_NAME + "/repo",
        "/home/" + REAL_NAME + "/repo",
        "~/.claude/projects/C--Users-" + REAL_NAME + "-Documents-repo",
    ):
        assert rules(text) == ["home-user"], text
    placeholders = "/home/alice/x /home/<user>/x /home/u/x /home/.../x /c/Users/runner/x"
    assert rules(placeholders) == []


def test_a_private_term_is_refused_and_reported_by_index_never_by_value() -> None:
    """Red proof: with the deny-list loop in ``findings`` removed, this fails on
    ``== [(1, "private-term", "#1")]``."""
    terms = ["alpha-host", "beta-host"]
    found = list(check.findings("docs/x.md", "deploy to BETA-HOST today", set(), terms))
    assert found == [(1, "private-term", "#1")]


def test_a_tracked_local_rules_file_is_refused(tmp_path: Path) -> None:
    """Red proof: with the ``LOCAL_RULES_FILE`` test in ``scan`` removed, this fails on
    ``== ["local-rules-file"]``."""
    (tmp_path / "CLAUDE.local.md").write_text("# private rules\n", encoding="utf-8")
    found, _, _, _ = check.scan(["CLAUDE.local.md"], set(), tmp_path)
    assert [rule for _, _, rule, _ in found] == ["local-rules-file"]


def test_the_ratchet_grandfathers_up_to_the_recorded_count_and_no_further() -> None:
    """Red proof, one mutation each, restored after:
    - ``n > baseline...`` changed to ``n >= baseline...``: the at-count case fails on ``== []``.
    - the ``f[2] not in RATCHETED`` term removed (every rule grandfathered): the ipv4 case fails on
      ``== ["ipv4"]``.
    """
    hits = [("a.md", 1, "home-user", "x"), ("a.md", 2, "home-user", "y")]
    at_count, grandfathered, slack = check.apply_baseline(hits, {"a.md": {"home-user": 2}})
    assert (at_count, grandfathered, slack) == ([], 2, 0)

    over, _, _ = check.apply_baseline(hits, {"a.md": {"home-user": 1}})
    assert over == hits, "every line is reported once a file exceeds its count"

    new_file, _, _ = check.apply_baseline(hits, {})
    assert new_file == hits

    shrunk, _, slack = check.apply_baseline(hits[:1], {"a.md": {"home-user": 2}})
    assert (shrunk, slack) == ([], 1)

    absolute, _, _ = check.apply_baseline([("a.md", 3, "ipv4", PUBLIC)], {"a.md": {"ipv4": 5}})
    assert [rule for _, _, rule, _ in absolute] == ["ipv4"]
