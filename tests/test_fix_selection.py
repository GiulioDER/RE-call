"""The CLI and selector require independent, exact observations before serving a fix."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from recall.fix_selection import FixRequest, select_fix
from tests.audit_memory_support import EXPECTED, REPO, copy_memos, lite_dsn
from tests.lite_cli_support import _offline, run_cli  # noqa: F401  # autouse fixture

REVISION = "a" * 40
COMMIT = "b" * 40
NODE = "tests/test_score.py::test_duplicate_rows"
MESSAGE = "expected duplicate row refusal"
NOW = datetime(2026, 11, 24, tzinfo=UTC)


def candidate(key: str = "right", **fields):
    return {"id": key, "pre_fix_revision": REVISION, "fix_commit": COMMIT,
            "test_node": NODE, "symptom_groups": [["duplicate"], ["question rows"]],
            "assertion_text": "duplicate question rows", "failure_marker": MESSAGE,
            "repair": "Reject the second row for a question.", **fields}


def envelope(key: str = "right", **fields):
    return {"id": key, "kind": "bug", "superseded_by": None, "expired": False,
            "source": f"{key}.md", "recall_record": {
                "summary": "Compare runs", "files": ["recall/score.py"],
                "tier": "proved", "fix_commit": COMMIT,
            }, **fields}


def observation(**fields):
    return {"revision": REVISION, "test_node": NODE, "outcome": "fail", "diagnostic": MESSAGE, **fields}


def decide(*, records=None, **fields):
    payload = {"report": "I see duplicate question rows", "revision": REVISION,
               "candidates": [candidate()], "observations": [observation()], **fields}
    return select_fix(records if records is not None else [envelope()], FixRequest.model_validate(payload), now=NOW)


def test_exact_regression_witness_serves_one_fix() -> None:
    assert decide()["fix"] == {
        "id": "right", "source": "right.md", "source_commit": COMMIT,
        "verification": NODE, "repair": "Reject the second row for a question.",
    }


@pytest.mark.parametrize("seen,reason", [
    ([], "missing_observation"),
    ([observation(test_node="tests/test_score.py::test_other")], "missing_observation"),
    ([observation(revision="c" * 40)], "missing_observation"),
    ([observation(outcome="pass")], "failure_not_matched"),
    ([observation(outcome="error")], "failure_not_matched"),
    ([observation(diagnostic="")], "failure_not_matched"),
    ([observation(diagnostic="unrelated assertion")], "failure_not_matched"),
    ([observation(diagnostic=f"expected something else\n{MESSAGE}")], "failure_not_matched"),
    ([observation(), observation(outcome="pass")], "ambiguous_observation"),
])
def test_failure_requires_the_exact_diagnostic(seen, reason) -> None:
    """Red proof: removing the diagnostic comparison makes the empty and wrong variants serve."""
    result = decide(observations=seen)
    assert result["decision"] == "abstain" and result["reason"] == reason and "fix" not in result


@pytest.mark.parametrize("records", [
    [],
    [envelope(superseded_by="<withheld>")],
    [envelope(superseded_by="<ambiguous>")],
    [envelope(expired=True)],
    [envelope(valid_from="2026-11-25")],
    [envelope(valid_from="bad metadata")],
    [envelope(), envelope()],
])
def test_no_fix_from_missing_or_ineligible_record(records) -> None:
    assert decide(records=records)["decision"] == "abstain"


def test_declared_fix_and_pre_fix_revision_must_match() -> None:
    assert decide(candidates=[candidate(fix_commit="c" * 40)])["decision"] == "abstain"
    assert decide(candidates=[candidate(pre_fix_revision="c" * 40)])["decision"] == "abstain"


def test_file_scope_prioritizes_three_and_retains_lexical_fallback() -> None:
    records = [envelope(key) for key in ("a", "b", "c", "d")]
    records[3]["recall_record"]["files"] = ["recall/winner.py"]
    manifests = [
        candidate(key, symptom_groups=[["separate"], ["retry"]], test_node=f"tests/test_score.py::test_{key}")
        for key in ("a", "b", "c")
    ]
    manifests.append(candidate("d"))
    result = decide(records=records, candidates=manifests, file_hints=["recall/winner.py"])
    assert [row["id"] for row in result["shortlist"]] == ["d", "a", "b"]
    assert result["fix"]["id"] == "d"


def test_a_failing_candidate_does_not_resolve_symptom_ambiguity() -> None:
    assert decide(report="A different regression")["decision"] == "abstain"
    candidates = [candidate(), candidate("other", test_node="tests/test_score.py::test_other")]
    result = decide(records=[envelope(), envelope("other")], candidates=candidates)
    assert result["reason"] == "ambiguous_symptom" and "fix" not in result


def test_cli_admits_only_policy_filtered_current_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = ["--embedder", "hashing", "--dsn", lite_dsn(tmp_path / "memory.db")]
    out = run_cli([*base, "index", str(copy_memos(tmp_path / "memos"))])
    assert "indexed" in out, out
    bug = EXPECTED["RE-call__abcdef012345__BUG-002.md"]
    request = tmp_path / "fix-request.json"
    request.write_text(json.dumps({
        "report": "I see duplicate question rows", "revision": REVISION,
        "candidates": [candidate(bug["id"], fix_commit=bug["fix_commit"])],
        "observations": [observation()],
    }), encoding="utf-8")
    call = [*base, "records", "--repo", REPO, "--select-fix", str(request)]
    result = json.loads(run_cli(call))
    assert result["decision"] == "serve" and result["fix"]["source_commit"] == bug["fix_commit"]

    policy = tmp_path / "deny.json"
    policy.write_text(json.dumps({"default_deny": True, "rules": []}), encoding="utf-8")
    monkeypatch.setenv("RECALL_SOURCE_POLICY_FILE", str(policy))
    result = json.loads(run_cli(call))
    assert result["decision"] == "abstain" and not result["shortlist"] and "fix" not in result


def test_cli_reports_missing_store_without_creating_it(tmp_path: Path) -> None:
    store = tmp_path / "absent.db"
    request = tmp_path / "fix-request.json"
    request.write_text(json.dumps({"report": "row", "revision": REVISION, "candidates": [candidate()]}), encoding="utf-8")
    result = json.loads(run_cli(["--dsn", lite_dsn(store), "records", "--repo", REPO, "--select-fix", str(request)]))
    assert result["decision"] == "unavailable" and not store.exists()



@pytest.mark.parametrize("withhold_successor", [False, True])
def test_cli_vetoes_an_explicitly_superseded_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, withhold_successor: bool,
) -> None:
    """Red proof: select_fix returned no_relevant_candidates instead of the supersession veto."""
    monkeypatch.delenv("RECALL_SOURCE_POLICY_FILE", raising=False)
    base = ["--embedder", "hashing", "--dsn", lite_dsn(tmp_path / "memory.db")]
    folder = copy_memos(tmp_path / "memos")
    old_name = "RE-call__fedcba987654__P2-4.md"
    new_name = "RE-call__abcdef012345__BUG-002.md"
    # Make the predecessor eligible on its own merits so the supersession edge is decisive.
    old = EXPECTED[old_name] | {"tier": "proved", "evidence": EXPECTED[new_name]["evidence"]}
    predecessor = folder / old_name
    lines = predecessor.read_text(encoding="utf-8").splitlines()
    predecessor.write_text(
        "\n".join("recall_record: " + json.dumps(old) if line.startswith("recall_record: ") else line
                  for line in lines) + "\n",
        encoding="utf-8",
    )
    request = tmp_path / "fix-request.json"
    request.write_text(json.dumps({
        "report": "I see duplicate question rows", "revision": REVISION,
        "candidates": [candidate(old["id"], fix_commit=old["fix_commit"])],
        "observations": [observation()],
    }), encoding="utf-8")
    call = [*base, "records", "--repo", REPO, "--select-fix", str(request)]
    current = folder / new_name
    successor_text = current.read_bytes()
    current.unlink()
    out = run_cli([*base, "index", str(folder)])
    assert "indexed" in out, out
    assert json.loads(run_cli(call))["fix"]["id"] == old["id"]

    current.write_bytes(successor_text)
    out = run_cli([*base, "index", str(folder)])
    assert "indexed" in out, out
    if withhold_successor:
        policy = tmp_path / "policy.json"
        policy.write_text(json.dumps({
            "default_deny": True,
            "rules": [{"prefix": old_name, "purposes": ["indexing", "retrieval"],
                       "principals": [], "redactions": []}],
        }), encoding="utf-8")
        monkeypatch.setenv("RECALL_SOURCE_POLICY_FILE", str(policy))
        out = run_cli([*base, "index", str(predecessor)])
        assert "indexed" in out, out

    history = json.loads(run_cli([*base, "records", "--id", old["id"], "--json"]))["records"]
    assert len(history) == 1
    assert history[0]["superseded_by"] == ("<withheld>" if withhold_successor else new_name)
    assert json.loads(run_cli(call)) == {"decision": "abstain", "reason": "superseded_fix", "shortlist": []}


@pytest.mark.parametrize("prefix", [b"", b"\xef\xbb\xbf"])
def test_stdin_request_decodes_as_utf8_whatever_the_console_code_page(
    monkeypatch: pytest.MonkeyPatch, prefix: bytes,
) -> None:
    """Standard input is read as UTF-8 bytes, as a request file is, not in the console code page.

    Red proof (CCA BUG-001, run 20261124T214848Z): with the pre-fix `sys.stdin.read` in
    `recall.cli_commands.records_cmd._read_fix_request`, a cp1252 stdin turned the arrow into
    mojibake ([b""] fails the repair assertion) and a UTF-8 BOM was refused ([BOM] exits).
    """
    import io

    from recall.cli_commands.records_cmd import _read_fix_request

    payload = {"report": "row", "revision": REVISION,
               "candidates": [candidate(repair="Reject → the second row.")]}
    stream = io.TextIOWrapper(io.BytesIO(prefix + json.dumps(payload, ensure_ascii=False).encode("utf-8")),
                              encoding="cp1252")
    monkeypatch.setattr("sys.stdin", stream)
    assert _read_fix_request("-").candidates[0].repair == "Reject → the second row."


def test_stdin_anchors_decode_as_utf8_whatever_the_console_code_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--anchors -` shares the request reader's decoding (CCA BUG-001 sibling, run 20261124T214848Z).

    Red proof: with the pre-fix `sys.stdin.read()` in `_read_anchors`, a cp1252 stdin turned the
    non-ASCII path into mojibake and the equality assertion failed.
    """
    import io

    from recall.cli_commands.records_cmd import _read_anchors

    anchors = [{"file": "docs/café.md", "symbol": None, "code_sha": None}]
    data = json.dumps(anchors, ensure_ascii=False).encode("utf-8")
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(data), encoding="cp1252"))
    assert _read_anchors("-") == anchors
