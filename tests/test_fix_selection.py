"""The CLI and selector require independent, exact observations before serving a fix."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from recall.fix_selection import FixRequest, select_fix
from tests.audit_memory_support import EXPECTED, MEMOS, REPO, copy_memos, lite_dsn
from tests.conftest import TEST_DSN, requires_db
from tests.lite_cli_support import _offline, run_cli  # noqa: F401  # autouse fixture
from tests.test_generations import (  # noqa: F401  # `manager` is a fixture, used by name
    _Embedder,
    _manifest,
    _pipeline,
    _reader,
    _ready,
    manager,
)

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


@pytest.mark.parametrize("hint", ["recall/winner.py", "./recall/winner.py", "recall\\winner.py"])
def test_file_scope_prioritizes_three_and_retains_lexical_fallback(hint: str) -> None:
    """The file-hinted candidate ranks last on BM25 alone, so only the file sort shortlists it.

    Red proof (CCA BUG-002 and DAT-001, run 20261124T222911Z): with the `ranked.sort` file key in
    `recall.fix_selection.select_fix` removed, every case shortlists a, b, c and fails the
    shortlist assertion; without the `FixRequest.repository_paths` normalisation the `./` and
    backslash cases fail it the same way.
    """
    records = [envelope(key) for key in ("a", "b", "c", "d")]
    records[3]["recall_record"]["files"] = ["recall/winner.py"]
    manifests = [
        candidate(key, symptom_groups=[["separate"], ["retry"]],
                  test_node=f"tests/test_score.py::test_duplicate_question_rows_{key}")
        for key in ("a", "b", "c")
    ]
    manifests.append(candidate("d", test_node="tests/test_score.py::test_winner"))
    seen = [observation(test_node="tests/test_score.py::test_winner")]
    result = decide(records=records, candidates=manifests, file_hints=[hint], observations=seen)
    assert [row["id"] for row in result["shortlist"]] == ["d", "a", "b"]
    assert result["fix"]["id"] == "d"


def test_a_test_node_shared_with_another_eligible_repair_abstains() -> None:
    """Red proof (CCA BUG-003, run 20261124T222911Z): with the `ambiguous_test_node` check in
    `select_fix` removed, the one symptom match is served through a node another repair shares."""
    candidates = [candidate(), candidate("other", symptom_groups=[["separate"], ["retry"]])]
    result = decide(records=[envelope(), envelope("other")], candidates=candidates)
    assert result["reason"] == "ambiguous_test_node" and "fix" not in result


def test_an_invalid_request_names_the_file_and_the_failed_field(tmp_path: Path) -> None:
    """Red proof (CCA CODE-002, run 20261124T222911Z): the pre-fix `_read_fix_request` printed one
    generic sentence, naming neither the request file nor the field that failed validation."""
    request = tmp_path / "fix-request.json"
    request.write_text(json.dumps({"report": "row", "revision": REVISION,
                                   "candidates": [candidate(symptom_groups=[["duplicate"]])]}), encoding="utf-8")
    out = run_cli(["--dsn", lite_dsn(tmp_path / "memory.db"), "records", "--repo", REPO, "--select-fix", str(request)])
    assert "EXIT" in out and "fix-request.json" in out and "candidates.0.symptom_groups" in out, out
    assert "duplicate" not in out, out


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
    # CCA ENV-004 (run 20261124T222911Z): the decision names the snapshot it was read from.
    assert result["tenant_id"] == "default" and str(result["generation_id"]).startswith("lite-"), result

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
    assert result["reason"] == "store_missing" and result["tenant_id"] == "default" and result["generation_id"] is None


def test_a_legacy_store_without_a_snapshot_is_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CCA ENV-006 (run 20261124T222911Z): a legacy PostgreSQL table reads records and the
    supersession graph in two statements, so selection refuses it rather than mixing two states.

    Red proof: with the `snapshot` check in `records_cmd._select` removed, the stub store reaches
    `lookup_records`, which cannot read it, and the decision is never `unavailable`.
    """
    import contextlib

    from recall.cli_commands import records_cmd

    monkeypatch.delenv("RECALL_INDEX_MODE", raising=False)
    monkeypatch.setattr(records_cmd, "open_legacy_store", lambda *args, **kwargs: contextlib.nullcontext(object()))
    request = tmp_path / "fix-request.json"
    request.write_text(json.dumps({"report": "row", "revision": REVISION, "candidates": [candidate()]}), encoding="utf-8")
    out = run_cli(["--dsn", "postgresql://nobody@127.0.0.1:1/none", "records", "--repo", REPO,
                   "--select-fix", str(request)])
    assert '"reason": "snapshot_unavailable"' in out, out
    assert json.loads(out)["decision"] == "unavailable", out


@requires_db
def test_the_generation_route_serves_from_the_active_generation(
    manager, monkeypatch: pytest.MonkeyPatch  # noqa: F811  # the imported fixture
) -> None:
    """CCA ENV-006 (run 20261124T222911Z): the generation-backed store the docs name is served,
    and the decision carries that generation's id."""
    name = "RE-call__abcdef012345__BUG-002.md"
    data = (MEMOS / name).read_bytes()
    manifest = _manifest(manager.tenant_id, data)
    generation = _ready(manager, manifest, _pipeline("model-a"), _reader(manifest, data), _Embedder(1))
    manager.promote(generation, unsafe_development=True)
    monkeypatch.setenv("RECALL_INDEX_MODE", "generation")
    bug = EXPECTED[name]
    request = {"report": "I see duplicate question rows", "revision": REVISION,
               "candidates": [candidate(bug["id"], fix_commit=bug["fix_commit"])], "observations": [observation()]}
    monkeypatch.setattr("sys.stdin", _utf8_stdin(json.dumps(request).encode("utf-8")))
    out = run_cli(["--dsn", TEST_DSN, "--tenant", manager.tenant_id, "records", "--repo", REPO, "--select-fix", "-"])
    result = json.loads(out)
    assert result["decision"] == "serve" and result["generation_id"] == generation, out


def _utf8_stdin(data: bytes):
    import io

    return io.TextIOWrapper(io.BytesIO(data), encoding="utf-8")



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
    result = json.loads(run_cli(call))
    # The envelope also names its snapshot (CCA ENV-004); the decision itself is unchanged.
    assert {key: result[key] for key in ("decision", "reason", "shortlist")} == {
        "decision": "abstain", "reason": "superseded_fix", "shortlist": []}


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


@pytest.mark.parametrize("reader", ["_read_fix_request", "_read_anchors"])
@pytest.mark.parametrize("stdin", [None, "text-only"])
def test_unreadable_stdin_is_refused_like_an_unreadable_file(
    monkeypatch: pytest.MonkeyPatch, reader: str, stdin: str | None,
) -> None:
    """A closed or text-only standard input gives the records refusal, not a traceback.

    Red proof (CCA BUG-002 sibling, deferred run 20261125T110735Z): with the pre-fix
    `sys.stdin.buffer.read` in both `recall.cli_commands.records_cmd` readers, every case raises
    AttributeError instead of the SystemExit `pytest.raises` expects.
    """
    import io

    from recall.cli_commands import records_cmd

    monkeypatch.setattr("sys.stdin", None if stdin is None else io.StringIO("{}"))
    with pytest.raises(SystemExit, match="recall records: cannot read"):
        getattr(records_cmd, reader)("-")


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
