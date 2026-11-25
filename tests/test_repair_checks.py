"""Exercise repair check selection and retry feedback through the CLI."""

import io
import json
import re
import sys
from pathlib import Path

import pytest

from recall.cli import main
from recall.cli_commands import repair_checks


@pytest.fixture
def request_payload() -> dict:
    return {
        "task": "Keep a note's date when renaming it.",
        "failure": "After reopening the note, the date is blank.",
        "failed_edit": "Update the title in the editor → save.",
        "checks": [
            {"id": "p1", "role": "preservation", "description": "Rename an undated note."},
            {"id": "d1", "role": "diagnostic", "description": "Inspect the editor title."},
            {"id": "d2", "role": "diagnostic", "description": "Reopen and inspect the stored date."},
        ],
        "prior_procedure": {
            "source": "cases/archive-round-trip.md",
            "procedure": "Measure metadata after a new read of the stored record.",
        },
    }


def _write(tmp_path: Path, name: str, payload: object) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_selection_only_uses_history_in_the_prompt(
    request_payload: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    request = _write(tmp_path, "request.json", request_payload)
    selection = _write(tmp_path, "selection.json", {"diagnostic": "d2", "preservation": "p1"})
    observations = _write(tmp_path, "observations.json", [
        {"id": "p1", "outcome": "pass"},
        {"id": "d2", "outcome": "fail", "detail": "The reopened date was empty."},
    ])
    # This path is unconfigured and must stay absent, even with a lite DSN.
    store = tmp_path / "missing.db"
    cli = ["--dsn", f"sqlite:///{store.as_posix()}", "repair-checks"]
    main([*cli, "prompt", request])
    output = json.loads(capsys.readouterr().out)
    assert output["input"] == request_payload
    assert output["instructions"]
    assert output["response_schema"]["$defs"]["_Selection"]["required"] == ["diagnostic", "preservation"]

    alternate = _write(tmp_path, "alternate.json", {
        key: value for key, value in request_payload.items() if key != "prior_procedure"
    })
    main([*cli, "prompt", alternate])
    assert "prior_procedure" not in json.loads(capsys.readouterr().out)["input"]

    main([*cli, "resolve", request, selection])
    assert json.loads(capsys.readouterr().out) == {
        "checks": [request_payload["checks"][2], request_payload["checks"][0]]
    }
    main([*cli, "feedback", request, selection, observations])
    feedback = capsys.readouterr().out
    assert json.loads(feedback) == {"checks": [
        request_payload["checks"][2] | {"outcome": "fail", "detail": "The reopened date was empty."},
        request_payload["checks"][0] | {"outcome": "pass"},
    ]}
    main([*cli, "feedback", alternate, selection, observations])
    assert capsys.readouterr().out == feedback
    assert not store.exists()


def test_stdin_decodes_as_utf8_whatever_the_console_code_page(
    request_payload: dict, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = json.dumps(request_payload, ensure_ascii=False).encode("utf-8-sig")
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(raw), encoding="cp1252"))
    main(["repair-checks", "prompt", "-"])
    assert json.loads(capsys.readouterr().out)["input"]["failed_edit"] == request_payload["failed_edit"]


@pytest.mark.parametrize("stdin", [None, io.StringIO("{}")], ids=["closed", "text-only"])
def test_unreadable_stdin_is_refused_like_an_unreadable_file(
    stdin: object, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Red proof (CCA BUG-002, deferred run 20261125T110735Z): with the pre-fix
    ``sys.stdin.buffer.read`` in ``recall/cli_commands/repair_checks.py`` ``_read_json``, both
    cases raise AttributeError instead of the refusal ``pytest.raises`` expects.
    """
    monkeypatch.setattr(sys, "stdin", stdin)
    with pytest.raises(SystemExit, match="could not read request"):
        main(["repair-checks", "prompt", "-"])
    assert capsys.readouterr().out == ""


def test_only_one_input_may_come_from_stdin(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Red proof (CCA CODE-003): with the ``paths.count("-") > 1`` guard in ``_run`` removed, the
    request is read from the captured stdin and the command exits with ``could not read request``,
    so the ``match`` fails.
    """
    observations = _write(tmp_path, "observations.json", [])
    with pytest.raises(SystemExit, match="read only one input from stdin"):
        main(["repair-checks", "feedback", "-", "-", observations])
    assert capsys.readouterr().out == ""


def test_input_over_the_size_limit_is_refused(
    request_payload: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Red proof (CCA CODE-003): with the ``len(raw) > _MAX_INPUT_BYTES`` guard in ``_read_json``
    removed, the read still stops one byte past the limit, so the truncated request is refused as
    ``invalid JSON`` and the ``match`` fails.
    """
    request_payload["failed_edit"] = "x" * repair_checks._MAX_INPUT_BYTES
    request = _write(tmp_path, "request.json", request_payload)
    with pytest.raises(SystemExit, match="request exceeds 512 KiB"):
        main(["repair-checks", "prompt", request])
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("selection", [
    {"diagnostic": "p1", "preservation": "p1"},
    {"diagnostic": "d2", "preservation": "unknown"},
])
def test_rejects_selection_outside_its_role(
    selection: dict, request_payload: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    request = _write(tmp_path, "request.json", request_payload)
    result = _write(tmp_path, "selection.json", selection)
    with pytest.raises(SystemExit, match="selection must name a menu check"):
        main(["repair-checks", "resolve", request, result])
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("observed_ids", [[], ["d2", "d2"], ["d1", "p1"], ["d2", "p1", "d1"]])
def test_rejects_incomplete_duplicate_or_unselected_observations(
    observed_ids: list[str], request_payload: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    request = _write(tmp_path, "request.json", request_payload)
    selection = _write(tmp_path, "selection.json", {"diagnostic": "d2", "preservation": "p1"})
    observations = _write(tmp_path, "observations.json", [
        {"id": check_id, "outcome": "pass"} for check_id in observed_ids
    ])
    with pytest.raises(SystemExit, match="observation"):
        main(["repair-checks", "feedback", request, selection, observations])
    assert capsys.readouterr().out == ""


def test_abstention_has_no_check_feedback(
    request_payload: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    request = _write(tmp_path, "request.json", request_payload)
    selection = _write(tmp_path, "selection.json", None)
    empty = _write(tmp_path, "observations.json", [])
    main(["repair-checks", "resolve", request, selection])
    assert json.loads(capsys.readouterr().out) == {"checks": []}
    main(["repair-checks", "feedback", request, selection, empty])
    assert json.loads(capsys.readouterr().out) == {"checks": []}


@pytest.mark.parametrize("invalid_case", ["duplicate-id", "missing-role", "extra-key"])
def test_invalid_menu_reports_no_request_text(
    invalid_case: str, request_payload: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    if invalid_case == "duplicate-id":
        request_payload["checks"][0]["id"] = "d1"
    elif invalid_case == "missing-role":
        request_payload["checks"].pop(0)
    else:
        request_payload["private caller text"] = "private value"
    request = _write(tmp_path, "request.json", request_payload)
    with pytest.raises(SystemExit, match="invalid request") as error:
        main(["repair-checks", "prompt", request])
    assert "private" not in str(error.value)
    assert capsys.readouterr().out == ""


def test_response_schema_patterns_are_portable_json_schema(
    request_payload: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The selector schema must be usable by a standard JSON Schema validator.

    Red proof: against c176f9bae (pattern ``\\A...\\z`` at
    ``recall/cli_commands/repair_checks.py`` ``_CheckId``) this fails in the loop's
    assertion that no pattern holds ``\\A``, ``\\Z`` or ``\\z``, because ECMA-262 has no
    such escapes and Python ``re`` before 3.14 rejects ``\\z``.
    """
    main(["repair-checks", "prompt", _write(tmp_path, "request.json", request_payload)])
    schema = json.loads(capsys.readouterr().out)["response_schema"]
    patterns = [field["pattern"] for field in schema["$defs"]["_Selection"]["properties"].values()]
    assert len(patterns) == 2
    for pattern in patterns:
        assert not re.search(r"\\[AZz]", pattern), pattern
        # JSON Schema applies a pattern as an unanchored search.
        assert re.search(pattern, "d2") and not re.search(pattern, "d2!")


@pytest.mark.parametrize("check_id", ["d2\n", "-d2", "d" * 65])
def test_rejects_malformed_selection_ids(
    check_id: str, request_payload: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A selection ID is refused unless it matches the whole ID pattern.

    Red proof: with the ``$`` anchor removed from ``_CheckId`` in
    ``recall/cli_commands/repair_checks.py``, the ``"d2\\n"`` and ``"d" * 65`` cases are
    accepted and ``pytest.raises`` fails.
    """
    request = _write(tmp_path, "request.json", request_payload)
    selection = _write(tmp_path, "selection.json", {"diagnostic": check_id, "preservation": "p1"})
    with pytest.raises(SystemExit, match="invalid selection"):
        main(["repair-checks", "resolve", request, selection])
    assert capsys.readouterr().out == ""


def test_rejects_duplicate_json_keys(request_payload: dict, tmp_path: Path) -> None:
    request = _write(tmp_path, "request.json", request_payload)
    selection = tmp_path / "selection.json"
    selection.write_text('{"diagnostic":"d1","diagnostic":"d2","preservation":"p1"}', encoding="utf-8")
    with pytest.raises(SystemExit, match="invalid JSON in selection"):
        main(["repair-checks", "resolve", request, str(selection)])
