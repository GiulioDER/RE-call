"""General knowledge hook, task 1: the index model and its fail-open reader.

Only invented entries appear here (ids `gk-example-...`, host `examplehost`). The conftest autouse
fixture `_confine_general_knowledge` points the corpus and the log at a per-test directory, so no
test reads the operator's real corpus.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from recall_hooks import general_knowledge as gk


def example_entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": "gk-example-tests",
        "kind": "procedure",
        "scope": "global",
        "triggers": [{"command": "pytest"}],
        "symptoms": ["collection error import failure", "fixture setup timeout"],
        "body": "Run the focused file on the build host; the full suite does not fit in memory here.",
    }
    entry.update(overrides)
    return entry


def write_index(home: Path, entries: list[Any], schema: int = 1) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    path = home / "index.json"
    path.write_text(
        json.dumps({"schema": schema, "built_at": "2026-10-22T00:00:00Z", "entries": entries}),
        encoding="utf-8",
    )
    return path


def test_load_index_reads_entries_and_triggers(tmp_path):
    """Invariant: every field of a well-formed record reaches the Entry, triggers in order."""
    write_index(tmp_path, [example_entry(triggers=[{"command": "pytest"}, {"host": "examplehost"}])])
    entries = gk.load_index(tmp_path)
    assert [e.id for e in entries] == ["gk-example-tests"]
    assert entries[0].triggers == (gk.Trigger("command", "pytest"), gk.Trigger("host", "examplehost"))
    assert (entries[0].kind, entries[0].scope) == ("procedure", "global")
    assert entries[0].symptoms == ("collection error import failure", "fixture setup timeout")
    assert entries[0].body.startswith("Run the focused file")


def test_load_index_defaults_to_the_knowledge_home(tmp_path, monkeypatch):
    """Invariant: with no `home` argument the reader reads `knowledge_home()`."""
    monkeypatch.setenv("RECALL_GENERAL_KNOWLEDGE_HOME", str(tmp_path))
    write_index(tmp_path, [example_entry()])
    assert [e.id for e in gk.load_index()] == ["gk-example-tests"]


@pytest.mark.parametrize("content", ["", "{", '{"schema": 2, "entries": []}', '["not", "a", "dict"]'])
def test_load_index_is_empty_for_corrupt_or_foreign_schema(tmp_path, content):
    """Invariant: a corrupt or non-object index is read as empty, never raised.

    The schema 2 case here holds no entries, so it passes even without the schema check; the test
    that carries the red proof for that check is
    `test_load_index_ignores_a_foreign_schema_that_holds_entries`.
    """
    (tmp_path / "index.json").write_text(content, encoding="utf-8")
    assert gk.load_index(tmp_path) == []


def test_load_index_is_empty_when_the_file_is_missing(tmp_path):
    """Invariant: a missing index (the state before any corpus is built) is an empty index."""
    assert gk.load_index(tmp_path / "nowhere") == []


def test_load_index_ignores_a_foreign_schema_that_holds_entries(tmp_path):
    """Invariant: an index whose schema this reader does not know is not guessed at.

    Red proof: node `tests/test_general_knowledge_hook.py::
    test_load_index_ignores_a_foreign_schema_that_holds_entries`. Mutation: delete the
    `document.get("schema") != INDEX_SCHEMA` clause from the guard in
    `recall_hooks.general_knowledge.load_index`. Failure: `assert gk.load_index(tmp_path) == []`
    failed with the entry `gk-example-tests` returned, i.e. a schema 2 file read as schema 1.
    Restored, the same test passes.
    """
    write_index(tmp_path, [example_entry()], schema=2)
    assert gk.load_index(tmp_path) == []


def test_load_index_skips_a_malformed_entry_and_keeps_the_rest(tmp_path):
    """Invariant: one bad record costs that record only."""
    bad_body = example_entry(id="gk-example-no-body")
    del bad_body["body"]
    write_index(
        tmp_path,
        [
            bad_body,
            "not a dict",
            example_entry(id="gk-example-empty-id", **{"body": ""}),
            example_entry(id="gk-example-triggers-not-list", triggers={"command": "pytest"}),
            example_entry(id="gk-example-symptoms-not-list", symptoms="cannot run"),
            example_entry(id="gk-example-kind-not-str", kind=3),
            example_entry(id=""),
            example_entry(id="gk-example-valid"),
        ],
    )
    assert [e.id for e in gk.load_index(tmp_path)] == ["gk-example-valid"]


def test_load_index_is_empty_when_entries_is_not_a_list(tmp_path):
    """Invariant: an `entries` value of the wrong type is an empty index."""
    (tmp_path / "index.json").write_text('{"schema": 1, "entries": {"a": 1}}', encoding="utf-8")
    assert gk.load_index(tmp_path) == []


def test_an_invalid_regex_trigger_is_dropped_not_raised(tmp_path):
    """Invariant: a regex that does not compile loses its trigger, not the entry."""
    write_index(tmp_path, [example_entry(triggers=[{"command_regex": "("}, {"command": "pytest"}])])
    entries = gk.load_index(tmp_path)
    assert [e.id for e in entries] == ["gk-example-tests"]
    assert entries[0].triggers == (gk.Trigger("command", "pytest"),)
    assert gk.compiled("(") is None


def test_a_valid_regex_trigger_is_compiled_once_at_load(tmp_path):
    """Invariant: a `command_regex` value is compiled at load and served from the module cache."""
    pattern = r"^pytest\b.*-n\s*\d+"
    write_index(tmp_path, [example_entry(triggers=[{"command_regex": pattern}])])
    assert gk.load_index(tmp_path)[0].triggers == (gk.Trigger("command_regex", pattern),)
    first = gk.compiled(pattern)
    assert isinstance(first, re.Pattern)
    gk.load_index(tmp_path)
    assert gk.compiled(pattern) is first


@pytest.mark.parametrize(
    "bad",
    [
        {"command": "pytest", "host": "examplehost"},
        {"flavour": "pytest"},
        {"command": ""},
        {"command": 7},
        "pytest",
        {},
    ],
)
def test_a_malformed_trigger_is_dropped_and_the_others_kept(tmp_path, bad):
    """Invariant: multi-key maps, unknown kinds and non-string or empty values drop like a bad regex."""
    write_index(tmp_path, [example_entry(triggers=[bad, {"host": "examplehost"}])])
    entries = gk.load_index(tmp_path)
    assert [e.triggers for e in entries] == [(gk.Trigger("host", "examplehost"),)]


def test_non_string_symptoms_are_dropped(tmp_path):
    """Invariant: symptoms are strings; anything else is dropped without losing the entry."""
    write_index(tmp_path, [example_entry(symptoms=["stale lockfile error", 5, None, "  "])])
    assert gk.load_index(tmp_path)[0].symptoms == ("stale lockfile error",)


def test_knowledge_home_honours_the_environment(monkeypatch, tmp_path):
    """Invariant: `RECALL_GENERAL_KNOWLEDGE_HOME` names the home exactly."""
    monkeypatch.setenv("RECALL_GENERAL_KNOWLEDGE_HOME", str(tmp_path))
    assert gk.knowledge_home() == tmp_path


def test_knowledge_home_defaults_beside_the_project_config(monkeypatch, tmp_path):
    """Invariant: with no override the home is `<project config home>/general-knowledge`."""
    monkeypatch.delenv("RECALL_GENERAL_KNOWLEDGE_HOME")
    monkeypatch.setenv("RECALL_PROJECT_CONFIG_HOME", str(tmp_path))
    assert gk.knowledge_home() == tmp_path / "general-knowledge"


def test_log_path_honours_the_environment_and_defaults_beside_the_project_config(monkeypatch, tmp_path):
    """Invariant: the log is the override if set, else `<project config home>/logs/general-knowledge.jsonl`."""
    monkeypatch.setenv("RECALL_GENERAL_KNOWLEDGE_LOG", str(tmp_path / "x.jsonl"))
    assert gk.log_path() == tmp_path / "x.jsonl"
    monkeypatch.delenv("RECALL_GENERAL_KNOWLEDGE_LOG")
    monkeypatch.setenv("RECALL_PROJECT_CONFIG_HOME", str(tmp_path))
    assert gk.log_path() == tmp_path / "logs" / "general-knowledge.jsonl"


def test_settings_defaults_and_overrides():
    """Invariant: absent block means enabled with min_score 0.0; the block overrides both."""
    assert gk.settings({}) == {"enabled": True, "min_score": 0.0}
    assert gk.settings() == {"enabled": True, "min_score": 0.0}
    assert gk.settings({"general_knowledge": {"enabled": False, "min_score": 2.5}}) == {
        "enabled": False,
        "min_score": 2.5,
    }


def test_settings_ignores_a_block_of_the_wrong_shape():
    """Invariant: a malformed block or min_score falls back to defaults instead of raising."""
    assert gk.settings({"general_knowledge": "off"}) == {"enabled": True, "min_score": 0.0}
    assert gk.settings({"general_knowledge": {"min_score": "high"}}) == {"enabled": True, "min_score": 0.0}


def test_load_settings_reads_the_project_config_home(monkeypatch, tmp_path):
    """Invariant: `load_settings` reads `recall-hook.json` under RECALL_PROJECT_CONFIG_HOME."""
    monkeypatch.setenv("RECALL_PROJECT_CONFIG_HOME", str(tmp_path))
    (tmp_path / "recall-hook.json").write_text(
        json.dumps({"general_knowledge": {"enabled": False, "min_score": 1.5}}), encoding="utf-8"
    )
    assert gk.load_settings() == {"enabled": False, "min_score": 1.5}


def test_load_settings_does_not_follow_the_hook_config_home(monkeypatch, tmp_path):
    """Invariant: the Codex-redirectable hook config home (`RECALL_HOOK_CONFIG_HOME`) is NOT read."""
    project, hook = tmp_path / "project", tmp_path / "hook"
    project.mkdir()
    hook.mkdir()
    monkeypatch.setenv("RECALL_PROJECT_CONFIG_HOME", str(project))
    monkeypatch.setenv("RECALL_HOOK_CONFIG_HOME", str(hook))
    (hook / "recall-hook.json").write_text(
        json.dumps({"general_knowledge": {"enabled": False}}), encoding="utf-8"
    )
    assert gk.load_settings() == {"enabled": True, "min_score": 0.0}


@pytest.mark.parametrize("content", [None, "{", "[1, 2]"])
def test_load_settings_is_default_for_missing_corrupt_or_non_dict(monkeypatch, tmp_path, content):
    """Invariant: a missing, corrupt or non-object hook config gives the defaults."""
    monkeypatch.setenv("RECALL_PROJECT_CONFIG_HOME", str(tmp_path))
    if content is not None:
        (tmp_path / "recall-hook.json").write_text(content, encoding="utf-8")
    assert gk.load_settings() == {"enabled": True, "min_score": 0.0}
