"""General knowledge hook, tasks 1 to 6: the index model, its fail-open reader, project detection, trigger matching, the
session dedupe, the event log, failure signatures, failure ranking, the PreToolUse integration, the
PostToolUseFailure integration and the Codex adapter.

Only invented entries appear here (ids `gk-example-...`, host `examplehost`). The conftest autouse
fixture `_confine_general_knowledge` points the corpus and the log at a per-test directory, so no
test reads the operator's real corpus.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

import recall_hooks
from recall_hooks import general_knowledge as gk
from recall_hooks import write_time


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
    """Invariant: every field of a well-formed record reaches the Entry, triggers in order.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_load_index_reads_entries_and_triggers`.
    Mutation: `_entry` returns its triggers reversed (`tuple(...)[::-1]`). Symbol: `_entry`.
    Seen: `assert (Trigger(kind=..., value='pytest'), ...) == (Trigger(...), Trigger(kind=..., value='examplehost'))`.
    """
    write_index(tmp_path, [example_entry(triggers=[{"command": "pytest"}, {"host": "examplehost"}])])
    entries = gk.load_index(tmp_path)
    assert [e.id for e in entries] == ["gk-example-tests"]
    assert entries[0].triggers == (gk.Trigger("command", "pytest"), gk.Trigger("host", "examplehost"))
    assert (entries[0].kind, entries[0].scope) == ("procedure", "global")
    assert entries[0].symptoms == ("collection error import failure", "fixture setup timeout")
    assert entries[0].body.startswith("Run the focused file")


def test_load_index_defaults_to_the_knowledge_home(tmp_path, monkeypatch):
    """Invariant: with no `home` argument the reader reads `knowledge_home()`.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_load_index_defaults_to_the_knowledge_home`.
    Mutation: `load_index` looks in a nonexistent path instead of `knowledge_home()` when `home` is None.
    Symbol: `load_index`. Seen: `assert [] == ['gk-example-tests']`.
    """
    monkeypatch.setenv("RECALL_GENERAL_KNOWLEDGE_HOME", str(tmp_path))
    write_index(tmp_path, [example_entry()])
    assert [e.id for e in gk.load_index()] == ["gk-example-tests"]


@pytest.mark.parametrize("content", ["", "{", '{"schema": 2, "entries": []}', '["not", "a", "dict"]'])
def test_load_index_is_empty_for_corrupt_or_foreign_schema(tmp_path, content):
    """Invariant: a corrupt or non-object index is read as empty, never raised.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_load_index_is_empty_for_corrupt_or_foreign_schema`.
    Mutation: both `except` bodies in `load_index` re-raise instead of returning `[]`. Symbol: `load_index`.
    Seen: JSONDecodeError escaped for the `''` and `'{'` cases. The `["not", "a", "dict"]` and schema 2
    empty cases stayed green under that mutation (the dict guard returns before anything can raise; the
    schema 2 file holds no entries), so they are controls; the schema check is proved by the next test but one.
    """
    (tmp_path / "index.json").write_text(content, encoding="utf-8")
    assert gk.load_index(tmp_path) == []


def test_load_index_is_empty_when_the_file_is_missing(tmp_path):
    """Invariant: a missing index (the state before any corpus is built) is an empty index.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_load_index_is_empty_when_the_file_is_missing`.
    Mutation: same as the previous test (both `except` bodies in `load_index` re-raise).
    Seen: `FileNotFoundError: [Errno 2] No such file or directory: '...nowhere\\index.json'`.
    """
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
    """Invariant: one bad record costs that record only.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_load_index_skips_a_malformed_entry_and_keeps_the_rest`.
    Mutation A: drop the `not isinstance(body, str) or not body` clause in `_entry`. Mutation B: drop the
    `isinstance(raw_triggers, list)` clause in `_entry`. Symbol: `_entry`.
    Seen, both: `assert ['gk-example-...xample-valid'] == ['gk-example-valid']` (a bad record was kept).
    """
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
    """Invariant: an `entries` value of the wrong type is an empty index.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_load_index_is_empty_when_entries_is_not_a_list`.
    Deliberate non-behavioural control, red proof does not apply: removing the `isinstance(records, list)`
    guard in `load_index` leaves this green, because iterating a dict yields strings that `_entry` skips.
    The guard documents intent; the behaviour is held by `_entry`'s skip.
    """
    (tmp_path / "index.json").write_text('{"schema": 1, "entries": {"a": 1}}', encoding="utf-8")
    assert gk.load_index(tmp_path) == []


def test_an_invalid_regex_trigger_is_dropped_not_raised(tmp_path):
    """Invariant: a regex that does not compile loses its trigger, not the entry.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_an_invalid_regex_trigger_is_dropped_not_raised`.
    Mutation: `_trigger` catches `KeyError` instead of `re.error`. Symbol: `_trigger`.
    Seen: `assert [] == ['gk-example-tests']` (the re.error escaped and emptied the whole index).
    """
    write_index(tmp_path, [example_entry(triggers=[{"command_regex": "("}, {"command": "pytest"}])])
    entries = gk.load_index(tmp_path)
    assert [e.id for e in entries] == ["gk-example-tests"]
    assert entries[0].triggers == (gk.Trigger("command", "pytest"),)
    assert gk.compiled("(") is None


@pytest.mark.parametrize("pattern", ["a{4294967296}", "(?:" * 5000 + ")" * 5000])
def test_a_regex_the_engine_cannot_compile_costs_one_trigger_not_the_index(tmp_path, pattern):
    """Invariant: OverflowError and RecursionError from `re.compile` drop the trigger only.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_regex_the_engine_cannot_compile_costs_one_trigger_not_the_index`.
    Mutation: `_trigger` catches only `re.error` (drop `OverflowError, RecursionError`). Symbol: `_trigger`.
    Seen, both params: `assert [] == ['gk-example-tests', 'gk-example-second']` (index emptied).
    """
    write_index(
        tmp_path,
        [
            example_entry(triggers=[{"command_regex": pattern}, {"command": "pytest"}]),
            example_entry(id="gk-example-second"),
        ],
    )
    entries = gk.load_index(tmp_path)
    assert [e.id for e in entries] == ["gk-example-tests", "gk-example-second"]
    assert entries[0].triggers == (gk.Trigger("command", "pytest"),)


def test_a_regex_trigger_is_compiled_once_across_loads(tmp_path, monkeypatch):
    """Invariant: a `command_regex` value is compiled by the module once, then served from its cache.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_regex_trigger_is_compiled_once_across_loads`.
    Mutation: `_trigger` drops the `value not in _COMPILED` condition, so it compiles every time.
    Symbol: `_trigger`. Seen: `assert 2 == 1` on `calls.count(pattern)`. (Counting `re.compile` calls
    is the point: the identity of the returned object proves nothing, since `re` has its own cache.)
    """
    pattern = r"^gk-example-unique-\d+$"
    monkeypatch.setattr(gk, "_COMPILED", {})
    real_compile, calls = re.compile, []

    def counting(value, *args, **kwargs):
        calls.append(value)
        return real_compile(value, *args, **kwargs)

    monkeypatch.setattr(gk.re, "compile", counting)
    write_index(tmp_path, [example_entry(triggers=[{"command_regex": pattern}])])
    first_load = gk.load_index(tmp_path)
    gk.load_index(tmp_path)
    assert first_load[0].triggers == (gk.Trigger("command_regex", pattern),)
    assert calls.count(pattern) == 1
    assert isinstance(gk.compiled(pattern), re.Pattern)


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
    """Invariant: multi-key maps, unknown kinds and non-string or empty values drop like a bad regex.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_malformed_trigger_is_dropped_and_the_others_kept`.
    Mutations, symbol `_trigger`. A: drop the `len(raw) != 1` clause; seen `assert [] == [(Trigger(kind=...,
    value='examplehost'),)]` for the two-key and `{}` params (the unpack error emptied the index).
    B: drop the `kind not in TRIGGER_KINDS` clause; seen the unknown-kind param keep a trigger.
    C: drop the `isinstance(value, str)` and `not value` clauses; seen the empty-value and int-value
    params keep a trigger. The bare-string param is held by the `isinstance(raw, dict)` guard and was
    not mutated.
    """
    write_index(tmp_path, [example_entry(triggers=[bad, {"host": "examplehost"}])])
    entries = gk.load_index(tmp_path)
    assert [e.triggers for e in entries] == [(gk.Trigger("host", "examplehost"),)]


def test_non_string_symptoms_are_dropped(tmp_path):
    """Invariant: symptoms are strings; anything else is dropped without losing the entry.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_non_string_symptoms_are_dropped`.
    Mutation: `_entry` keeps `tuple(raw_symptoms)` unfiltered. Symbol: `_entry`.
    Seen: `assert ('stale lockf...', 5, None, '  ') == ('stale lockfile error',)`.
    """
    write_index(tmp_path, [example_entry(symptoms=["stale lockfile error", 5, None, "  "])])
    assert gk.load_index(tmp_path)[0].symptoms == ("stale lockfile error",)


def test_knowledge_home_honours_the_environment(monkeypatch, tmp_path):
    """Invariant: `RECALL_GENERAL_KNOWLEDGE_HOME` names the home exactly.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_knowledge_home_honours_the_environment`.
    Mutation: `knowledge_home` ignores `RECALL_GENERAL_KNOWLEDGE_HOME`. Symbol: `knowledge_home`.
    Seen: `assert WindowsPath('.../fake-home.../.claude/general-knowledge') == WindowsPath(<tmp_path>)`.
    """
    monkeypatch.setenv("RECALL_GENERAL_KNOWLEDGE_HOME", str(tmp_path))
    assert gk.knowledge_home() == tmp_path


def test_knowledge_home_defaults_beside_the_project_config(monkeypatch, tmp_path):
    """Invariant: with no override the home is `<project config home>/general-knowledge`.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_knowledge_home_defaults_beside_the_project_config`.
    Mutation: the default directory name `general-knowledge` becomes `gk`. Symbol: `knowledge_home`.
    Seen: `assert WindowsPath('.../gk') == WindowsPath('...') / 'general-knowledge'`.
    """
    monkeypatch.delenv("RECALL_GENERAL_KNOWLEDGE_HOME")
    monkeypatch.setenv("RECALL_PROJECT_CONFIG_HOME", str(tmp_path))
    assert gk.knowledge_home() == tmp_path / "general-knowledge"


def test_log_path_honours_the_environment_and_defaults_beside_the_project_config(monkeypatch, tmp_path):
    """Invariant: the log is the override if set, else `<project config home>/logs/general-knowledge.jsonl`.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_log_path_honours_the_environment_and_defaults_beside_the_project_config`.
    Mutation A: `log_path` ignores `RECALL_GENERAL_KNOWLEDGE_LOG`; seen the first assertion fail with
    `.../.claude/logs/general-knowledge.jsonl` instead of the override. Mutation B: the default file name
    becomes `gk.jsonl`; seen the second assertion fail on `.../logs/gk.jsonl`. Symbol: `log_path`.
    """
    monkeypatch.setenv("RECALL_GENERAL_KNOWLEDGE_LOG", str(tmp_path / "x.jsonl"))
    assert gk.log_path() == tmp_path / "x.jsonl"
    monkeypatch.delenv("RECALL_GENERAL_KNOWLEDGE_LOG")
    monkeypatch.setenv("RECALL_PROJECT_CONFIG_HOME", str(tmp_path))
    assert gk.log_path() == tmp_path / "logs" / "general-knowledge.jsonl"


def test_settings_defaults_and_overrides():
    """Invariant: absent block means enabled with min_score 0.0; the block overrides both.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_settings_defaults_and_overrides`.
    Mutations, symbol `settings`. A: default for `enabled` becomes False; B: default for `min_score` becomes
    1.0. Seen: `assert {'enabled': False, ...} == {'enabled': True, ...}` and
    `assert {..., 'min_score': 1.0} == {..., 'min_score': 0.0}`.
    """
    assert gk.settings({}) == {"enabled": True, "min_score": 0.0}
    assert gk.settings() == {"enabled": True, "min_score": 0.0}
    assert gk.settings({"general_knowledge": {"enabled": False, "min_score": 2.5}}) == {
        "enabled": False,
        "min_score": 2.5,
    }


@pytest.mark.parametrize("block", ["off", "false", [], [1], None, 0, 1, True])
def test_settings_a_present_block_that_is_not_an_object_disables(block):
    """Invariant: a PRESENT `general_knowledge` block of the wrong shape fails closed (ruling R9).

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_settings_a_present_block_that_is_not_an_object_disables`.
    Mutation: the previous behaviour, `settings` normalises a non-dict block to `{}` (defaults, so enabled).
    Symbol: `settings`. Seen: every param fails, `assert {'enabled': True, 'min_score': 0.0} == {'enabled': False, 'min_score': 0.0}`.
    """
    assert gk.settings({"general_knowledge": block}) == {"enabled": False, "min_score": 0.0}


def test_settings_an_absent_block_is_still_the_enabled_default():
    """Invariant: only a PRESENT malformed block disables; an absent one (or a config without it) is the default.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_settings_an_absent_block_is_still_the_enabled_default`.
    Mutation: `settings` treats a missing key like a malformed block (`block = config.get(...)`, then
    `if not isinstance(block, dict): return disabled`). Symbol: `settings`.
    Seen: `assert {'enabled': False, ...} == {'enabled': True, ...}`.
    """
    assert gk.settings({"other": 1}) == {"enabled": True, "min_score": 0.0}
    assert gk.settings({"general_knowledge": {}}) == {"enabled": True, "min_score": 0.0}


@pytest.mark.parametrize("score", [10**400, -(10**400), 10**309])
def test_settings_never_raises_on_an_int_too_large_for_a_float(score):
    """Invariant: `settings` never raises; an int no float can hold falls back to the default min_score.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_settings_never_raises_on_an_int_too_large_for_a_float`.
    Mutation: the previous code, `not math.isfinite(min_score)` evaluated on the raw int with no
    `float()` guard. Symbol: `settings`. Seen: `OverflowError: int too large to convert to float`.
    """
    assert gk.settings({"general_knowledge": {"min_score": score}}) == {"enabled": True, "min_score": 0.0}


@pytest.mark.parametrize("enabled", ["false", "no", "true", 0, 1, None, []])
def test_settings_a_present_enabled_that_is_not_a_bool_disables(enabled):
    """Invariant: `enabled` fails closed; a present non-bool (even a quoted "true") turns injection OFF.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_settings_a_present_enabled_that_is_not_a_bool_disables`.
    Mutation A: the previous behaviour, the non-bool branch in `settings` sets `enabled = True`
    (fall back to the default). Seen: every param fails, `assert True is False`. Mutation B: coerce
    with `bool(enabled)`. Seen: the truthy params (`'false'`, `'no'`, `'true'`, `1`) fail with
    `assert True is False`. Symbol: `settings`.
    """
    assert gk.settings({"general_knowledge": {"enabled": enabled}})["enabled"] is False


def test_settings_enabled_absent_or_a_real_bool_is_taken_as_is():
    """Invariant: absent means enabled; a real bool is honoured exactly.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_settings_enabled_absent_or_a_real_bool_is_taken_as_is`.
    Mutation: the non-bool branch in `settings` also applies when the key is absent (default
    `block.get("enabled", False)`). Seen: `assert False is True` on the absent case. Symbol: `settings`.
    """
    assert gk.settings({"general_knowledge": {"min_score": 1.0}})["enabled"] is True
    assert gk.settings({"general_knowledge": {"enabled": True}})["enabled"] is True
    assert gk.settings({"general_knowledge": {"enabled": False}})["enabled"] is False


@pytest.mark.parametrize("score", ["high", "2.5", True, False, None, float("nan"), float("inf"), -float("inf"), []])
def test_settings_min_score_must_be_a_finite_number(score):
    """Invariant: only a finite int or float (not a bool) sets `min_score`; else the default 0.0.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_settings_min_score_must_be_a_finite_number`.
    Mutations, symbol `settings`. A: coerce with `float(min_score)`; seen `'2.5'`, `True`, `nan`, `inf`, `-inf`
    fail (min_score no longer the default 0.0). B: drop the `math.isfinite` clause; seen `nan`, `inf`,
    `-inf` fail. C: drop the `isinstance(min_score, bool)` clause; seen `True` fail. The `False` param
    equals 0.0 either way and cannot go red.
    """
    assert gk.settings({"general_knowledge": {"min_score": score}})["min_score"] == 0.0


def test_settings_min_score_accepts_an_int_and_returns_a_float():
    """Invariant: a finite int is a valid min_score and is returned as a float.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_settings_min_score_accepts_an_int_and_returns_a_float`.
    Mutation: `settings` returns `min_score` without `float(...)`. Symbol: `settings`.
    Seen: the `isinstance(value, float)` half of the assertion failed (value was the int 2).
    """
    value = gk.settings({"general_knowledge": {"min_score": 2}})["min_score"]
    assert value == 2.0 and isinstance(value, float)


def test_load_settings_reads_the_project_config_home(monkeypatch, tmp_path):
    """Invariant: `load_settings` reads `recall-hook.json` under RECALL_PROJECT_CONFIG_HOME.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_load_settings_reads_the_project_config_home`.
    Mutations, symbol `load_settings`. A: read `claude_config_home()` instead of the project home; B: ignore
    the file (`return settings({})`). Seen both: `assert {'enabled': True, 'min_score': 0.0} ==
    {'enabled': False, 'min_score': 1.5}`.
    """
    monkeypatch.setenv("RECALL_PROJECT_CONFIG_HOME", str(tmp_path))
    (tmp_path / "recall-hook.json").write_text(
        json.dumps({"general_knowledge": {"enabled": False, "min_score": 1.5}}), encoding="utf-8"
    )
    assert gk.load_settings() == {"enabled": False, "min_score": 1.5}


def test_load_settings_does_not_follow_the_hook_config_home(monkeypatch, tmp_path):
    """Invariant: the Codex-redirectable hook config home (`RECALL_HOOK_CONFIG_HOME`) is NOT read.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_load_settings_does_not_follow_the_hook_config_home`.
    Mutation: `load_settings` reads `claude_config_home()` (what `load_config` uses). Symbol: `load_settings`.
    Seen: `assert {'enabled': False, ...} == {'enabled': True, ...}` (it followed RECALL_HOOK_CONFIG_HOME).
    """
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
    """Invariant: a missing, corrupt or non-object hook config gives the defaults.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_load_settings_is_default_for_missing_corrupt_or_non_dict`.
    Mutations, symbol `load_settings`. A: catch only `OSError`; seen `json.decoder.JSONDecodeError` for the
    `'{'` param. B: pass the parsed document to `settings` without the dict check; seen
    `AttributeError: 'list' object has no attribute 'get'` for the `'[1, 2]'` param. The missing-file
    param was not mutated.
    """
    monkeypatch.setenv("RECALL_PROJECT_CONFIG_HOME", str(tmp_path))
    if content is not None:
        (tmp_path / "recall-hook.json").write_text(content, encoding="utf-8")
    assert gk.load_settings() == {"enabled": True, "min_score": 0.0}


# ---------------------------------------------------------------------------------------------
# Task 2: project detection and trigger matching.
# ---------------------------------------------------------------------------------------------


def loaded(tmp_path: Path, *entries: dict[str, Any]) -> list[gk.Entry]:
    """Entries as `load_index` builds them, so `command_regex` values are in the compiled cache."""
    write_index(tmp_path, list(entries))
    return gk.load_index(tmp_path)


@pytest.mark.parametrize(
    "command,expected",
    [
        ("FOO=1 sudo pytest -q && git status", ["pytest", "git"]),
        ('& "C:\\Program Files\\Tool\\tool.exe" -a; Get-ChildItem C:\\', ["tool", "get-childitem"]),
        ("cat x | grep y", ["cat", "grep"]),
        ("make a || make b", ["make", "make"]),
        ("a\nb\r\nc", ["a", "b", "c"]),
        ("env FOO=1 BAR=2 time nohup Python -m pytest", ["python"]),
        ("/usr/local/bin/PyTest -q", ["pytest"]),
        ("C:/tools/Tool.EXE --x", ["tool"]),
        ('"C:\\Program Files\\Git\\bin\\git.exe" status', ["git"]),
        ("'my tool' --x", ["my tool"]),
        ('echo "a;b|c && d" | grep x', ["echo", "grep"]),
        ("ls 2>&1 | grep a", ["ls", "grep"]),
        ("x &> out; y", ["x", "y"]),
        ("a & b", ["a", "b"]),
        ("$env:X=1; pytest", ["pytest"]),
        ("", []),
        ("   ", []),
        (";; && || |", []),
    ],
)
def test_command_programs(command, expected):
    """Invariant: the program of every simple command, normalised, for POSIX and Windows shapes alike.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_command_programs`. Symbol group:
    `_SCAN`, `_WRAPPERS`, `_ASSIGNMENT`, `_program`, `_unquote`. Mutations, each failing on the
    params named:
    A: `_PATH_SEPARATOR = re.compile(r"/")` (basename split on `/` only), seen on the two Windows path params: `assert ['c:\\program...et-childitem'] == ['tool', 'get-childitem']`.
    B: `_unquote` returns its argument unchanged (quotes kept), seen: `assert ['tool.exe"', 'get-childitem'] == ['tool', 'get-childitem']` and `assert ['git.exe"'] == ['git']`.
    C: `_WRAPPERS` emptied, seen: `assert ['sudo', 'git'] == ['pytest', 'git']` and `assert ['env'] == ['python']`.
    """
    assert gk.command_programs(command) == expected


def test_segments_are_split_on_every_separator_but_not_inside_quotes_or_redirections():
    """Invariant: `&&`, `||`, `;`, `|`, a lone `&` and newlines end a segment; a quoted separator and
    the `&` of `2>&1` or `&>` do not.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_segments_are_split_on_every_separator_but_not_inside_quotes_or_redirections`.
    Symbol: `_SCAN`. Mutations: A: the `sep` group loses `&&|\\|\\||&(?!>)` (`assert ['a', 'd', 'g'] == ['a', 'b', 'c', ...]`). B: the `sep` group
    loses `;` (`assert ['a', 'b', 'c', 'e', 'f', 'g'] == [...]`). C: the word group loses the `[<>]+&?` redirection atom, so the `&` of `2>&1` is
    a separator (`assert ['make', '1', 'tee'] == ['make', 'tee']`).
    """
    assert gk.command_programs("a && b || c ; d | e & f\ng") == list("abcdefg")
    assert gk.command_programs('echo "x && y" ; echo \'p | q\'') == ["echo", "echo"]
    assert gk.command_programs("make 2>&1 | tee log") == ["make", "tee"]
    assert gk.command_programs("make &> log; true") == ["make", "true"]


def test_a_quoted_token_with_spaces_is_one_token_and_quotes_are_stripped():
    """Invariant: `parse_command` keeps a quoted token whole and drops the quotes, so a host alias or a
    program path with spaces is matchable.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_quoted_token_with_spaces_is_one_token_and_quotes_are_stripped`.
    Mutation: `_unquote` returns its argument unchanged. Symbol: `_unquote`. Seen: `assert ('-o', '"Prox...ple'", '"x"y') == ('-o', 'Proxy...a mple', 'xy')`.
    """
    (segment,) = gk.parse_command('ssh -o "ProxyJump=a b" \'exa mple\' "x"y')
    assert segment.program == "ssh"
    assert segment.args == ("-o", "ProxyJump=a b", "exa mple", "xy")


HOSTILE = [
    '"',
    "'",
    'pytest "unterminated',
    "pytest 'unterminated",
    "\"a' b",
    "|",
    "&",
    "&&&&||||;;;;",
    ">&",
    "2>&",
    "\\",
    "C:\\",
    "\\\\server\\share\\x.exe --a",
    "& ",
    "=",
    "===",
    "\x00",
    "pytest\x00 -q",
    "\u00a0pytest\u2003-q",
    "'' '' \"\"",
    "x" * 100_000,
    "a | " * 25_000,
    "'" * 50_001,
    "ssh " + "@" * 1000,
]


@pytest.mark.parametrize("command", HOSTILE, ids=lambda c: f"{len(c)}:{c[:12]!r}")
def test_hostile_strings_never_raise(tmp_path, command):
    """Invariant: every public function of the matching layer returns a value for any string.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_hostile_strings_never_raise`.
    Mutation: `_program` indexes `_PATH_SEPARATOR.split(token)[1]` instead of `[-1]`. Symbol: `_program`.
    Seen: `IndexError: list index out of range` raised from `_program`, on the params that hold a word
    without a path separator.
    """
    entries = loaded(
        tmp_path,
        example_entry(id="gk-example-a", triggers=[{"command": "pytest"}, {"host": "examplehost"}]),
        example_entry(id="gk-example-b", triggers=[{"command_regex": r"(^|\s)make test\b"}]),
    )
    assert isinstance(gk.command_programs(command), list)
    assert isinstance(gk.parse_command(command), list)
    assert isinstance(gk.match_triggers(entries, command, "alpha"), list)
    assert isinstance(gk.repository_name(command), (str, type(None)))


def test_unbalanced_quotes_take_the_rest_of_the_command_as_one_token():
    """Invariant: an unterminated quote is not an error; it quotes to the end of the command.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_unbalanced_quotes_take_the_rest_of_the_command_as_one_token`.
    Mutation: the word group's `"[^"]*"?` becomes `"[^"]*"` (closing quote required). Symbol: `_SCAN`.
    Seen: `assert ['pytest', 'git'] == ['pytest']`.
    """
    assert gk.command_programs('pytest "-q && git status') == ["pytest"]
    assert gk.command_programs('"pytest -q && git status') == ["pytest -q && git status"]
    assert gk.parse_command("ssh 'examplehost")[0].args == ("examplehost",)


def test_command_trigger_matches_program_not_substring(tmp_path):
    """Invariant: a `command` trigger equals a segment's program; it is not a substring of the text.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_command_trigger_matches_program_not_substring`.
    Mutation (the brief's): in `matched_trigger` the `command` branch tests
    `trigger.value.lower() in command.lower()` instead of comparing programs. Symbol: `matched_trigger`.
    Seen: `assert not True` on the `python -m pytest` line.
    """
    entries = loaded(tmp_path, example_entry(triggers=[{"command": "pytest"}]))
    matches = lambda command: bool(gk.match_triggers(entries, command, None))  # noqa: E731
    assert matches("pytest -q")
    assert matches("cd x && PYTEST -q")
    assert matches("C:\\Python\\Scripts\\pytest.exe tests")
    assert not matches("python -m pytest")
    assert not matches("echo pytest")
    assert not matches("pytester run")


def test_host_trigger_matches_ssh_alias_and_scp_prefix(tmp_path):
    """Invariant: a `host` trigger fires when an ssh, scp, sftp or rsync segment names the alias as a
    whole token, as `alias:<path>`, or after `user@`; never in other programs and never as a substring.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_host_trigger_matches_ssh_alias_and_scp_prefix`.
    Mutations, symbol `matched_trigger` / `_names_host`. A: the `host` branch drops the
    `s.program in _HOST_PROGRAMS` test (`assert not True` on the `echo examplehost` line). B: `_names_host` tests `wanted in arg` (substring;
    `assert not True` on the `ssh examplehost2 ls` line). C: `_names_host` drops the `startswith(wanted + ":")` clause (`assert False` on the `scp f examplehost:/tmp` line).
    """
    entries = loaded(tmp_path, example_entry(triggers=[{"host": "examplehost"}]))
    matches = lambda command: bool(gk.match_triggers(entries, command, None))  # noqa: E731
    assert matches("ssh examplehost 'ls'")
    assert matches("ssh -p 22 deploy@examplehost uptime")
    assert matches("scp f examplehost:/tmp")
    assert matches("rsync -a src/ deploy@examplehost:/srv/")
    assert matches('sftp "examplehost"')
    assert not matches("echo examplehost")
    assert not matches("ssh otherhost 'echo examplehost-backup'")
    assert not matches("ssh examplehost2 ls")
    assert not matches("scp f myexamplehost:/tmp")
    assert not matches("curl http://examplehost:8080/")


def test_regex_trigger(tmp_path):
    """Invariant: a `command_regex` trigger is a `re.search` over the whole command text.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_regex_trigger`.
    Mutations, symbol `matched_trigger`. A: `pattern.fullmatch(command)` instead of `search` (`assert False` on `matches('cd build && make test -j4')`).
    B: `pattern.search` is run on the first segment's program only (`assert False` on the first `matches(...)` line).
    """
    entries = loaded(tmp_path, example_entry(triggers=[{"command_regex": r"(^|\s)make test\b"}]))
    matches = lambda command: bool(gk.match_triggers(entries, command, None))  # noqa: E731
    assert matches("make test")
    assert matches("cd build && make test -j4")
    assert not matches("make tests")
    assert not matches("cmake test")


def test_a_regex_trigger_uses_the_load_time_cache_and_never_compiles(tmp_path, monkeypatch):
    """Invariant: matching never calls `re.compile`; the pattern comes from the load-time cache.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_regex_trigger_uses_the_load_time_cache_and_never_compiles`.
    Mutation: `matched_trigger` evaluates `re.compile(trigger.value)` instead of `compiled(trigger.value)`.
    Symbol: `matched_trigger`. Seen: `AssertionError: re.compile called while matching`.
    """
    entries = loaded(tmp_path, example_entry(id="gk-example-cache", triggers=[{"command_regex": r"zzz-\d+"}]))

    def refuse(*args, **kwargs):
        raise AssertionError("re.compile called while matching")

    monkeypatch.setattr(re, "compile", refuse)
    assert [e.id for e in gk.match_triggers(entries, "run zzz-12", None)] == ["gk-example-cache"]


def test_an_entry_whose_regex_is_not_cached_does_not_match_and_does_not_raise():
    """Invariant: a hand-built entry with an uncached regex is a non-match, not an exception.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_an_entry_whose_regex_is_not_cached_does_not_match_and_does_not_raise`.
    Mutation: `matched_trigger` reads `_COMPILED[trigger.value]` instead of `compiled(trigger.value)`.
    Symbol: `matched_trigger`. Seen: `KeyError: 'never-cached-\\d'`.
    """
    entry = gk.Entry("gk-example-x", "procedure", "global", (gk.Trigger("command_regex", "never-cached-\\d"),), (), "b")
    assert gk.match_triggers([entry], "never-cached-1", None) == []


def test_project_scope_filters_entries(tmp_path):
    """Invariant: `global` is always in scope; `project:<name>` only for exactly that project.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_project_scope_filters_entries`.
    Mutations, symbol `in_scope`. A: always `True`; seen `assert ['gk-example-global', 'gk-example-alpha'] ==
    ['gk-example-global']` for project `beta`. B: the `startswith(prefix)` test is dropped, so the check is
    `entry.scope[len(prefix):] == project`; seen `assert not True` on the `projekt:alpha` entry (same length
    prefix, different word).
    """
    entries = loaded(
        tmp_path,
        example_entry(id="gk-example-global", scope="global"),
        example_entry(id="gk-example-alpha", scope="project:alpha"),
    )
    ids = lambda project: [e.id for e in gk.match_triggers(entries, "pytest", project)]  # noqa: E731
    assert ids("alpha") == ["gk-example-global", "gk-example-alpha"]
    assert ids("beta") == ["gk-example-global"]
    assert ids(None) == ["gk-example-global"]
    assert ids("Alpha") == ["gk-example-global"]
    odd = gk.Entry("gk-example-odd", "procedure", "projekt:alpha", (gk.Trigger("command", "pytest"),), (), "b")
    assert not gk.in_scope(odd, "alpha")
    assert not gk.in_scope(gk.Entry("i", "k", "project:", (), (), "b"), None)


def test_match_triggers_keeps_input_order_and_returns_no_duplicates(tmp_path):
    """Invariant: input order is kept, and an id that appears twice in the input is returned once.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_match_triggers_keeps_input_order_and_returns_no_duplicates`.
    Mutations, symbol `match_triggers`. A: sort matches by id (`assert ['gk-example-...gk-example-z'] == ['gk-example-...gk-example-m']`). B: drop the `seen` check
    (the same assertion (the repeated entry appears twice)).
    """
    entries = loaded(
        tmp_path,
        example_entry(id="gk-example-z", triggers=[{"command": "pytest"}]),
        example_entry(id="gk-example-m", triggers=[{"command": "git"}, {"command": "pytest"}]),
        example_entry(id="gk-example-a", triggers=[{"command": "nothing"}]),
    )
    both = entries + [entries[0]]
    assert [e.id for e in gk.match_triggers(both, "git status && pytest", None)] == ["gk-example-z", "gk-example-m"]
    assert gk.match_triggers([], "pytest", None) == []
    assert gk.match_triggers(entries, "", None) == []
    assert gk.match_triggers(entries, "   \n", None) == []


def test_matched_trigger_reports_the_first_trigger_that_matched(tmp_path):
    """Invariant: `matched_trigger` returns the FIRST matching trigger of the entry, `None` when none
    matches or the scope does not fit. The event log reads the trigger from here.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_matched_trigger_reports_the_first_trigger_that_matched`.
    Mutations, symbol `matched_trigger`. A: iterate `reversed(entry.triggers)` (`assert Trigger(kind='command_regex', value='pyt') == Trigger(kind='host', value='examplehost')`). B: drop the
    `in_scope` guard (`assert Trigger(kind='command', value='pytest') is None`).
    """
    (entry,) = loaded(
        tmp_path,
        example_entry(
            scope="project:alpha",
            triggers=[{"host": "examplehost"}, {"command": "pytest"}, {"command_regex": r"pyt"}],
        ),
    )
    command = "ssh examplehost pytest"
    segments = gk.parse_command(command)
    assert gk.matched_trigger(entry, command, segments, "alpha") == gk.Trigger("host", "examplehost")
    command = "pytest -q"
    assert gk.matched_trigger(entry, command, gk.parse_command(command), "alpha") == gk.Trigger("command", "pytest")
    command = "python -m pyt"
    assert gk.matched_trigger(entry, command, gk.parse_command(command), "alpha") == gk.Trigger("command_regex", "pyt")
    assert gk.matched_trigger(entry, "pytest", gk.parse_command("pytest"), "beta") is None
    assert gk.matched_trigger(entry, "ls", gk.parse_command("ls"), "alpha") is None


def test_the_command_is_parsed_once_per_match_call(tmp_path, monkeypatch):
    """Invariant: `match_triggers` parses the command once, not once per entry.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_the_command_is_parsed_once_per_match_call`.
    Mutation: `matched_trigger` is passed nothing and `match_triggers` calls `parse_command(command)`
    inside its loop. Symbol: `match_triggers`. Seen: `assert 6 == 1`.
    """
    entries = loaded(tmp_path, *[example_entry(id=f"gk-example-{i}", triggers=[{"command": "git"}]) for i in range(5)])
    calls: list[str] = []
    real = gk.parse_command
    monkeypatch.setattr(gk, "parse_command", lambda c: calls.append(c) or real(c))
    assert len(gk.match_triggers(entries, "git status", None)) == 5
    assert len(calls) == 1


def _git_dir(root: Path, name: str) -> Path:
    repo = root / name
    (repo / ".git").mkdir(parents=True)
    return repo


def _git_file(root: Path, name: str, content: str | bytes) -> Path:
    checkout = root / name
    checkout.mkdir(parents=True)
    data = content.encode("utf-8") if isinstance(content, str) else content
    (checkout / ".git").write_bytes(data)
    return checkout


def test_repository_name_for_a_checkout_found_from_a_subdirectory(tmp_path):
    """Invariant: a `.git` directory names the repository after its parent, found from any depth.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_repository_name_for_a_checkout_found_from_a_subdirectory`.
    Mutation: the loop checks only `start`, not its parents. Symbol: `repository_name`. Seen: `assert None == 'alpha'`.
    """
    repo = _git_dir(tmp_path, "alpha")
    deep = repo / "src" / "pkg"
    deep.mkdir(parents=True)
    assert gk.repository_name(str(repo)) == "alpha"
    assert gk.repository_name(str(deep)) == "alpha"


def test_repository_name_for_a_linked_worktree_is_the_main_checkout(tmp_path):
    """Invariant: a `.git` file `gitdir: <main>/.git/worktrees/<name>` names the MAIN checkout, so all
    worktrees of one repository are one project; the path may be POSIX, Windows or relative.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_repository_name_for_a_linked_worktree_is_the_main_checkout`.
    Mutation: the `.git` file branch returns the worktree directory's own name (`directory.name`).
    Symbol: `_name_from_gitdir_file`. Seen: `assert 'wt' == 'alpha'`.
    """
    main = _git_dir(tmp_path, "alpha")
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    posix = _git_file(tmp_path, "wt", f"gitdir: {(main / '.git' / 'worktrees' / 'wt').as_posix()}\n")
    assert gk.repository_name(str(posix)) == "alpha"
    assert gk.repository_name(str(posix / "sub")) is None  # a path that does not exist is never guessed at
    (posix / "sub").mkdir()
    assert gk.repository_name(str(posix / "sub")) == "alpha"
    windows = _git_file(tmp_path, "wt2", "gitdir: C:\\Users\\someone\\repos\\alpha\\.git\\worktrees\\wt2")
    assert gk.repository_name(str(windows)) == "alpha"
    relative = _git_file(tmp_path, "wt3", "gitdir: ../alpha/.git/worktrees/wt3")
    assert gk.repository_name(str(relative)) == "alpha"
    up = _git_file(tmp_path, "wt4", "gitdir: ../.git/worktrees/wt4")
    assert gk.repository_name(str(up)) == tmp_path.name


def test_repository_name_for_a_submodule_style_gitdir_is_the_checkouts_own_name(tmp_path):
    """Invariant: a well-formed `gitdir:` that is not a linked worktree names the checkout itself.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_repository_name_for_a_submodule_style_gitdir_is_the_checkouts_own_name`.
    Mutation: the fall-through `return directory.name or None` becomes `return None`. Symbol:
    `_name_from_gitdir_file`. Seen: `assert None == 'gk-example-sub'`.
    """
    sub = _git_file(tmp_path, "gk-example-sub", "gitdir: ../.git/modules/gk-example-sub\n")
    assert gk.repository_name(str(sub)) == "gk-example-sub"


@pytest.mark.parametrize(
    "content", [b"", b"not a gitdir line\n", b"gitdir:\n", b"gitdir:   \n", b"\xff\xfe\x00bad", b"\n\ngitdir: x/.git/worktrees/w"]
)
def test_an_unreadable_or_malformed_git_file_is_none_and_the_walk_stops(tmp_path, content):
    """Invariant: the first `.git` found is authoritative; if it is a file that cannot be read as a
    `gitdir:` line the answer is `None`, not the enclosing repository and not an exception.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_an_unreadable_or_malformed_git_file_is_none_and_the_walk_stops`.
    Mutations, symbol `repository_name` / `_name_from_gitdir_file`. A: on a malformed file the loop
    `continue`s upward (`assert 'alpha' is None` on three params). B: `_name_from_gitdir_file` drops the `startswith("gitdir:")` test
    (`assert 'bad' is None` on the `not a gitdir line` param). C: `repository_name` drops `ValueError` from its handler, so undecodable bytes escape
    (`UnicodeDecodeError` on the `\xff\xfe\x00bad` param).
    """
    _git_dir(tmp_path, "alpha")
    bad = _git_file(tmp_path / "alpha", "bad", content)
    assert gk.repository_name(str(bad)) is None


def test_repository_name_is_none_for_empty_missing_or_file_cwd(tmp_path):
    """Invariant: an empty, whitespace, nonexistent or non-directory `cwd` is `None`, never an error.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_repository_name_is_none_for_empty_missing_or_file_cwd`.
    Mutation: the `start.is_dir()` guard is removed, so a missing path walks up to the real
    repository. Symbol: `repository_name`. Seen: `assert 'alpha' is None`.
    """
    repo = _git_dir(tmp_path, "alpha")
    (repo / "f.txt").write_text("x", encoding="utf-8")
    assert gk.repository_name("") is None
    assert gk.repository_name("   ") is None
    assert gk.repository_name(str(repo / "missing" / "deeper")) is None
    assert gk.repository_name(str(repo / "f.txt")) is None
    assert gk.repository_name("\x00") is None


@pytest.mark.parametrize("config", [5, 1.5, True, "text", ["general_knowledge"], (1,)])
def test_settings_a_config_that_is_not_an_object_is_the_defaults(config):
    """Invariant: `settings` never raises; a non-dict config means "no block", i.e. the defaults.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_settings_a_config_that_is_not_an_object_is_the_defaults`.
    Mutation: the `isinstance(config, dict)` guard in `settings` is replaced by `config = config or {}`.
    Symbol: `settings`. Seen: `TypeError: argument of type 'int' is not iterable` (and the same for
    `float` and `bool`) on the first three params.
    """
    assert gk.settings(config) == {"enabled": True, "min_score": 0.0}


def test_the_analysed_command_is_capped(tmp_path):
    """Invariant (ruling R12): program extraction and `command_regex` triggers see only the first
    `MAX_COMMAND_CHARS` characters, so a program or a pattern hit that exists only past the cap does
    not match, and one inside it still does.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_the_analysed_command_is_capped`.
    Mutations, symbols `parse_command` / `matched_trigger` / `match_triggers`. A: `parse_command` scans
    the whole string (`text = command`); seen `assert 'pytest' not in ['xxxx...', 'pytest']`. B: the regex
    runs on `command` and `match_triggers` no longer truncates; seen `assert [Entry(id='gk-example-r', ...)]
    == []` (the hit past the cap matched).
    """
    cap = gk.MAX_COMMAND_CHARS
    filler = "x" * (cap - 10)
    inside = filler + " ; pytest"
    late = filler + " " * 20 + " ; pytest"
    assert gk.command_programs(inside)[-1] == "pytest"
    assert "pytest" not in gk.command_programs(late)
    entries = loaded(
        tmp_path,
        example_entry(id="gk-example-c", triggers=[{"command": "pytest"}]),
        example_entry(id="gk-example-r", triggers=[{"command_regex": r"LATEMARK"}]),
    )
    assert [e.id for e in gk.match_triggers(entries, inside, None)] == ["gk-example-c"]
    assert gk.match_triggers(entries, late, None) == []
    assert [e.id for e in gk.match_triggers(entries, "LATEMARK " + filler, None)] == ["gk-example-r"]
    late_regex = filler + " " * 20 + "LATEMARK"
    assert gk.match_triggers(entries, late_regex, None) == []
    assert gk.matched_trigger(entries[1], late_regex, [], None) is None


def test_a_megabyte_command_is_analysed_within_a_time_bound(tmp_path):
    """Invariant (ruling R12): analysing a 1 MB command is bounded by the cap, not by its length.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_megabyte_command_is_analysed_within_a_time_bound`.
    Mutation: both truncations are removed (`parse_command` reads `text = command` and `match_triggers`
    drops `command = command[:MAX_COMMAND_CHARS]`). Symbols: `parse_command`, `match_triggers`.
    Seen: `assert (… - …) < 1.0`, the uncapped 1 MB `parse_command` took 3.55 s. Each bound (1.0 s) is more than ten times the capped cost.
    """
    import time

    entries = loaded(tmp_path, example_entry(triggers=[{"command": "pytest"}, {"command_regex": r"zzz-\d+"}]))
    command = "a | " * 262_144
    for call in (lambda: gk.parse_command(command), lambda: gk.match_triggers(entries, command, None)):
        started = time.perf_counter()
        call()
        assert time.perf_counter() - started < 1.0


def test_repository_name_accepts_a_relative_cwd(tmp_path, monkeypatch):
    """Invariant: a relative `cwd` (`.`, `sub`) is made absolute first, so a checkout root named `.` still
    finds its own `.git` directory and its name.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_repository_name_accepts_a_relative_cwd`.
    Mutation: `repository_name` builds `Path(cwd)` without `os.path.abspath`. Symbol: `repository_name`.
    Seen: `assert None == 'alpha'` for `repository_name('.')`.
    """
    repo = _git_dir(tmp_path, "alpha")
    (repo / "src").mkdir()
    monkeypatch.chdir(repo)
    assert gk.repository_name(".") == "alpha"
    assert gk.repository_name("src") == "alpha"
    assert gk.repository_name("../alpha/src") == "alpha"


def test_a_host_alias_survives_an_at_sign_in_the_path_part(tmp_path):
    """Invariant: `user@` is split off only from the part before the first colon, so an `@` in the path
    neither hides the alias nor lets a path name one.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_host_alias_survives_an_at_sign_in_the_path_part`.
    Mutation: `_names_host` goes back to `arg.rpartition("@")[2]` over the whole argument. Symbol:
    `_names_host`. Seen: `assert False` for `scp examplehost:/var/x@1 .`.
    """
    entries = loaded(tmp_path, example_entry(triggers=[{"host": "examplehost"}]))
    matches = lambda command: bool(gk.match_triggers(entries, command, None))  # noqa: E731
    assert matches("scp examplehost:/var/x@1 .")
    assert matches("scp deploy@examplehost:/var/x@1 .")
    assert not matches("scp f other:/x@examplehost")
    assert not matches("scp f other:examplehost@x")


# ---------------------------------------------------------------------------------------------
# Task 3: once-per-session state and the event log.
# ---------------------------------------------------------------------------------------------


def test_an_entry_is_claimed_once_per_session(tmp_path):
    """Invariant: an id already injected in a session is not returned again in that session.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_an_entry_is_claimed_once_per_session`.
    Mutation: `claim_for_session` does not subtract the claimed ids (`fresh = unique[:MAX_PER_CALL]`).
    Symbol: `claim_for_session`. Seen: `assert ['a', 'b'] == ['c']` on the second claim.
    """
    assert gk.claim_for_session("s1", ["a", "b"], tmp_path) == ["a", "b"]
    assert gk.claim_for_session("s1", ["a", "b", "c"], tmp_path) == ["c"]
    assert gk.claim_for_session("s1", ["a", "b", "c"], tmp_path) == []


def test_two_sessions_do_not_share_state(tmp_path):
    """Invariant: two sessions running at once keep independent state, so neither one's claims
    suppress the other's first injection.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_two_sessions_do_not_share_state`.
    Mutation: the state file is one shared `shared.json` (`path = state_dir / "shared.json"`).
    Symbol: `claim_for_session`. Seen: `assert [] == ['a']` for session `s2`.
    """
    assert gk.claim_for_session("s1", ["a"], tmp_path) == ["a"]
    assert gk.claim_for_session("s2", ["a"], tmp_path) == ["a"]
    assert gk.claim_for_session("s1", ["a"], tmp_path) == []
    assert gk.claim_for_session("s2", ["a"], tmp_path) == []
    assert sorted(p.name for p in (tmp_path / "state").iterdir()) == ["s1.json", "s2.json"]


@pytest.mark.parametrize("session_id", ["", "///", "  ", None])
def test_no_session_id_caps_and_writes_nothing(tmp_path, session_id):
    """Invariant: a payload with no usable session id still injects at most `MAX_PER_CALL` entries and
    writes no per-session state file (an id that sanitises to empty counts as none).

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_no_session_id_caps_and_writes_nothing`.
    Mutation: the `if not key` early return is removed, so an empty key falls through to the state
    path (`<home>/state/.json`). Symbol: `claim_for_session`. Seen: `assert ['c'] == ['a', 'b']` for id `///` (the second call, after the first wrote `state/x.json`).
    """
    assert gk.claim_for_session(session_id, ["a", "b", "c"], tmp_path) == ["a", "b"]
    assert gk.claim_for_session(session_id, ["a", "b", "c"], tmp_path) == ["a", "b"]
    assert not (tmp_path / "state").exists()


def test_a_session_claim_is_capped_and_the_rest_stays_claimable(tmp_path):
    """Invariant: one call claims at most `MAX_PER_CALL` ids, and an id cut by the cap is NOT recorded,
    so a later call can still deliver it.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_session_claim_is_capped_and_the_rest_stays_claimable`.
    Mutation: `fresh` is not sliced (`[... for i in unique if i not in taken]`). Symbol:
    `claim_for_session`. Seen: `assert ['a', 'b', 'c'] == ['a', 'b']` on the first claim.
    """
    assert gk.claim_for_session("s1", ["a", "b", "c"], tmp_path) == ["a", "b"]
    assert gk.claim_for_session("s1", ["a", "b", "c"], tmp_path) == ["c"]


def test_order_is_kept_and_duplicates_are_ignored(tmp_path):
    """Invariant: the caller's order is preserved and a repeated id counts once.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_order_is_kept_and_duplicates_are_ignored`.
    Mutation: `unique` is built with `sorted(set(...))` instead of `dict.fromkeys`. Symbol:
    `claim_for_session`. Seen: `assert ['a', 'b'] == ['b', 'a']`.
    """
    assert gk.claim_for_session("s1", ["b", "a", "b"], tmp_path) == ["b", "a"]
    assert gk.claim_for_session("", ["z", "z", "y", "x"], tmp_path) == ["z", "y"]


def test_a_hostile_session_id_cannot_leave_the_state_directory(tmp_path):
    """Invariant: the session id is untrusted input to a path. It is sanitised to `[A-Za-z0-9_-]`, so a
    traversal id writes inside `<home>/state/` only, and ids that differ only in the removed
    characters (`a/b` and `ab`) stay two sessions.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_hostile_session_id_cannot_leave_the_state_directory`.
    Mutation: `_session_key` returns `session_id` unchanged. Symbol: `_session_key`.
    Seen: `assert not True` for `(tmp_path / 'evil.json').exists()` (the traversal id escaped `state/`).
    """
    home = tmp_path / "home"
    assert gk.claim_for_session("../../evil", ["a"], home) == ["a"]
    assert gk.claim_for_session("a/b", ["x"], home) == ["x"]
    assert gk.claim_for_session("ab", ["x"], home) == ["x"]
    assert not (tmp_path / "evil.json").exists()
    assert not (home.parent / "evil.json").exists()
    names = [p.name for p in (home / "state").iterdir()]
    assert len(names) == 3
    assert all(re.fullmatch(r"[A-Za-z0-9_-]+\.json", name) for name in names)
    # The same hostile id is stable: it finds its own state again.
    assert gk.claim_for_session("../../evil", ["a"], home) == []
    # A very long id still gives a usable file name.
    assert gk.claim_for_session("s" * 500, ["a"], home) == ["a"]
    assert gk.claim_for_session("s" * 500, ["a"], home) == []


@pytest.mark.parametrize(
    "content",
    ["{not json", '{"a": 1}', '"a"', "null", "", '["a", 3, null, "b"]'],
    ids=["garbage", "object", "string", "null", "empty", "mixed-list"],
)
def test_a_corrupt_state_file_counts_as_nothing_claimed(tmp_path, content):
    """Invariant: an unreadable, corrupt or non-list state file never raises and counts as "nothing
    claimed yet" (only string items of a list count), and the next claim rewrites it as valid JSON.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_corrupt_state_file_counts_as_nothing_claimed`.
    Mutation: `_read_claimed` returns `["a", "b", "c"]` when the content is not a list (treats foreign
    state as "everything claimed"). Symbol: `_read_claimed`. Seen: `assert [] == ['a', 'b']` for the `string` case (the claim returned nothing).
    """
    state = tmp_path / "state"
    state.mkdir()
    (state / "s1.json").write_text(content, encoding="utf-8")
    already = ["a", "b"] if content == '["a", 3, null, "b"]' else []
    expected = [i for i in ["a", "b", "c"] if i not in already][:2]
    assert gk.claim_for_session("s1", ["a", "b", "c"], tmp_path) == expected
    rewritten = json.loads((state / "s1.json").read_text(encoding="utf-8"))
    assert rewritten == already + expected


def test_an_unwritable_state_directory_still_delivers(tmp_path):
    """Invariant: when the state cannot be written (here `<home>/state` is a file) `claim_for_session`
    does not raise and still returns the claimable ids: fail open on delivery.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_an_unwritable_state_directory_still_delivers`.
    Mutation: the inner `except Exception` around `_write_claimed` returns `[]` instead of passing. Symbol: `claim_for_session`. Seen: `assert [] == ['a', 'b']` (nothing delivered).
    """
    (tmp_path / "state").write_text("not a directory", encoding="utf-8")
    assert gk.claim_for_session("s1", ["a", "b", "c"], tmp_path) == ["a", "b"]
    assert gk.claim_for_session("s1", ["a"], tmp_path) == ["a"]


def test_a_claim_without_a_home_uses_the_knowledge_home(tmp_path, monkeypatch):
    """Invariant: with no `home` argument the state lives under `knowledge_home()`.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_claim_without_a_home_uses_the_knowledge_home`.
    Mutation: the default home is a fixed `Path(".tmp-nohome")` instead of `knowledge_home()`. Symbol:
    `claim_for_session`. Seen: `assert False` for `(home / 'state' / 's1.json').is_file()`.
    """
    home = tmp_path / "elsewhere"
    monkeypatch.setenv("RECALL_GENERAL_KNOWLEDGE_HOME", str(home))
    assert gk.claim_for_session("s1", ["a"]) == ["a"]
    assert (home / "state" / "s1.json").is_file()
    assert gk.claim_for_session("s1", ["a"]) == []


def test_state_files_older_than_seven_days_are_pruned(tmp_path):
    """Invariant: a claim deletes `*.json` state files untouched for more than 7 days, and nothing else:
    a recent state file, a non-JSON file and a directory named like a state file all survive.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_state_files_older_than_seven_days_are_pruned`.
    Mutation A: `_prune_state` is not called. Mutation B: its `.endswith(".json")` test is removed,
    so the old `.txt` file is deleted too. Symbol: `_prune_state`. Seen: A: `assert [..., 'old.json', ...] == [..., 'recent.json', ...]` (old file kept). B: the `.txt` file was deleted, `'recent.json' != 'notes.txt'` at index 1.
    """
    import os
    import time

    state = tmp_path / "state"
    state.mkdir()
    old = time.time() - 8 * 24 * 3600
    (state / "old.json").write_text("[]", encoding="utf-8")
    (state / "recent.json").write_text("[]", encoding="utf-8")
    (state / "notes.txt").write_text("keep", encoding="utf-8")
    (state / "dir.json").mkdir()
    for name in ("old.json", "notes.txt", "dir.json"):
        os.utime(state / name, (old, old))
    assert gk.claim_for_session("s1", ["a"], tmp_path) == ["a"]
    assert sorted(p.name for p in state.iterdir()) == ["dir.json", "notes.txt", "recent.json", "s1.json"]


def test_a_failing_prune_does_not_break_the_claim(tmp_path, monkeypatch):
    """Invariant: pruning is best effort; a failure inside it (here `os.scandir` raising) never reaches
    the caller and the claim still succeeds and is recorded.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_failing_prune_does_not_break_the_claim`.
    Mutation: `_prune_state` catches `ZeroDivisionError` instead of `OSError`, so the scandir error
    escapes it and only the outer fallback of `claim_for_session` stops it. Symbol: `_prune_state`.
    Seen: `assert ['a', 'b'] == ['b']` (the fallback re-delivers the already claimed `a`).
    """
    import os

    def boom(*args, **kwargs):
        raise PermissionError("denied")

    assert gk.claim_for_session("s1", ["a"], tmp_path) == ["a"]
    monkeypatch.setattr(os, "scandir", boom)
    assert gk.claim_for_session("s1", ["a", "b"], tmp_path) == ["b"]
    monkeypatch.undo()
    assert json.loads((tmp_path / "state" / "s1.json").read_text(encoding="utf-8")) == ["a", "b"]


def test_log_event_appends_json_lines(tmp_path):
    """Invariant: each call appends exactly one parseable JSON line holding the given fields plus a UTC
    `ts`, earlier lines are kept, and a missing parent directory is created.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_log_event_appends_json_lines`.
    Mutation: the log is opened with mode `"w"` instead of `"a"`. Symbol: `log_event`.
    Seen: `assert ['failure'] == ['trigger', 'failure']` (the first line was truncated away).
    """
    from datetime import datetime, timedelta

    path = tmp_path / "logs" / "deeper" / "gk.jsonl"
    gk.log_event({"event": "trigger", "project": "alpha", "injected": ["gk-example-tests"]}, path)
    gk.log_event({"event": "failure", "signature": "exit:1", "note": "café", "ts": "forged"}, path)
    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n") and "\r" not in raw
    rows = [json.loads(line) for line in raw.splitlines()]
    assert [r["event"] for r in rows] == ["trigger", "failure"]
    assert rows[0]["injected"] == ["gk-example-tests"] and rows[0]["project"] == "alpha"
    assert rows[1]["note"] == "café"
    assert "\\u00e9" in raw  # default ensure_ascii
    for row in rows:
        stamp = datetime.fromisoformat(row["ts"])
        assert stamp.utcoffset() == timedelta(0)
        assert abs((datetime.now(stamp.tzinfo) - stamp).total_seconds()) < 60


def test_log_event_defaults_to_the_log_path(tmp_path, monkeypatch):
    """Invariant: with no `path` the event goes to `log_path()`.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_log_event_defaults_to_the_log_path`.
    Mutation: the default target is `Path("general-knowledge.jsonl")` instead of `log_path()`.
    Symbol: `log_event`. Seen: `FileNotFoundError` on the expected `events.jsonl` (the line went to the cwd instead).
    """
    target = tmp_path / "somewhere" / "events.jsonl"
    monkeypatch.setenv("RECALL_GENERAL_KNOWLEDGE_LOG", str(target))
    gk.log_event({"event": "trigger"})
    assert json.loads(target.read_text(encoding="utf-8"))["event"] == "trigger"


def test_log_event_never_raises_on_an_unwritable_path(tmp_path):
    """Invariant: a log line that cannot be written (the path is a directory, or its parent is a file)
    is dropped silently; `log_event` returns None and never raises.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_log_event_never_raises_on_an_unwritable_path`.
    Mutation: `except (OSError, ...)` in `log_event` becomes `except (TypeError, ValueError, ...)`.
    Symbol: `log_event`. Seen: `PermissionError: [Errno 13]` raised out of `log_event` for the directory path.
    """
    directory = tmp_path / "a-directory"
    directory.mkdir()
    assert gk.log_event({"event": "trigger"}, directory) is None
    blocker = tmp_path / "blocker"
    blocker.write_text("file", encoding="utf-8")
    assert gk.log_event({"event": "trigger"}, blocker / "child" / "log.jsonl") is None


def test_log_event_never_raises_on_an_unserialisable_record(tmp_path):
    """Invariant: a record `json.dumps` rejects (a set, a circular reference) is dropped without raising
    and without leaving a half line behind.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_log_event_never_raises_on_an_unserialisable_record`.
    Mutation: `except (OSError, TypeError, ValueError, ...)` loses `TypeError` and `ValueError`.
    Symbol: `log_event`. Seen: `TypeError: Object of type set is not JSON serializable` raised out of `log_event`.
    """
    path = tmp_path / "gk.jsonl"
    loop: dict[str, Any] = {}
    loop["self"] = loop
    assert gk.log_event({"event": "trigger", "injected": {"a"}}, path) is None
    assert gk.log_event({"event": "trigger", "loop": loop}, path) is None
    assert not path.exists() or path.read_text(encoding="utf-8") == ""
    gk.log_event({"event": "trigger"}, path)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_client_defaults_to_claude_and_follows_the_environment(monkeypatch):
    """Invariant: `client()` is `claude` unless `RECALL_HOOK_CLIENT` says otherwise (the Codex adapter).

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_client_defaults_to_claude_and_follows_the_environment`.
    Mutation: `client` returns the constant `"claude"`. Symbol: `client`. Seen: `assert 'claude' == 'codex'`.
    """
    monkeypatch.delenv("RECALL_HOOK_CLIENT", raising=False)
    assert gk.client() == "claude"
    monkeypatch.setenv("RECALL_HOOK_CLIENT", "codex")
    assert gk.client() == "codex"


# ---------------------------------------------------------------------------------------------
# Task 4: failure signature and failure ranking.
# ---------------------------------------------------------------------------------------------


def entry(**overrides: Any) -> gk.Entry:
    """An `Entry` built from the same invented record `example_entry` gives, without an index round trip."""
    record = example_entry(**overrides)
    triggers = tuple(gk.Trigger(*next(iter(t.items()))) for t in record["triggers"])
    return gk.Entry(record["id"], record["kind"], record["scope"], triggers, tuple(record["symptoms"]), record["body"])


def test_failure_signature_normalises_paths_numbers_and_hex():
    """Invariant: a path, a hex run and a number in the detail line become placeholders, so two
    occurrences of one failure share a signature.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_failure_signature_normalises_paths_numbers_and_hex`.
    Mutation: `_normalise` drops the `_DIGITS.sub("<n>", line)` line. Symbol: `_normalise`.
    Seen: `assert 'pytest|error: file <path> line 42 hash <hex> not found' == 'pytest|error: file <path> line <n> hash <hex> not found'`.
    """
    error = "Exit code 2\nERROR: file C:\\Users\\x\\proj\\a.py line 42 hash 9f3c2ab1e not found"
    assert gk.failure_signature("pytest -q", error) == "pytest|error: file <path> line <n> hash <hex> not found"


@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        ("cannot open C:\\Users\\x\\proj\\a.py now", "cannot open <path> now"),
        ("cannot open c:/a/b/c.txt now", "cannot open <path> now"),
        ("cannot open \\\\examplehost\\share\\x.txt now", "cannot open <path> now"),
        ("no such file /a/b/c.py here", "no such file <path> here"),
        ("no such file ~/x/y here", "no such file <path> here"),
        ("no such file '/a/b' here", "no such file '<path>' here"),
        ("failed at c:\\a\\b.py:42: bad thing", "failed at <path>:<n>: bad thing"),
        ("checked 1/2 and and/or / root", "checked <n>/<n> and and/or / root"),
    ],
)
def test_failure_signature_replaces_paths_but_not_ratios_or_a_bare_slash(detail, expected):
    """Invariant: Windows, UNC and POSIX paths (and `~/x`) become `<path>`; `1/2`, `and/or` and a lone
    `/` do not, because they are not paths.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_failure_signature_replaces_paths_but_not_ratios_or_a_bare_slash`.
    Mutations, one per group, each applied alone and each seen red in the named parameter:
    (a) the `_WINDOWS_PATH.sub` line removed: the three `c:` cases fail, e.g.
    `failed at c:\\a\\b.py:<n>: bad thing` instead of `failed at <path>:<n>: bad thing`;
    (b) the `_UNC_PATH.sub` line removed: the `\\\\examplehost\\share` case keeps its text;
    (c) `_PATH_START = ""` (no lookbehind): `checked <n><path> and and<path> / root` instead of
    `checked <n>/<n> and and/or / root`, the ratio case only;
    (d) the `~[\\\\/]` alternative of `_POSIX_PATH` removed: `no such file ~/x/y here` is not replaced.
    Symbols: `_WINDOWS_PATH`, `_UNC_PATH`, `_POSIX_PATH`, `_PATH_START`.
    """
    assert gk.failure_signature("make", "Exit code 1\n" + detail) == "make|" + expected


def test_a_hex_run_needs_a_digit_so_english_words_survive():
    """Invariant (ruling R5): a run of 7 or more hex characters is `<hex>` only when it holds a digit.
    `defaced` and `effaced` are words made of the letters a to f.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_hex_run_needs_a_digit_so_english_words_survive`.
    Mutation: `_HEX_RUN` loses its lookahead (`\\b[0-9a-f]{7,}\\b`). Symbol: `_HEX_RUN`.
    Seen: `assert 'make|object <hex> at <hex>' == 'make|object defaced at <hex>'`.
    """
    error = "Exit code 1\nobject defaced at 0a1b2c3d4e5f"
    assert gk.failure_signature("make", error) == "make|object defaced at <hex>"


def test_a_plain_number_is_a_number_and_a_hash_with_a_letter_is_hex():
    """Invariant (ruling R17): a run of 7 or more digits is `<n>`, not `<hex>`; a hex id needs a digit AND
    a letter a to f; a word with no digit (`defaced`) is untouched.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_plain_number_is_a_number_and_a_hash_with_a_letter_is_hex`.
    Mutation: `_HEX_RUN` loses its second lookahead `(?=[0-9a-f]*[a-f])`. Symbol: `_HEX_RUN`.
    Seen: `assert 'make|line <hex>' == 'make|line <n>'`.
    """
    assert gk.failure_signature("make", "Exit code 1\nline 1234567") == "make|line <n>"
    assert gk.failure_signature("make", "Exit code 1\nhash 9f3c2ab1e") == "make|hash <hex>"
    assert gk.failure_signature("make", "Exit code 1\nobject defaced") == "make|object defaced"


def test_the_error_is_cut_before_the_detail_line_is_looked_for():
    """Invariant (review focus 2, deterministic form): only the first `MAX_ERROR_CHARS` characters are read,
    so a detail line that starts after them does not exist. An implementation that cut late would find it.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_the_error_is_cut_before_the_detail_line_is_looked_for`.
    Mutation: `text = error[:MAX_ERROR_CHARS] ...` becomes `text = error ...` in `failure_signature`.
    Symbol: `failure_signature`. Seen: `assert 'make|boom' == 'make|exit:1'`.
    """
    error = "Exit code 1\n" + "\n" * 5000 + "boom"
    assert gk.failure_signature("make", error) == "make|exit:1"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("Exit code 2\nboom happened", "make|boom happened"),
        ("EXIT CODE 137   \n\n   killed by signal", "make|killed by signal"),
        ("boom happened\nsecond line", "make|boom happened"),
        ("Exit code 1\n", "make|exit:1"),
        ("Exit code -1", "make|exit:-1"),
        ("Exit code 3\n   \n\t\n", "make|exit:3"),
        ("", "make|exit:?"),
        ("  \n \n", "make|exit:?"),
    ],
)
def test_failure_signature_skips_the_exit_header_and_falls_back_to_the_code(error, expected):
    """Invariant: the leading `Exit code N` line is not the detail (any case, any padding); with no detail
    the line is `exit:N` unnormalised, and `exit:?` when there is no header either (ruling R6).

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_failure_signature_skips_the_exit_header_and_falls_back_to_the_code`.
    Mutations: (a) the `lines = lines[1:]` that drops the header is removed: the five cases that start
    with a header fail (the detail becomes `exit code <n>`); (b) the `'?'` fallback is replaced by `''`:
    the last two cases fail (`make|exit:`). Symbol: `failure_signature`.
    """
    assert gk.failure_signature("make test", error) == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("ssh examplehost 'uptime'", "ssh:examplehost"),
        ("ssh -p 2222 examplehost uptime", "ssh:examplehost"),
        ("ssh -i key.pem -o StrictHostKeyChecking=no alice@examplehost uptime", "ssh:examplehost"),
        ("ssh -vp 22 examplehost", "ssh:examplehost"),
        ("ssh -p22 examplehost", "ssh:examplehost"),
        ("ssh -tt examplehost", "ssh:examplehost"),
        ("ssh -O exit examplehost", "ssh:examplehost"),
        ("ssh -l bob [::1]", "ssh:::1"),
        ("ssh bob@[fe80::1]:22 uptime", "ssh:fe80::1"),
        ("ssh -- examplehost uptime", "ssh:examplehost"),
        ("ssh -J jumphost examplehost uptime", "ssh:examplehost"),
        ("ssh ExampleHost", "ssh:examplehost"),
        ("ssh", "ssh"),
        ("ssh -v", "ssh"),
        ("pytest -q tests/", "pytest"),
        ("FOO=1 python -m pytest", "python"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_command_family_is_the_program_and_for_ssh_the_alias(command, expected):
    """Invariant: the family is the first segment's program, and `ssh:<alias>` for ssh, skipping options
    and the values of the options that take one.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_command_family_is_the_program_and_for_ssh_the_alias`.
    Mutations: (a) `_ssh_host` never skips an option's value (`skip = False`): five cases fail, e.g.
    `ssh:2222`, `ssh:jumphost`, `ssh:key.pem`, `ssh:22`, `ssh:exit`; (b) `.rpartition("@")[2]` removed: the
    `alice@examplehost` case gives `ssh:alice@examplehost`; (c) the alias dropped (`return "ssh"`): ten
    cases fail; (d) `O` removed from `_SSH_VALUE_OPTIONS`: only `ssh -O exit examplehost` fails
    (`ssh:exit`); (e) the bracketed-IPv6 branch (`if host.startswith("[")`) disabled: the two `[...]`
    cases fail (`ssh:[` and `ssh:[fe80`). Symbols: `_ssh_host`, `command_family`.
    """
    assert gk.command_family(command) == expected


def test_a_non_string_command_or_error_does_not_raise():
    """Invariant: the failure path takes whatever the payload holds. A non-string is empty text.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_non_string_command_or_error_does_not_raise`.
    Mutation: `failure_signature` slices `error[:MAX_ERROR_CHARS]` without the `isinstance` guard.
    Symbol: `failure_signature`. Seen: `TypeError: 'NoneType' object is not subscriptable`.
    """
    assert gk.failure_signature(None, None) == "|exit:?"  # type: ignore[arg-type]
    assert gk.failure_signature(["pytest"], {"error": "x"}) == "|exit:?"  # type: ignore[arg-type]
    assert gk.failure_signature("pytest", 7) == "pytest|exit:?"  # type: ignore[arg-type]
    assert gk.command_family(None) == ""  # type: ignore[arg-type]
    assert gk.rank_for_failure([entry()], None, 0.0) == []  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "payload",
    ["x" * 2_000_000, "1 " * 1_000_000],
    ids=["one-long-word", "two-million-chars-of-numbers"],
)
def test_a_two_megabyte_error_is_bounded(payload):
    """Invariant (review focus 2): a 2 MB error costs no more than a 4 KB one. The signature is cut to
    `MAX_ERROR_CHARS` before any splitting or matching, and its line part is at most `SIGNATURE_CHARS`.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_two_megabyte_error_is_bounded`.
    Mutations: (a) `text = error[:MAX_ERROR_CHARS]` becomes `text = error`: the numbers payload runs the
    digit and hex rules over 2 MB and fails the time assertion (`assert (5625.89 - 5624.58) < 0.4`, 1.3 s;
    the `x` payload alone does NOT catch this, which is why the numbers payload exists); (b) the
    `[:SIGNATURE_CHARS]` slice removed: both payloads fail the length assertion (`4091 <= 167`,
    `5454 <= 167`). Symbol: `failure_signature`.
    """
    error = "Exit code 1\n" + payload
    started = time.perf_counter()
    signature = gk.failure_signature("pytest", error)
    gk.rank_for_failure([entry()], error, 0.0)
    assert time.perf_counter() - started < 0.4  # half the 800 ms budget; measured about 5 ms
    assert signature.startswith("pytest|")
    assert len(signature) <= len("pytest|") + gk.SIGNATURE_CHARS


def test_a_huge_command_is_bounded_through_the_capped_parse():
    """Invariant: the family is read from the capped parse, so an `ssh` whose host sits megabytes into the
    command line is `ssh`, and finding that out is fast.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_huge_command_is_bounded_through_the_capped_parse`.
    Mutation: `MAX_COMMAND_CHARS = 10**9` (the cap `parse_command` applies; `command_family` must keep
    going through it). Symbol: `command_family` over `parse_command`.
    Seen: `assert 'ssh:examplehost' == 'ssh'`.
    """
    command = "ssh " + "-v " * 1_000_000 + "examplehost"
    started = time.perf_counter()
    family = gk.command_family(command)
    elapsed = time.perf_counter() - started
    assert family == "ssh"
    assert elapsed < 0.8
    assert gk.failure_signature(command, "Exit code 1\nboom").startswith("ssh|")


def test_rank_for_failure_returns_the_entry_whose_symptoms_match():
    """Invariant: an error that contains an entry's symptom phrase brings that entry back, and an
    entry about something else does not.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_rank_for_failure_returns_the_entry_whose_symptoms_match`.
    Mutation: the final `return [by_id[name] ... if score >= min_score][:MAX_PER_CALL]` of
    `rank_for_failure` becomes `return []`. Symbol: `rank_for_failure`.
    Seen: `assert [] == ['gk-example-memory']`.
    """
    unrelated = entry(
        id="gk-example-network",
        symptoms=["connection refused by remote host", "network unreachable timeout"],
        body="Check the tunnel before retrying the deploy script.",
    )
    memory = entry(
        id="gk-example-memory",
        symptoms=["worker crashed out of memory during collection"],
        body="Run the focused file on the build host.",
    )
    error = "Exit code 1\nworker crashed with an out of memory failure during collection of tests"
    found = gk.rank_for_failure([unrelated, memory], error, 0.0)
    assert [e.id for e in found] == ["gk-example-memory"]


@pytest.mark.parametrize("order", [(0, 1), (1, 0)], ids=["body-first", "symptom-first"])
def test_rank_for_failure_prefers_the_symptom_match_over_the_same_words_in_a_body(order):
    """Invariant (ruling R4): an entry that lists the error's words as a symptom outranks one that has
    them once in its body, whichever comes first in the list. Without the weight the order flips.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_rank_for_failure_prefers_the_symptom_match_over_the_same_words_in_a_body`.
    Mutation: `SYMPTOM_WEIGHT = 1`. Symbol: `rank_for_failure` (the `* SYMPTOM_WEIGHT` repeat).
    Seen, in both parameters: the list comes back with `gk-example-body` first instead of
    `gk-example-symptom` (scores 1.12 against 1.07 unweighted, 1.59 against 1.30 weighted).
    """
    in_body = entry(
        id="gk-example-body",
        symptoms=[],
        body="Worker memory allocation failure during collection: crashed. Reduce parallelism on the build host.",
    )
    in_symptoms = entry(
        id="gk-example-symptom",
        symptoms=["worker crashed memory allocation failure during collection"],
        body="Run the focused file on the build host.",
    )
    error = "worker crashed after memory allocation failure during collection of tests"
    found = gk.rank_for_failure([[in_body, in_symptoms][i] for i in order], error, 0.0)
    assert [e.id for e in found] == ["gk-example-symptom", "gk-example-body"]


def test_min_score_filters_weak_matches_and_keeps_an_equal_one():
    """Invariant: an entry scoring below `min_score` is dropped; one scoring exactly `min_score` stays
    (`>=`). The threshold is taken from the ranker itself, so the test names no magic number.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_min_score_filters_weak_matches_and_keeps_an_equal_one`.
    Mutations: (a) the filter `if score >= min_score` becomes `if True`: the `score + 1e-9` call returns
    the entry (`assert [Entry(...)] == []`); (b) `>=` becomes `>`: the equal-score call returns `[]`
    (`assert [] == ['gk-example-memory']`). Symbol: `rank_for_failure`.
    """
    from recall_hooks import prompt_time

    memory = entry(
        id="gk-example-memory",
        symptoms=["worker crashed out of memory during collection"],
        body="Run the focused file on the build host.",
    )
    error = "worker crashed with an out of memory failure during collection of tests"
    document = (memory.id, "", prompt_time.tokenize(" ".join(list(memory.symptoms) * gk.SYMPTOM_WEIGHT) + " " + memory.body))
    (_, _, score) = prompt_time.rank([document], error, k=gk.MAX_PER_CALL)[0]
    assert score > 0
    assert [e.id for e in gk.rank_for_failure([memory], error, score)] == ["gk-example-memory"]
    assert gk.rank_for_failure([memory], error, score + 1e-9) == []
    assert [e.id for e in gk.rank_for_failure([memory], error, 0.0)] == ["gk-example-memory"]


def test_rank_for_failure_returns_at_most_the_per_call_cap_and_never_raises():
    """Invariant: at most `MAX_PER_CALL` entries come back, a duplicate id counts once, and a broken
    entry list or an empty error gives `[]` rather than an exception.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_rank_for_failure_returns_at_most_the_per_call_cap_and_never_raises`.
    Mutations: (a) `k=MAX_PER_CALL` becomes `k=10` AND the trailing `[:MAX_PER_CALL]` is removed (each
    alone survives, the two are redundant on purpose): `assert 3 == 2`; (b) `except Exception` narrowed to
    `except ZeroDivisionError`: the broken-list call raises `AttributeError: 'dict' object has no
    attribute 'id'`; (c) the final return replaced by `[]`: `assert 0 == 2`. Symbol: `rank_for_failure`.
    """
    symptoms = ["worker crashed out of memory during collection"]
    entries = [
        entry(id=f"gk-example-{n}", symptoms=symptoms, body="Run the focused file on the build host.") for n in range(3)
    ]
    error = "worker crashed with an out of memory failure during collection of tests"
    assert len(gk.rank_for_failure(entries, error, 0.0)) == gk.MAX_PER_CALL
    assert len(gk.rank_for_failure([entries[0], entries[0]], error, 0.0)) <= 1
    assert gk.rank_for_failure([example_entry()], error, 0.0) == []  # type: ignore[list-item]
    assert gk.rank_for_failure(entries, "   \n", 0.0) == []
    assert gk.rank_for_failure([], error, 0.0) == []


def test_importing_the_module_does_not_import_prompt_time_until_a_failure_is_ranked():
    """Invariant: the `pre-tool-use` path never pays for `prompt_time`; only `rank_for_failure` imports it.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_importing_the_module_does_not_import_prompt_time_until_a_failure_is_ranked`.
    Mutation: `from . import prompt_time` at module level of `recall_hooks/general_knowledge.py`.
    Symbol: `rank_for_failure`. Seen: `assert ['True', 'True'] == ['False', 'True']`.
    """
    code = (
        "import sys\n"
        "import recall_hooks.general_knowledge as gk\n"
        "before = 'recall_hooks.prompt_time' in sys.modules\n"
        "e = gk.Entry('gk-example-x', 'procedure', 'global', (), ('worker crashed',), 'body words here')\n"
        "gk.rank_for_failure([e], 'worker crashed with a failure', 0.0)\n"
        "print(before, 'recall_hooks.prompt_time' in sys.modules)\n"
    )
    root = Path(__file__).resolve().parents[1]
    done = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=120)
    assert done.stdout.split() == ["False", "True"], done.stderr


def test_render_formats_each_entry_with_its_id_and_separates_them_with_a_blank_line():
    """Invariant: the injected text is the header, then `[id]` and the body per entry, blank line between,
    and nothing for no entries.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_render_formats_each_entry_with_its_id_and_separates_them_with_a_blank_line`.
    Mutation: the separator `"\\n\\n"` in `render` becomes `"\\n"`. Symbol: `render`.
    Seen: the two entries come back separated by one newline instead of a blank line (`...body a\\n[gk-example-b]...`).
    """
    first = entry(id="gk-example-a", body="body a")
    second = entry(id="gk-example-b", body="body b")
    assert gk.render([first, second]) == (
        "General knowledge (owner-approved):\n\n[gk-example-a]\nbody a\n\n[gk-example-b]\nbody b"
    )
    assert gk.render([first]) == "General knowledge (owner-approved):\n\n[gk-example-a]\nbody a"
    assert gk.render([]) == ""


# ---------------------------------------------------------------------------------------------
# PreToolUse integration (task 5): `python -m recall_hooks pre-tool-use` through `main`.
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def hook(tmp_path, monkeypatch, capsys):
    """An isolated hook environment plus a `run(payload)` that feeds `main` and returns its stdout.

    Every location a hook could read is a tmp dir: the corpus, the event log, and both config homes
    (so write time never sees a real `recall-hook.json`, which would carry a `dsn`).
    """
    home = tmp_path / "gk-home"
    config_home = tmp_path / "config-home"
    config_home.mkdir()
    log = tmp_path / "gk-events.jsonl"
    monkeypatch.setenv("RECALL_GENERAL_KNOWLEDGE_HOME", str(home))
    monkeypatch.setenv("RECALL_GENERAL_KNOWLEDGE_LOG", str(log))
    monkeypatch.setenv("RECALL_PROJECT_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("RECALL_HOOK_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_home))
    monkeypatch.delenv("RECALL_HOOK_CLIENT", raising=False)

    class Env:
        def __init__(self) -> None:
            self.home, self.config_home, self.log = home, config_home, log

        def index(self, *entries: dict[str, Any]) -> None:
            write_index(home, list(entries))

        def config(self, **block: Any) -> None:
            (config_home / "recall-hook.json").write_text(
                json.dumps({"general_knowledge": block}), encoding="utf-8"
            )

        def events(self) -> list[dict[str, Any]]:
            if not log.exists():
                return []
            return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

        def run(self, payload: dict[str, Any], event: str = "pre-tool-use") -> str:
            monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
            capsys.readouterr()
            assert recall_hooks.main([event]) == 0
            return capsys.readouterr().out

    return Env()


def shell(command: str, session: str | None = "s1", tool: str = "Bash", cwd: Any = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"tool_name": tool, "tool_input": {"command": command}}
    if session is not None:
        payload["session_id"] = session
    if cwd is not None:
        payload["cwd"] = str(cwd)
    return payload


def injected_text(output: str) -> str:
    document = json.loads(output)
    assert document["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    return document["hookSpecificOutput"]["additionalContext"]


def test_main_injects_a_matching_entry_once(hook):
    """Invariant: a matching entry is injected on the first call of a session and suppressed on the
    second, and both calls are logged, the second with the id in `suppressed`.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_main_injects_a_matching_entry_once`.
    Mutation: in `pre_tool_use_context` replace `claim_for_session(session, [e.id for e in matched])` with
    `[e.id for e in matched][:MAX_PER_CALL]` (no dedupe). Symbol: `pre_tool_use_context`. Seen: the second
    call printed the injection again, `assert '{"hookSpecif...ry here."}}\n' == ''`.
    """
    hook.index(example_entry())
    first = hook.run(shell("pytest -q"))
    assert "[gk-example-tests]" in injected_text(first)
    assert hook.run(shell("pytest -q")) == ""
    events = hook.events()
    assert [e["event"] for e in events] == ["trigger", "trigger"]
    assert (events[0]["injected"], events[0]["suppressed"]) == (["gk-example-tests"], [])
    assert (events[1]["injected"], events[1]["suppressed"]) == ([], ["gk-example-tests"])


def test_the_trigger_record_has_exactly_the_documented_fields(hook, monkeypatch):
    """Invariant: a `"trigger"` event carries `event`, `project`, `session`, `client`, `trigger`
    (one `"<kind>:<value>"` per matched entry, in match order, the first trigger that fired),
    `injected` and `suppressed`, plus the logger's `ts`, and nothing else.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_the_trigger_record_has_exactly_the_documented_fields`.
    Mutation: the `trigger` list is built from `entry.triggers[0]` instead of the trigger that fired.
    Symbol: `pre_tool_use_context`. Seen: `assert ['command:pyt...ommand:rsync'] == ['command:pyt...:examplehost']`
    (the second entry lists `rsync`, which did not match, instead of `host:examplehost`).
    """
    monkeypatch.setenv("RECALL_HOOK_CLIENT", "codex")
    hook.index(
        example_entry(id="gk-example-tests"),
        example_entry(id="gk-example-ssh", triggers=[{"command": "rsync"}, {"host": "examplehost"}]),
        example_entry(id="gk-example-never", triggers=[{"command": "unrelated"}]),
    )
    out = hook.run(shell("pytest -q && ssh examplehost uptime", session="s9"))
    assert injected_text(out).count("[gk-example-") == 2
    (event,) = hook.events()
    assert set(event) == {"ts", "event", "project", "session", "client", "trigger", "injected", "suppressed"}
    assert event["event"] == "trigger"
    assert (event["session"], event["client"], event["project"]) == ("s9", "codex", None)
    assert event["trigger"] == ["command:pytest", "host:examplehost"]
    assert event["injected"] == ["gk-example-tests", "gk-example-ssh"]
    assert event["suppressed"] == []


def test_a_capped_match_is_logged_as_suppressed(hook):
    """Invariant: a match the per-call cap drops is logged in `suppressed`, in match order, and
    matching nothing logs nothing.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_capped_match_is_logged_as_suppressed`.
    Mutation: the logged `suppressed` is a constant `[]`. Symbol: `pre_tool_use_context`. Seen:
    `assert (['gk-example...ample-b'], []) == (['gk-example...k-example-c'])`; the once test fails too.
    """
    hook.index(*(example_entry(id=f"gk-example-{n}") for n in "abc"))
    assert hook.run(shell("echo nothing here")) == ""
    assert hook.events() == []
    text = injected_text(hook.run(shell("pytest -q")))
    assert text.count("[gk-example-") == gk.MAX_PER_CALL
    (event,) = hook.events()
    assert (event["injected"], event["suppressed"]) == (["gk-example-a", "gk-example-b"], ["gk-example-c"])


def test_main_is_silent_when_disabled(hook):
    """Invariant: `general_knowledge.enabled: false` means no output, no log line and no state.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_main_is_silent_when_disabled`.
    Mutation: replace the `if not load_settings()["enabled"]:` gate in `pre_tool_use_context` with `if False:`.
    Symbol: `pre_tool_use_context`. Seen: `assert '{"hookSpecif...ry here."}}\n' == ''`.
    """
    hook.index(example_entry())
    hook.config(enabled=False)
    assert hook.run(shell("pytest -q")) == ""
    assert hook.events() == []
    assert not (hook.home / "state").exists()


def test_a_disabled_hook_does_not_read_the_index(hook, monkeypatch):
    """Invariant: the switch is read before the index, so a disabled feature costs one small file.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_disabled_hook_does_not_read_the_index`.
    Mutation: call `load_index()` before the `load_settings()` gate in `pre_tool_use_context`.
    Symbol: `pre_tool_use_context`. Seen: `assert ['read'] == []`.
    """
    hook.config(enabled=False)
    reads: list[str] = []
    monkeypatch.setattr(gk, "load_index", lambda *a, **k: reads.append("read") or [])
    assert hook.run(shell("pytest -q")) == ""
    assert reads == []


@pytest.mark.parametrize("tool", ["Write", "Read", "Edit"])
def test_main_ignores_non_shell_tools(hook, tool):
    """Invariant: only `Bash` and `PowerShell` are matched, even when the input has a `command` key.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_main_ignores_non_shell_tools`.
    Mutation: `SHELL_TOOLS` gains `"Write"`. Symbol: `SHELL_TOOLS`. Seen: the `[Write]` case failed with
    `assert '{"hookSpecif...ry here."}}\n' == ''`; `[Read]` and `[Edit]` stayed green, as they should.
    """
    hook.index(example_entry())
    assert hook.run(shell("pytest -q", tool=tool)) == ""
    assert hook.events() == []


def test_a_powershell_command_is_matched(hook):
    """Invariant: `PowerShell` is a shell tool and matches like `Bash`.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_powershell_command_is_matched`.
    Mutation: `SHELL_TOOLS = frozenset({"Bash"})`. Symbol: `SHELL_TOOLS`. Seen:
    `json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)` from `injected_text`,
    because nothing was printed.
    """
    hook.index(example_entry())
    assert "[gk-example-tests]" in injected_text(hook.run(shell("pytest -q", tool="PowerShell")))


@pytest.mark.parametrize("tool_input", [None, "pytest", {"command": 3}, {"command": "   "}, {}])
def test_a_shell_payload_without_a_command_string_is_silent(hook, tool_input):
    """Invariant: a malformed `tool_input` is no command, not an error.

    Red proof. NONE POSSIBLE for the guard itself: the fail-open `try` turns the error a missing
    `isinstance` check would raise into the same silence, so this is a CONTROL for the contract
    that a bad payload exits 0 with no output, and is labelled as such.
    """
    hook.index(example_entry())
    assert hook.run({"tool_name": "Bash", "tool_input": tool_input}) == ""


def test_main_survives_a_raising_matcher(hook, monkeypatch):
    """Invariant: any exception inside the general knowledge path is silence, never a crash.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_main_survives_a_raising_matcher`.
    Mutation: narrow the `except Exception` at the end of `pre_tool_use_context` to `except KeyError`.
    Symbol: `pre_tool_use_context`. Seen: `RuntimeError: matcher exploded` at the direct call (same with
    the dispatcher guard narrowed as well). The dispatcher guard is proved by
    `test_one_source_failing_does_not_silence_the_other`.
    """
    hook.index(example_entry())

    def explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("matcher exploded")

    monkeypatch.setattr(gk, "matched_trigger", explode)
    assert gk.pre_tool_use_context(shell("pytest -q")) is None
    assert hook.run(shell("pytest -q")) == ""


def test_both_contexts_share_one_json_object(hook, monkeypatch):
    """Invariant: when both sources speak, ONE object is printed whose text holds both, general
    knowledge first, joined by a blank line.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_both_contexts_share_one_json_object`.
    Mutation: the dispatcher sends `texts[-1]` instead of `"\n\n".join(texts)` (the first source is lost).
    Symbol: `recall_hooks._pre_tool_use`. Seen: `assert False` on the `startswith("General knowledge")` line.
    """
    hook.index(example_entry())
    monkeypatch.setattr(write_time, "context_for", lambda payload: "WT")
    text = injected_text(hook.run(shell("pytest -q")))
    assert text.startswith("General knowledge (owner-approved):")
    assert text.endswith("\n\nWT")


def test_write_time_alone_prints_what_it_always_printed(hook, monkeypatch, capsys):
    """Invariant: with no general knowledge match, the printed bytes are exactly the object
    `write_time.pre_tool_use` prints for the same text (same `json.dumps` form, same key order).

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_write_time_alone_prints_what_it_always_printed`.
    Mutation: the dispatcher's `hookEventName` becomes `"PreToolUse "`. Symbol: `recall_hooks._pre_tool_use`.
    Seen: `via_main == capsys.readouterr().out` fails, the two printed strings differ.
    """
    monkeypatch.setattr(write_time, "context_for", lambda payload: "WT")
    via_main = hook.run(shell("ls"))
    capsys.readouterr()
    write_time.pre_tool_use(shell("ls"))
    assert via_main == capsys.readouterr().out
    assert injected_text(via_main) == "WT"


def test_one_source_failing_does_not_silence_the_other(hook, monkeypatch):
    """Invariant: a raising write-time path leaves the general knowledge text, and a raising general
    knowledge path leaves the write-time text; the exit status is 0 either way.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_one_source_failing_does_not_silence_the_other`.
    Mutation: narrow the dispatcher's first `except Exception` to `except ImportError`, and separately the
    second. Symbol: `recall_hooks._pre_tool_use`. Seen, both: `RuntimeError: source exploded` escaped
    `main` (first mutation in the second half, second mutation in the first half).
    """
    hook.index(example_entry())

    def explode(payload: dict[str, Any]) -> str:
        raise RuntimeError("source exploded")

    with monkeypatch.context() as patch:
        patch.setattr(write_time, "context_for", explode)
        text = injected_text(hook.run(shell("pytest -q", session="a")))
    assert "[gk-example-tests]" in text and "WT" not in text

    with monkeypatch.context() as patch:
        patch.setattr(gk, "pre_tool_use_context", explode)
        patch.setattr(write_time, "context_for", lambda payload: "WT")
        assert injected_text(hook.run(shell("pytest -q", session="b"))) == "WT"


def test_general_knowledge_needs_no_write_time_configuration(hook):
    """Invariant: with no `dsn` and no `project_root` (write time unconfigured) general knowledge
    still speaks, because it runs before and independently of the write-time gates.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_general_knowledge_needs_no_write_time_configuration`.
    Mutation: the dispatcher returns 0 first when `load_config().get("dsn")` is falsy. Symbol:
    `recall_hooks._pre_tool_use`. Seen: `json.decoder.JSONDecodeError` from `injected_text` (no output).
    """
    hook.index(example_entry())
    assert not (hook.config_home / "recall-hook.json").exists()
    assert "[gk-example-tests]" in injected_text(hook.run(shell("pytest -q", cwd=hook.config_home)))


def test_without_a_session_id_the_cap_holds_and_no_state_is_written(hook):
    """Invariant: no session id means at most `MAX_PER_CALL` entries are injected, the rest are
    `suppressed`, and no `state/` directory is created.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_without_a_session_id_the_cap_holds_and_no_state_is_written`.
    Mutation: `pre_tool_use_context` uses `"anonymous"` instead of `""` for a missing `session_id`. Symbol:
    `pre_tool_use_context`. Seen: `assert 'anonymous' == ''` on the logged session (a state file would follow).
    """
    hook.index(*(example_entry(id=f"gk-example-{n}") for n in "abc"))
    text = injected_text(hook.run(shell("pytest -q", session=None)))
    assert text.count("[gk-example-") == gk.MAX_PER_CALL
    (event,) = hook.events()
    assert event["session"] == ""
    assert (event["injected"], event["suppressed"]) == (["gk-example-a", "gk-example-b"], ["gk-example-c"])
    assert not (hook.home / "state").exists()


def test_a_project_scoped_entry_fires_only_in_that_repository(hook, tmp_path):
    """Invariant: a `project:alpha` entry is injected from inside the `alpha` checkout and nowhere
    else, and the logged `project` is the repository name.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_project_scoped_entry_fires_only_in_that_repository`.
    Mutation: (a) `project = None` in `pre_tool_use_context`: the in-repo call is silent, `JSONDecodeError`
    on the last injection assertion. (b) `project = "alpha"` constant: the `beta` call injects,
    `assert '{"hookSpecif...ry here."}}\n' == ''`. Symbol: `pre_tool_use_context`.
    """
    alpha = tmp_path / "alpha"
    (alpha / ".git").mkdir(parents=True)
    beta = tmp_path / "beta"
    (beta / ".git").mkdir(parents=True)
    hook.index(example_entry(id="gk-example-alpha-only", scope="project:alpha"))
    assert hook.run(shell("pytest -q", session="s1", cwd=beta)) == ""
    assert hook.events() == []
    assert hook.run(shell("pytest -q", session="s2", cwd=tmp_path)) == ""
    assert "[gk-example-alpha-only]" in injected_text(hook.run(shell("pytest -q", session="s3", cwd=alpha)))
    (event,) = hook.events()
    assert event["project"] == "alpha"


# ---------------------------------------------------------------------------------------------
# PostToolUseFailure integration and the Codex adapter (task 6).
# ---------------------------------------------------------------------------------------------

#: An error with well over three content words, four of them shared with the symptoms of
#: `example_entry` ("collection error import failure"), which is what `prompt_time.rank` gates on.
MATCHING_ERROR = "Exit code 2\nERROR collection error: import failure while collecting tests"
#: An error that shares no word with any invented symptom or body.
UNRELATED_ERROR = "Exit code 3\nsomething completely unrelated happened yesterday afternoon"
FAILURE_EVENT = "post-tool-use-failure"


def failure(command: str = "pytest -q", error: Any = MATCHING_ERROR, session: str | None = "s1",
            tool: str = "Bash", **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"tool_name": tool, "tool_input": {"command": command}, "error": error}
    if session is not None:
        payload["session_id"] = session
    payload.update(extra)
    return payload


def failure_text(output: str) -> str:
    document = json.loads(output)
    assert document["hookSpecificOutput"]["hookEventName"] == "PostToolUseFailure"
    return document["hookSpecificOutput"]["additionalContext"]


def test_failure_injects_and_logs(hook):
    """Invariant: a failed command whose error matches an entry's symptoms injects that entry as
    `PostToolUseFailure` context and logs one `"failure"` event holding the signature and the id.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_failure_injects_and_logs`.
    Baseline: the tree before this task, where `main` has no `post-tool-use-failure` branch and
    returns 0 with no output. Symbol: `recall_hooks.main`. Seen: `json.decoder.JSONDecodeError` from
    `failure_text` (empty stdout) on the first assertion.
    """
    hook.index(example_entry())
    command = "pytest -q"
    out = hook.run(failure(command), FAILURE_EVENT)
    assert "[gk-example-tests]" in failure_text(out)
    (event,) = hook.events()
    assert event["event"] == "failure"
    assert event["signature"] == gk.failure_signature(command, MATCHING_ERROR)
    assert event["signature"].startswith("pytest|")
    assert (event["injected"], event["suppressed"]) == (["gk-example-tests"], [])


def test_a_repeated_failure_is_suppressed_not_reinjected(hook):
    """Invariant: the once-per-session dedupe is shared with the failure path: the second identical
    failure prints nothing and logs the id as `suppressed`.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_repeated_failure_is_suppressed_not_reinjected`.
    Mutation: in `post_tool_use_failure` replace `set(claim_for_session(session, ranked_ids))` with
    `set(ranked_ids[:MAX_PER_CALL])` (no dedupe). Symbol: `post_tool_use_failure`. Seen: the second run printed
    the injection again, `assert '{"hookSpecif...ry here."}}\n' == ''`.
    """
    hook.index(example_entry())
    assert "[gk-example-tests]" in failure_text(hook.run(failure(), FAILURE_EVENT))
    assert hook.run(failure(), FAILURE_EVENT) == ""
    first, second = hook.events()
    assert (first["injected"], first["suppressed"]) == (["gk-example-tests"], [])
    assert (second["injected"], second["suppressed"]) == ([], ["gk-example-tests"])


def test_unmatched_failure_is_logged_without_output(hook):
    """Invariant: EVERY failure is logged (spec 8), matched or not: an unmatched one prints nothing
    and logs its signature with empty `injected` and `suppressed`.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_unmatched_failure_is_logged_without_output`.
    Mutation: in `post_tool_use_failure` the `log_event` call is guarded by `if ranked_ids:` (log only when
    something matched). Symbol: `post_tool_use_failure`. Seen: `ValueError: not enough values to unpack
    (expected 1, got 0)` on `(event,) = hook.events()`.
    """
    hook.index(example_entry())
    assert hook.run(failure(error=UNRELATED_ERROR), FAILURE_EVENT) == ""
    (event,) = hook.events()
    assert event["event"] == "failure"
    assert event["signature"] == gk.failure_signature("pytest -q", UNRELATED_ERROR)
    assert (event["injected"], event["suppressed"]) == ([], [])


def test_a_failure_with_no_index_is_still_logged(hook):
    """Invariant: no index at all is not a reason to drop the event; it is the case the log is for.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_failure_with_no_index_is_still_logged`.
    Mutation: `post_tool_use_failure` returns early `if not entries:` before logging. Symbol:
    `post_tool_use_failure`. Seen: `ValueError: not enough values to unpack (expected 1, got 0)`.
    """
    assert hook.run(failure(), FAILURE_EVENT) == ""
    (event,) = hook.events()
    assert (event["event"], event["injected"], event["suppressed"]) == ("failure", [], [])


def test_interrupt_is_ignored(hook):
    """Invariant: a user interrupt is not a failure of the command: no output and no log line, while
    `is_interrupt: false` is an ordinary failure.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_interrupt_is_ignored`.
    Mutation: remove the `payload.get("is_interrupt") is True` skip in `post_tool_use_failure`. Symbol:
    `post_tool_use_failure`. Seen: `assert '{"hookSpecif...ry here."}}\n' == ''`.
    """
    hook.index(example_entry())
    assert hook.run(failure(is_interrupt=True), FAILURE_EVENT) == ""
    assert hook.events() == []
    assert "[gk-example-tests]" in failure_text(hook.run(failure(is_interrupt=False), FAILURE_EVENT))


@pytest.mark.parametrize("error", [None, 3, ["Exit code 1"], {"text": "x"}])
def test_a_non_string_error_is_ignored(hook, error):
    """Invariant: an `error` that is not text is not analysed: no output, no log line.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_non_string_error_is_ignored`.
    Mutation: replace the `if not isinstance(error, str):` skip in `post_tool_use_failure` with `if False:`.
    Symbol: `post_tool_use_failure`. Seen: one `"failure"` event was logged (first parameter `None`, `-x`),
    `assert [{'ts': ..., 'event': 'failure', ...}] == []`.
    """
    hook.index(example_entry())
    assert hook.run(failure(error=error), FAILURE_EVENT) == ""
    assert hook.events() == []


def test_a_failure_without_an_error_field_is_ignored(hook):
    """Invariant: a payload with no `error` key at all is skipped like a non-text one.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_failure_without_an_error_field_is_ignored`.
    Mutation: `error = payload.get("error", "")` in `post_tool_use_failure`. Symbol: `post_tool_use_failure`.
    Seen: `assert [{'ts': ..., 'event': 'failure', ...}] == []`.
    """
    hook.index(example_entry())
    payload = failure()
    del payload["error"]
    assert hook.run(payload, FAILURE_EVENT) == ""
    assert hook.events() == []


@pytest.mark.parametrize("tool", ["Write", "Read", "Edit", "Task"])
def test_a_failure_of_a_non_shell_tool_is_ignored(hook, tool):
    """Invariant: only `Bash` and `PowerShell` failures are analysed.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_failure_of_a_non_shell_tool_is_ignored`.
    Mutation: remove the `tool_name not in SHELL_TOOLS` skip in `post_tool_use_failure`. Symbol:
    `post_tool_use_failure`. Seen: one `"failure"` event was logged for every tool,
    `assert [{'ts': ..., 'event': 'failure', ...}] == []`.
    """
    hook.index(example_entry())
    assert hook.run(failure(tool=tool), FAILURE_EVENT) == ""
    assert hook.events() == []


def test_a_powershell_failure_is_analysed(hook):
    """Invariant: `PowerShell` carries the same `command` and is a shell tool.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_powershell_failure_is_analysed`.
    Mutation: `SHELL_TOOLS = frozenset({"Bash"})`. Symbol: `SHELL_TOOLS`. Seen: `JSONDecodeError` from
    `failure_text` (empty stdout).
    """
    hook.index(example_entry())
    assert "[gk-example-tests]" in failure_text(hook.run(failure(tool="PowerShell"), FAILURE_EVENT))


def test_a_disabled_hook_logs_nothing_on_failure(hook, monkeypatch):
    """Invariant: `enabled: false` means no output, no log line, no state and no index read, even
    for a failure that would match.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_disabled_hook_logs_nothing_on_failure`.
    Mutation 1: replace the `if not config["enabled"]:` gate in `post_tool_use_failure` with `if False:`.
    Mutation 2: call `load_index()` ahead of the gate. Symbol: `post_tool_use_failure`. Seen: (1)
    `assert '{"hookSpecif...ry here."}}\n' == ''`; (2) `assert ['read'] == []`.
    """
    hook.index(example_entry())
    hook.config(enabled=False)
    reads: list[str] = []
    real_load = gk.load_index
    monkeypatch.setattr(gk, "load_index", lambda *a, **k: reads.append("read") or real_load(*a, **k))
    assert hook.run(failure(), FAILURE_EVENT) == ""
    assert hook.events() == []
    assert reads == []
    assert not (hook.home / "state").exists()


def test_the_failure_record_has_exactly_the_documented_fields(hook, monkeypatch):
    """Invariant: a `"failure"` event carries `event`, `project`, `session`, `client`, `signature`,
    `injected` and `suppressed`, plus the logger's `ts`, and nothing else; `client` follows
    `RECALL_HOOK_CLIENT` and `project` is the repository name.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_the_failure_record_has_exactly_the_documented_fields`.
    Mutation 1: the logged record gains `"trigger": []` (the trigger event's shape). Mutation 2:
    `"client": "claude"` as a constant. Mutation 3: `"project": None`. Symbol: `post_tool_use_failure`.
    Seen: (1) the key set differs by `'trigger'`; (2) `assert ('s9', 'claude', 'alpha') == ('s9', 'codex', 'alpha')`;
    (3) `assert ('s9', 'codex', None) == ('s9', 'codex', 'alpha')`.
    """
    monkeypatch.setenv("RECALL_HOOK_CLIENT", "codex")
    alpha = hook.config_home.parent / "alpha"
    (alpha / ".git").mkdir(parents=True)
    hook.index(example_entry())
    out = hook.run(failure(session="s9", cwd=str(alpha)), FAILURE_EVENT)
    assert "[gk-example-tests]" in failure_text(out)
    (event,) = hook.events()
    assert set(event) == {"ts", "event", "project", "session", "client", "signature", "injected", "suppressed"}
    assert (event["session"], event["client"], event["project"]) == ("s9", "codex", "alpha")


def test_a_project_scoped_entry_is_not_ranked_outside_its_repository(hook, tmp_path):
    """Invariant: failure ranking only sees entries in scope for the failing command's repository.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_project_scoped_entry_is_not_ranked_outside_its_repository`.
    Mutation: in `post_tool_use_failure` the entry list drops the `in_scope(e, project)` filter. Symbol:
    `post_tool_use_failure`. Seen: the `beta` run injected, `assert '{"hookSpecif...ry here."}}\n' == ''`.
    """
    alpha = tmp_path / "alpha"
    (alpha / ".git").mkdir(parents=True)
    beta = tmp_path / "beta"
    (beta / ".git").mkdir(parents=True)
    hook.index(example_entry(scope="project:alpha"))
    assert hook.run(failure(session="a", cwd=str(beta)), FAILURE_EVENT) == ""
    assert "[gk-example-tests]" in failure_text(hook.run(failure(session="b", cwd=str(alpha)), FAILURE_EVENT))


def test_min_score_from_the_config_applies_to_failures(hook):
    """Invariant: `general_knowledge.min_score` is passed to the ranker: a score floor above any real
    BM25 score injects nothing, but the failure is still logged.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_min_score_from_the_config_applies_to_failures`.
    Mutation: `post_tool_use_failure` passes `0.0` instead of `config["min_score"]` to `rank_for_failure`.
    Symbol: `post_tool_use_failure`. Seen: `assert '{"hookSpecif...ry here."}}\n' == ''`.
    """
    hook.index(example_entry())
    hook.config(enabled=True, min_score=1000.0)
    assert hook.run(failure(), FAILURE_EVENT) == ""
    (event,) = hook.events()
    assert (event["injected"], event["suppressed"]) == ([], [])


def test_a_failure_without_a_session_id_caps_and_writes_no_state(hook):
    """Invariant: no session id means at most `MAX_PER_CALL` entries are injected (three entries match,
    the ranker's own cap keeps two) and no `state/` directory is created.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_failure_without_a_session_id_caps_and_writes_no_state`.
    Mutation: `post_tool_use_failure` uses `"anonymous"` instead of `""` for a missing `session_id`.
    Symbol: `post_tool_use_failure`. Seen: `assert 'anonymous' == ''` on the logged session (a state file
    would follow).
    """
    hook.index(*(example_entry(id=f"gk-example-{n}") for n in "abc"))
    text = failure_text(hook.run(failure(session=None), FAILURE_EVENT))
    assert text.count("[gk-example-") == gk.MAX_PER_CALL
    (event,) = hook.events()
    assert event["session"] == ""
    assert len(event["injected"]) == gk.MAX_PER_CALL
    assert not (hook.home / "state").exists()


@pytest.mark.parametrize("tool_input", [None, {}, {"command": 3}, "pytest"])
def test_a_failure_with_no_command_text_is_still_analysed(hook, tool_input):
    """Invariant: a missing or non-text `command` is the empty command, not a reason to skip: the
    error is what matters on this path, and the logged signature has the empty family.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_a_failure_with_no_command_text_is_still_analysed`.
    Mutation: `post_tool_use_failure` returns 0 `if not command`. Symbol: `post_tool_use_failure`.
    Seen: `JSONDecodeError` from `failure_text` (empty stdout) for every parameter.
    """
    hook.index(example_entry())
    payload = failure()
    payload["tool_input"] = tool_input
    assert "[gk-example-tests]" in failure_text(hook.run(payload, FAILURE_EVENT))
    (event,) = hook.events()
    assert event["signature"] == gk.failure_signature("", MATCHING_ERROR)


def test_ranking_raising_still_logs_the_signature(hook, monkeypatch):
    """Invariant: the signature is computed and logged even when ranking fails; the hook is silent
    and returns 0.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_ranking_raising_still_logs_the_signature`.
    Mutation: narrow the inner `except Exception` around `rank_for_failure` to `except ZeroDivisionError`,
    so the outer guard swallows the error before `log_event` runs. Symbol: `post_tool_use_failure`. Seen: `ValueError: not enough values to unpack
    (expected 1, got 0)` on `(event,) = hook.events()`.
    """
    hook.index(example_entry())

    def explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("ranking exploded")

    monkeypatch.setattr(gk, "rank_for_failure", explode)
    assert hook.run(failure(), FAILURE_EVENT) == ""
    (event,) = hook.events()
    assert event["signature"] == gk.failure_signature("pytest -q", MATCHING_ERROR)
    assert (event["injected"], event["suppressed"]) == ([], [])


def test_the_failure_path_never_raises(hook, monkeypatch):
    """Invariant: a settings reader that explodes is silence and exit 0, from the function and
    through `main`.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_the_failure_path_never_raises`.
    Mutation: narrow the outer `except Exception` of `post_tool_use_failure` to `except KeyError`.
    Symbol: `post_tool_use_failure`. Seen: `RuntimeError: settings exploded` at the direct call.
    """
    hook.index(example_entry())

    def explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("settings exploded")

    monkeypatch.setattr(gk, "load_settings", explode)
    assert gk.post_tool_use_failure(failure()) == 0
    assert hook.run(failure(), FAILURE_EVENT) == ""


def test_main_survives_a_raising_failure_hook(hook, monkeypatch):
    """Invariant: if `post_tool_use_failure` itself raises, `main` still returns 0 and prints nothing.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_main_survives_a_raising_failure_hook`.
    Mutation: narrow the dispatch's `except Exception` in `recall_hooks.main` to `except KeyError`.
    Symbol: `recall_hooks.main`. Seen: `RuntimeError: hook exploded` escaped `main`.
    """
    hook.index(example_entry())

    def explode(payload: dict[str, Any]) -> int:
        raise RuntimeError("hook exploded")

    monkeypatch.setattr(gk, "post_tool_use_failure", explode)
    assert hook.run(failure(), FAILURE_EVENT) == ""


def test_main_survives_a_missing_general_knowledge_module(hook, monkeypatch):
    """Invariant: an install without the module (an older checkout) degrades to silence and exit 0.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_main_survives_a_missing_general_knowledge_module`.
    Mutation: the dispatch narrows its guard to `except RuntimeError`, so the import failure of the
    missing module is not caught. Symbol: `recall_hooks.main`. Seen: `ModuleNotFoundError: import of
    recall_hooks.general_knowledge halted; None in sys.modules` escaped `main`.
    """
    hook.index(example_entry())
    monkeypatch.setitem(sys.modules, "recall_hooks.general_knowledge", None)
    assert hook.run(failure(), FAILURE_EVENT) == ""


def codex_run(monkeypatch, capsys, payload: dict[str, Any], event: str = "pre-tool-use") -> str:
    from recall_hooks import codex

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    capsys.readouterr()
    assert codex.main([event]) == 0
    return capsys.readouterr().out


def test_codex_adapter_marks_the_client(hook, tmp_path, monkeypatch, capsys):
    """Invariant: a hook run through the Codex adapter logs `client == "codex"`, writes to the SAME
    log and knowledge home as a Claude run (ruling R1: one corpus, one log, one switch even though
    the adapter redirects `RECALL_HOOK_CONFIG_HOME`), obeys the shared switch, and leaves
    `RECALL_HOOK_CLIENT` exactly as it found it (absent, then a preset value).

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_codex_adapter_marks_the_client`.
    Mutation 1: remove the `os.environ["RECALL_HOOK_CLIENT"] = "codex"` line in `recall_hooks.codex.main`.
    Mutation 2: delete the `RECALL_HOOK_CLIENT` restore branches of the `finally`. Symbol:
    `recall_hooks.codex.main`. Seen: (1) `assert ['claude', 'claude'] == ['claude', 'codex']`;
    (2) `assert 'RECALL_HOOK_CLIENT' not in environ(...)`.
    """
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    hook.index(example_entry())
    # A Claude run first, then a Codex run: one file, two clients.
    assert "[gk-example-tests]" in injected_text(hook.run(shell("pytest -q", session="claude-s")))
    out = codex_run(monkeypatch, capsys, shell("pytest -q", session="codex-s"))
    assert "[gk-example-tests]" in injected_text(out)
    events = hook.events()
    assert [e["client"] for e in events] == ["claude", "codex"]
    assert [e["session"] for e in events] == ["claude-s", "codex-s"]
    assert sorted(p.name for p in (hook.home / "state").iterdir()) == ["claude-s.json", "codex-s.json"]
    assert "RECALL_HOOK_CLIENT" not in os.environ

    # The shared switch lives on the Claude side and governs Codex too.
    hook.config(enabled=False)
    assert codex_run(monkeypatch, capsys, shell("pytest -q", session="codex-t")) == ""
    assert len(hook.events()) == 2

    # A value that was there before the run is put back, not removed.
    hook.config(enabled=True)
    monkeypatch.setenv("RECALL_HOOK_CLIENT", "preset")
    codex_run(monkeypatch, capsys, shell("pytest -q", session="codex-u"))
    assert os.environ["RECALL_HOOK_CLIENT"] == "preset"
    assert hook.events()[-1]["client"] == "codex"


def test_codex_does_not_handle_the_failure_event(hook, tmp_path, monkeypatch, capsys):
    """Invariant: Codex has no post-tool event, so its adapter does not accept `post-tool-use-failure`:
    it exits 0 with no output and no log line even for a matching failure.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_codex_does_not_handle_the_failure_event`.
    Mutation: add `"post-tool-use-failure"` to the event set in `recall_hooks.codex.main`. Symbol:
    `recall_hooks.codex.main`. Seen: `assert '{"hookSpecif...ry here."}}\n' == ''`.
    """
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    hook.index(example_entry())
    assert codex_run(monkeypatch, capsys, failure(), FAILURE_EVENT) == ""
    assert hook.events() == []
