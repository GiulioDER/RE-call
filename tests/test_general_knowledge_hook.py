"""General knowledge hook, tasks 1 and 2: the index model, its fail-open reader, project detection and trigger matching.

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
    Mutation: `_unquote` reads `m.group(1)` instead of `m.group(m.lastindex or 3)`, so a single quoted
    or unquoted word yields `None` to `"".join`. Symbol: `_unquote`. Seen: `IndexError: list index out of range` at `_program`, on every param.
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
    """Invariant: a hand-built entry with an uncached regex is a non-match, not an exception."""
    entry = gk.Entry("gk-example-x", "procedure", "global", (gk.Trigger("command_regex", "never-cached-\\d"),), (), "b")
    assert gk.match_triggers([entry], "never-cached-1", None) == []


def test_project_scope_filters_entries(tmp_path):
    """Invariant: `global` is always in scope; `project:<name>` only for exactly that project.

    Red proof. Node: `tests/test_general_knowledge_hook.py::test_project_scope_filters_entries`.
    Mutations, symbol `in_scope`. A: always `True` (`assert ['gk-example-...xample-alpha'] == ['gk-example-global']`). B: compare `startswith` instead of equality,
    so `project:alp` also fits `alpha` is not tested; instead B drops the `project is not None` guard
    and the prefix check, comparing `entry.scope[8:] == project` (`assert not True` on the `projekt:alpha` entry).
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
