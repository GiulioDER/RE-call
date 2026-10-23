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
