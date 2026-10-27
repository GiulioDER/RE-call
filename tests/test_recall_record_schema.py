"""The `recall_record` v1 schema and the golden audit memos agree, and the schema refuses drift.

Contracts C1 and C4 of ``docs/AUDIT_MEMORY_PLAN.md``. The golden memos are the exact bytes
cca-audit must render; ``expected_records.json`` is what RE-call must return for each.

Red proof (2026-10-21, `recall/schemas/recall_record.v1.json`, run on the deployment host inside
the test bounds):
- every ``additionalProperties: false`` set to true: `test_unknown_field_is_invalid` failed (the
  record validated).
- ``minItems: 3`` removed from lesson ``members``: `test_lesson_with_two_members_is_invalid`
  failed.
"""
from __future__ import annotations

import copy
import json
from importlib import resources
from pathlib import Path

import jsonschema
import pytest

from recall.frontmatter import parse_frontmatter

FIXTURES = Path(__file__).parent / "fixtures" / "audit_memory"
EXPECTED: dict[str, dict] = json.loads(
    (FIXTURES / "expected_records.json").read_text(encoding="utf-8")
)


def _schema() -> dict:
    text = resources.files("recall.schemas").joinpath("recall_record.v1.json").read_text(
        encoding="utf-8"
    )
    return json.loads(text)


@pytest.fixture(scope="module")
def validator() -> jsonschema.Draft202012Validator:
    schema = _schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _by_kind(kind: str) -> dict:
    return copy.deepcopy(next(r for r in EXPECTED.values() if r["kind"] == kind))


def test_schema_ships_as_package_data() -> None:
    assert _schema()["title"] == "recall_record v1"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_expected_record_validates(
    validator: jsonschema.Draft202012Validator, name: str
) -> None:
    assert list(validator.iter_errors(EXPECTED[name])) == []


def test_every_kind_is_covered() -> None:
    assert {r["kind"] for r in EXPECTED.values()} == {
        "bug", "false_positive", "lesson", "false_positive_retraction", "lesson_retraction",
    }


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_golden_memo_carries_its_record_on_one_canonical_line(name: str) -> None:
    text = (FIXTURES / "memos" / name).read_bytes().decode("utf-8")
    lines = [line for line in text.split("\n") if line.startswith("recall_record: ")]
    assert len(lines) == 1
    raw = lines[0].removeprefix("recall_record: ")
    assert json.loads(raw) == EXPECTED[name]
    assert raw == json.dumps(
        EXPECTED[name], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_existing_parser_reads_the_validity_keys(name: str) -> None:
    meta, _ = parse_frontmatter((FIXTURES / "memos" / name).read_text(encoding="utf-8"))
    assert meta["valid_from"] == EXPECTED[name]["date"]
    target = EXPECTED[name].get("target_id") or EXPECTED[name].get("supersedes")
    if target:
        assert meta["supersedes"] == target.replace(":", "__")
    else:
        assert "supersedes" not in meta


def test_hard_cases_are_present() -> None:
    texts = [json.dumps(r, ensure_ascii=False) for r in EXPECTED.values()]
    assert any(len(t.encode("utf-8")) > 10_000 for t in texts)
    assert any("\\n---\\n" in t and "日本語" in t for t in texts)
    assert any(a["symbol"] is None for r in EXPECTED.values() for a in r.get("anchors", []))


def test_unknown_field_is_invalid(validator: jsonschema.Draft202012Validator) -> None:
    record = _by_kind("bug")
    record["status"] = "open"
    assert list(validator.iter_errors(record))


def test_false_positive_with_null_code_sha_is_invalid(validator: jsonschema.Draft202012Validator) -> None:
    record = _by_kind("false_positive")
    record["anchors"][0]["code_sha"] = None
    assert list(validator.iter_errors(record))


def test_lesson_with_two_members_is_invalid(validator: jsonschema.Draft202012Validator) -> None:
    record = _by_kind("lesson")
    record["members"] = record["members"][:2]
    assert list(validator.iter_errors(record))


def test_windows_path_in_anchor_is_invalid(validator: jsonschema.Draft202012Validator) -> None:
    record = _by_kind("bug")
    record["anchors"][0]["file"] = "recall\\pool.py"
    assert list(validator.iter_errors(record))
