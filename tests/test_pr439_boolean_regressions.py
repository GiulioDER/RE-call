"""DAT-004: serialized benchmark outcomes retain their boolean meaning.

The rejection cases must fail with DID NOT RAISE on baseline 8ef55caf0 in
SessionRecord.from_mapping, whose bool(...) calls turn malformed outcomes into
accepted records. The exact same nodes must pass after raw-value validation.
The round-trip controls also require a production mutation reversing either
`success=success` or `abstained=abstained` to fail their identity assertions.
The absent-field control requires changing the production abstained default
from False to True to fail its identity assertion. Execution receipts are
recorded by the parent audit on the remote test host, not inferred from these expectations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.agent_ab import RECALL_ON, SessionRecord, read_jsonl, write_jsonl


INVALID_OUTCOMES = [
    pytest.param("false", id="false-string"),
    pytest.param("true", id="true-string"),
    pytest.param("", id="empty-string"),
    pytest.param(0, id="zero"),
    pytest.param(1, id="one"),
    pytest.param(0.5, id="fraction"),
    pytest.param(None, id="null"),
]


def _raw_record() -> dict[str, Any]:
    return {"task_id": "boolean-outcome", "variant": RECALL_ON, "success": False}


@pytest.mark.parametrize("field", ["success", "abstained"])
@pytest.mark.parametrize("invalid", INVALID_OUTCOMES)
def test_from_mapping_rejects_nonboolean_outcomes(field: str, invalid: Any) -> None:
    """Raw runner mappings cannot silently change failure or abstention labels."""
    raw = _raw_record()
    raw[field] = invalid

    with pytest.raises(TypeError, match=rf"^{field} must be a bool$"):
        SessionRecord.from_mapping(raw)


@pytest.mark.parametrize("field", ["success", "abstained"])
@pytest.mark.parametrize("invalid", INVALID_OUTCOMES)
def test_read_jsonl_rejects_nonboolean_outcomes(
    tmp_path: Path, field: str, invalid: Any
) -> None:
    """The public JSONL boundary enforces the same outcome contract as mappings."""
    raw = _raw_record()
    raw[field] = invalid
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    with pytest.raises(TypeError, match=rf"^{field} must be a bool$"):
        read_jsonl(path)


@pytest.mark.parametrize("success", [False, True], ids=["failure", "success"])
@pytest.mark.parametrize("abstained", [False, True], ids=["answered", "abstained"])
def test_round_trip_preserves_boolean_outcomes(
    tmp_path: Path, success: bool, abstained: bool
) -> None:
    """Reversing either production outcome assignment must fail these assertions."""
    original = SessionRecord(
        task_id="boolean-outcome",
        variant=RECALL_ON,
        success=success,
        abstained=abstained,
        input_tokens=None,
        output_tokens=0,
    )
    mapping_record = SessionRecord.from_mapping(original.to_dict())
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [original])
    [jsonl_record] = read_jsonl(path)

    for restored in (mapping_record, jsonl_record):
        assert restored.success is success
        assert restored.abstained is abstained
        assert restored == original


def test_absent_abstained_defaults_to_false_at_both_boundaries(tmp_path: Path) -> None:
    """Changing the production missing-field default to True must fail this control."""
    raw = _raw_record()
    mapping_record = SessionRecord.from_mapping(raw)
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    [jsonl_record] = read_jsonl(path)

    assert mapping_record.abstained is False
    assert jsonl_record.abstained is False
    assert mapping_record.success is False
    assert jsonl_record.success is False
