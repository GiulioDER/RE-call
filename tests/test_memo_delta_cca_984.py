"""Public memo-loader regression for PR984 BUG-005 / FIX-006.

Node: tests/test_memo_delta_cca_984.py::test_corrupt_utf8_delta_keeps_valid_memos
Target: recall.records_files._read_cache_file, optional delta-journal decoding.
Baseline: e6d9ee13a52292ec7d63feda471235b095fb7240.
Predicted RED: the caught UnicodeDecodeError fails the explicit availability assertion.
Observed 2026-10-26 on the isolated testbench: intended assertion RED on the baseline above, then the same case GREEN in the 199-test run (zero failures, errors or skips).
"""

from __future__ import annotations

import json
from pathlib import Path

from recall.records_files import load_memo_records
from tests.audit_memory_support import EXPECTED, copy_memos


def test_corrupt_utf8_delta_keeps_valid_memos(tmp_path: Path) -> None:
    """An optional corrupt journal cannot block admitted memos or survive warm recovery."""
    memos = copy_memos(tmp_path / "memos")
    cache = tmp_path / "cache" / "memos.json"
    first = load_memo_records(memos, cache_file=cache)
    expected_ids = {record["id"] for record in EXPECTED.values()}
    assert expected_ids, "the admitted-memo control must not be empty"
    assert {memo.record["id"] for memo in first} == expected_ids
    assert cache.is_file(), "the control must establish a valid persisted base cache"

    delta = cache.with_name(cache.name + ".delta")
    delta.write_bytes(b"\xff\xfe\n")
    recovered = None
    decode_error: UnicodeDecodeError | None = None
    try:
        # No deadline: this is the ordinary warm load that folds or repairs the journal.
        recovered = load_memo_records(memos, cache_file=cache)
    except UnicodeDecodeError as exc:
        decode_error = exc

    assert decode_error is None, "an optional corrupt delta blocked valid memo records"
    assert recovered is not None
    assert {memo.record["id"] for memo in recovered} == expected_ids
    assert recovered.skipped == first.skipped

    if delta.exists():
        replacement = None
        try:
            replacement = delta.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            pass
        assert replacement is not None, "ordinary warm loading retained the corrupt journal"
        for line in replacement.splitlines():
            if line.strip():
                assert isinstance(json.loads(line), dict), "replacement journal is not JSONL"
