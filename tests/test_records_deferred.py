"""Deferred CCA fixes for PR 977 (the records API), no database.

Invariants and the failure each one catches:
- D1 a nested union that matches no branch (an anchor's `code_sha`) is reported at its own field,
  not as an unknown record kind (CCA BUG-003).
- D2 an integral float is an `integer`, as draft 2020-12 and `jsonschema` read it (CCA BUG-004).
- D3 a memo that declares `recall_record` twice has its record refused, never the last line kept
  in silence (CCA BUG-005).
- D4 `recall records --anchors` accepts a file with a byte order mark, and refuses a file that is
  not UTF-8 with a message instead of a traceback (CCA BUG-006).
- D5 `caused_by` keeps one chunk per related record, so a long memo cannot crowd another linked
  record out of `max_items` (CCA BUG-007).
- D6 a successor keyed by its absolute source (a path two roots hold) is mapped to its
  root-relative file before the source policy decides on it, and reported as its graph key (the
  absolute source); one it cannot map is still withheld (CCA BUG-008).
- D7 the `recall_records` tool description states the limits and kinds the code enforces
  (CCA CODE-009). A drift guard, not a behaviour fix: it holds on the pre-fix code too.
- D8 the `recall_records` tool refuses a query with neither `anchors` nor `ids` without naming
  `all_records`, a parameter the tool does not take (CCA DOC-006).
- D9 the supersession graph is not asked about a candidate the anchor query discards
  (CCA PERF-005).
- D10 the tool serialises its envelope off the event loop thread, and the bytes are those of
  `json.dumps(envelope, ensure_ascii=False, indent=2)` (CCA PERF-006).

Node IDs, all in `tests/test_records_deferred.py`:
D1 `test_d1_a_nested_union_names_its_field`,
D2 `test_d2_an_integral_float_is_an_integer`,
D3 `test_d3_a_record_declared_twice_is_refused`,
D4 `test_d4_anchors_file_with_a_bom_is_read`, `test_d4_anchors_file_not_utf8_is_refused`,
D5 `test_d5_caused_by_keeps_one_chunk_per_record`,
D6 `test_d6_a_collided_successor_is_decided_as_its_file`,
D7 `test_d7_the_tool_description_states_the_enforced_limits`,
D8 `test_d8_the_tool_refusal_does_not_name_all_records`,
D9 `test_d9_a_discarded_candidate_is_not_resolved`,
D10 `test_d10_the_envelope_is_serialised_off_the_loop`,
D11 `test_d11_two_memos_sharing_a_record_id_both_come_back` (BUG-007 rework: the caused_by dedup
  groups by memo, so two memos sharing a record id both come back; red by mutation, grouping by
  record id again fails "two memos sharing a record id came back as one").

Red proof, 2026-10-28, on this workstation: this file run against a `git archive` of the pre-fix
tree (HEAD `06c1603a9`), each behaviour test failing in its own assertion, then green on the fix:
D1 "the refusal blamed the wrong field: 'anchors[0].code_sha: kind None is not a record kind'";
D2 "an ordinal of 1.0 was refused: 'ordinal: expected integer, got float'";
D3 "a twice-declared record was read as {...}" (the second line kept);
D4 "a file with a byte order mark was refused" and "a file that is not UTF-8 crashed
('UnicodeDecodeError')";
D5 "a long memo crowded another linked record out: ['A', 'A', 'A']";
D6 "a readable collided successor was reported as '<withheld>'";
D8 "the tool refused with 'give exactly one of anchors, ids or all_records'";
D9 "the supersession graph was asked about ['kept', 'discarded']";
D10 "the envelope was serialised on the event loop thread".
D7 passes on the pre-fix tree by design; mutating `recall.records.MAX_QUERY_ANCHORS` to 250 fails
it: "the recall_records description does not state ['250']".
"""

from __future__ import annotations

import copy
import json
import threading
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from recall.calibration import Calibration
from recall.document import parse_document
from recall.records import (
    MAX_QUERY_ANCHORS,
    MAX_RECORDS,
    RECORD_KINDS,
    WITHHELD,
    RecordCandidate,
    RecordsQueryError,
    lookup_records,
    match_candidates,
    record_problem,
    record_schema,
)
from recall.related import trusted_related
from recall.supersession import SupersessionGraph
from recall.trust_policy import TrustPolicy
from recall.types import Chunk
from tests.audit_memory_support import EXPECTED, MEMOS, REPO
from tests.test_lite_mcp import _serve, lite_env  # noqa: F401  # the fixture is used by name

BUG = EXPECTED["RE-call__0123456789ab__BUG-001.md"]
NOW = datetime(2026, 11, 1, tzinfo=UTC)


def test_d1_a_nested_union_names_its_field() -> None:
    record = copy.deepcopy(BUG)
    record["anchors"][0]["code_sha"] = "XYZ"
    problem = record_problem(record) or ""
    assert "code_sha" in problem and "kind None" not in problem, f"the refusal blamed the wrong field: {problem!r}"


def test_d2_an_integral_float_is_an_integer() -> None:
    record = {**copy.deepcopy(BUG), "ordinal": 1.0}
    problem = record_problem(record)
    assert problem is None, f"an ordinal of 1.0 was refused: {problem!r}"
    jsonschema = pytest.importorskip("jsonschema")
    agrees = not list(jsonschema.Draft202012Validator(record_schema()).iter_errors(record))
    assert agrees, "jsonschema refuses the record this validator admits"


def test_d3_a_record_declared_twice_is_refused() -> None:
    from recall.records import admit_record

    text = (MEMOS / "RE-call__0123456789ab__BUG-001.md").read_text(encoding="utf-8")
    line = next(row for row in text.split("\n") if row.startswith("recall_record:"))
    other = line.replace(BUG["id"], BUG["id"] + "-other")
    twice = text.replace(line + "\n", line + "\n" + other + "\n", 1)
    meta = parse_document(twice).meta
    problem = record_problem(meta.get("recall_record")) or ""
    assert "more than once" in problem, f"a twice-declared record was read as {meta.get('recall_record')!r}"
    assert "recall_record" not in admit_record(meta, "twice.md"), "a twice-declared record was admitted"


def _read(path: Path) -> tuple[str, object]:
    from recall.cli_commands.records_cmd import _read_anchors

    try:
        return "ok", _read_anchors(str(path))
    except SystemExit as exc:
        return "refused", str(exc)
    except Exception as exc:  # noqa: BLE001  # which exception escaped is what is asserted on
        return "crashed", type(exc).__name__


def test_d4_anchors_file_with_a_bom_is_read(tmp_path: Path) -> None:
    anchors = [{"file": "a.py", "symbol": None, "code_sha": None}]
    path = tmp_path / "anchors.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(anchors).encode("utf-8"))
    outcome, value = _read(path)
    assert (outcome, value) == ("ok", anchors), f"a file with a byte order mark was {outcome}: {value!r}"


def test_d4_anchors_file_not_utf8_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "anchors.json"
    path.write_bytes(b'[{"file": "caf\xe9.py"}]')
    outcome, value = _read(path)
    assert outcome == "refused", f"a file that is not UTF-8 {outcome} ({value!r}) instead of a clean refusal"
    assert "cannot read anchors" in str(value)


def _memo_chunk(chunk_id: str, source: str, ordinal: int, record: dict[str, Any]) -> Chunk:
    return Chunk(chunk_id, source, f"text of {chunk_id}", {"file": source, "ord": ordinal, "recall_record": record})


class _RelatedStore:
    """The seed and its candidates, answered as a store's bounded `related_chunks` answers them."""

    tenant = "t"

    def __init__(self, seed: Chunk, candidates: list[Chunk]) -> None:
        self.seed, self.candidates = seed, candidates

    def iter_chunks(self, batch_size: int = 1000) -> list[Chunk]:
        return [self.seed, *self.candidates]

    def related_chunks(self, seed_chunk_id: str, relation: str, max_items: int) -> tuple[Chunk, list[Chunk]]:
        return self.seed, list(self.candidates)

    def supersession_all(self) -> tuple[dict[str, str], frozenset[str], dict[str, Any]]:
        return {}, frozenset(), {}


def test_d5_caused_by_keeps_one_chunk_per_record() -> None:
    seed = _memo_chunk("s3", "seed.md", 3, {"id": "S", "caused_by": ["A", "B"]})
    long_memo = [_memo_chunk(f"a{i}", "a.md", i, {"id": "A"}) for i in range(6)]
    short_memo = [_memo_chunk("b0", "b.md", 0, {"id": "B"})]
    result = trusted_related(
        _RelatedStore(seed, [*long_memo, *short_memo]),
        "s3",
        relation="caused_by",
        max_items=3,
        calibration=Calibration(embedder="fixture", threshold=0.5),
        policy=TrustPolicy.development(),
        now=NOW,
    )
    records = [item.chunk.metadata["recall_record"]["id"] for item in result.items]
    assert "B" in records, f"a long memo crowded another linked record out: {records}"
    assert records.count("A") == 1, f"one record came back as several chunks: {records}"


def test_d11_two_memos_sharing_a_record_id_both_come_back() -> None:
    """BUG-007 rework after the regression review: the caused_by dedup groups by memo (source),
    not by record id, so two memos that share one id (an authoring error) are both returned and a
    superseded twin cannot stand in for the current one. Red by mutation of the fix (group by
    record id again): "two memos sharing a record id came back as one"."""
    seed = _memo_chunk("s0", "seed.md", 0, {"id": "S", "caused_by": ["A"]})
    twins = [_memo_chunk("x0", "old.md", 0, {"id": "A"}), _memo_chunk("y0", "new.md", 0, {"id": "A"})]
    result = trusted_related(
        _RelatedStore(seed, twins),
        "s0",
        relation="caused_by",
        max_items=3,
        calibration=Calibration(embedder="fixture", threshold=0.5),
        policy=TrustPolicy.development(),
        now=NOW,
    )
    sources = sorted(item.chunk.source for item in result.items)
    both = sources == ["new.md", "old.md"]
    assert both, f"two memos sharing a record id came back as one: {sources}"


class _CollidedStore:
    """Two roots' `README.md`, the first superseded by the second: the graph keys both by source."""

    tenant = "t"

    def __init__(self, successor: str) -> None:
        self.chunks = [
            Chunk("r1", "/one/README.md", "", {"file": "README.md", "recall_record": {**BUG, "id": "R1"}}),
            Chunk("r2", "/two/README.md", "", {"file": "README.md", "recall_record": {**BUG, "id": "R2"}}),
        ]
        self.graph = SupersessionGraph(edges={"/one/README.md": successor}, collided=frozenset({"README.md"}))

    def record_chunks(self, code_files: list[str] | None = None) -> list[Chunk]:
        return list(self.chunks)

    def supersession_graph(self) -> SupersessionGraph:
        return self.graph


class _RelativeOnly:
    """A source policy written, as policies are, against root-relative files."""

    def decide(self, file: str, context: object) -> Any:
        return types.SimpleNamespace(allowed=not file.startswith("/"))


def test_d6_a_collided_successor_is_decided_as_its_file() -> None:
    def successor(store: _CollidedStore) -> object:
        found = lookup_records(
            store, ids=["R1"], security_policy=_RelativeOnly(), access_context=object(), now=NOW  # type: ignore[arg-type]
        )
        return found["records"][0]["superseded_by"]

    mapped = successor(_CollidedStore("/two/README.md"))
    assert mapped == "/two/README.md", f"a readable collided successor was reported as {mapped!r}"
    # Not red on the pre-fix code, which withheld every collided successor: a key no chunk maps
    # must still fail closed.
    unmapped = successor(_CollidedStore("/three/README.md"))
    assert unmapped == WITHHELD, f"an unmappable successor was reported as {unmapped!r}"


def test_d7_the_tool_description_states_the_enforced_limits() -> None:
    import asyncio

    from recall_mcp.server import build_server

    tools = {tool.name: tool for tool in asyncio.run(build_server().list_tools())}
    description = tools["recall_records"].description or ""
    missing = [
        value for value in (str(MAX_QUERY_ANCHORS), str(MAX_RECORDS), *sorted(RECORD_KINDS)) if value not in description
    ]
    assert not missing, f"the recall_records description does not state {missing}"


def test_d8_the_tool_refusal_does_not_name_all_records() -> None:
    from recall_mcp.related_api import records_memory

    message = ""
    try:
        records_memory(object())  # type: ignore[arg-type]
    except RecordsQueryError as exc:
        message = str(exc)
    assert "anchors" in message and "all_records" not in message, f"the tool refused with {message!r}"


def test_d9_a_discarded_candidate_is_not_resolved() -> None:
    def candidate(record_id: str, file: str) -> RecordCandidate:
        anchor = {"file": file, "symbol": None, "code_sha": None}
        record = {"id": record_id, "repo": REPO, "anchors": [anchor]}
        return RecordCandidate(record_id, record, "bug", [anchor], {}, file, file, None, [])

    asked: list[str] = []

    def superseded_by(found: RecordCandidate) -> str | None:
        asked.append(found.id)
        return None

    match_candidates(
        [candidate("kept", "a.py"), candidate("discarded", "b.py")],
        query_anchors=[{"file": "a.py", "symbol": None, "code_sha": None}],
        repo=REPO,
        wanted_ids=None,
        wanted_kinds=None,
        superseded_by=superseded_by,
        now=NOW,
    )
    assert asked == ["kept"], f"the supersession graph was asked about {asked}"


def test_d10_the_envelope_is_serialised_off_the_loop(
    lite_env: dict[str, str], monkeypatch: pytest.MonkeyPatch  # noqa: F811  # the imported fixture
) -> None:
    from recall_mcp import server as srv

    envelope = {"tenant_id": "t", "generation_id": "g", "truncated": False, "records": [{"name": "Ünïcødé ✅"}]}
    monkeypatch.setattr(srv, "records_memory", lambda store, **kwargs: envelope)
    threads: list[int] = []
    real_dumps = json.dumps

    def dumps(value: Any, **kwargs: Any) -> str:
        if value is envelope:
            threads.append(threading.get_ident())
        return real_dumps(value, **kwargs)

    monkeypatch.setattr(srv, "json", types.SimpleNamespace(**{**vars(json), "dumps": dumps}))

    async def body(_state: dict[str, Any], call: Any) -> tuple[str, int]:
        return await call("recall_records", ids=["x"]), threading.get_ident()

    out, loop_thread = _serve(lite_env, body)
    assert out == real_dumps(envelope, ensure_ascii=False, indent=2), f"the tool's output bytes changed: {out[:200]!r}"
    assert threads and loop_thread not in threads, "the envelope was serialised on the event loop thread"
