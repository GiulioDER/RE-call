"""An empty validity window (`valid_from` after `valid_until`) is invalid metadata, everywhere.

The validity-frontmatter spec 1.0 (github.com/GiulioDER/validity-frontmatter, section 5.2 and test
vector 4) makes such a document `invalid_metadata`. RE-call reported `expired` or `not_yet_valid`
instead, the divergence its appendix note D records, open since 2026-08-19 and re-measured on
master `52be498cc` on 2026-10-22 (2026-10-21 supersession audit). `recall.frontmatter.validity_bounds`
now refuses the window, so every reader that already treats a malformed date as invalid agrees:

- E1 the trust gate verdicts it `invalid_metadata` (spec vector 4);
- E2 a one-day window (both keys the same date) stays valid all day, on the gate and on its card,
  because the comparison is made on the bounds, `valid_until` running to the end of its day;
- E3 `recall lint` reports it, as `invalid-date`;
- E4 the indexer refuses it as it refuses a malformed date, through a lite refresh and (E5) on
  PostgreSQL, so what lint calls an error is what `recall index` refuses;
- E6 the writers refuse to produce it: the lite assistant memo writer, and (in
  `tests/test_rewrite_routing_contract.py`) `recall rewrite`, since one such memo would now stop a
  refresh of the whole corpus;
- E7 graph expansion rejects the neighbour as invalid, not as expired.

Measured before the change, 2026-10-22: no memo in the local memory stores or `docs/` holds an
empty window (2,270 files with frontmatter, 0 empty, 0 malformed), so the indexer's new refusal
changes nothing already indexed. Re-measure with a scan of `~/.claude/projects/*/memory/**/*.md`
through `validity_bounds`.

Red proof (2026-10-22): each test's docstring names its baseline or mutation. "Master" means
`52be498cc` with this file copied in.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import pytest

from recall.embeddings import HashingEmbedder
from recall.frontmatter import validity_bounds
from recall.index import Indexer
from recall.lint import lint_corpus
from recall.lite import LiteStore
from recall_mcp.assistant import AssistantMemos
from recall.lite.refresh import Outcome, refresh
from recall.store import PgVectorStore
from recall.trust_gate import evaluate
from recall_mcp.graph_expansion import temporal_bounds
from tests.conftest import requires_db
from tests.fact_fixtures import TEA, evidence_card
from tests.test_lite_calibration import DIM, ContentWordEmbedder, _write_memos
from tests.test_trust import CAL, _hit, _result

EMPTY = {"valid_from": "2026-12-01", "valid_until": "2026-11-01"}
ONE_DAY = {"valid_from": "2026-10-22", "valid_until": "2026-10-22"}


def _memo(meta: dict[str, str]) -> str:
    return "---\n" + "".join(f"{key}: {value}\n" for key, value in meta.items()) + "---\nThe limit is 100.\n"


def test_e1_an_empty_window_is_invalid_metadata_spec_vector_4() -> None:
    """Node tests/test_validity_empty_window.py::test_e1_an_empty_window_is_invalid_metadata_spec_vector_4.

    Spec 1.0 test vector 4: `d.md`, `valid_from` 2026-12-01, `valid_until` 2026-11-01, evaluated at
    2026-08-19. Red against master: fails on "an empty window read as not_yet_valid".
    """
    at = datetime(2026, 8, 19, tzinfo=UTC)
    res = evaluate(_result([_hit("d", "d.md", 0.9, **EMPTY)]), {}, CAL, at)
    assert res.hits[0].verdict == "invalid_metadata", f"an empty window read as {res.hits[0].verdict}"
    assert res.abstained is True


def test_e2_a_one_day_window_stays_valid_all_day() -> None:
    """Node tests/test_validity_empty_window.py::test_e2_a_one_day_window_stays_valid_all_day.

    Green on master (the control for the bound E1 adds). Red, mutation of `validity_bounds`
    (recall/frontmatter.py) reading `valid_until` as midnight (`_parse_date(...)` in place of the
    `time.max` combine): fails on its first assertion, `start < end`, the two bounds being equal.
    """
    start, end = validity_bounds(ONE_DAY)
    assert start is not None and end is not None and start < end
    late = datetime(2026, 10, 22, 23, tzinfo=UTC)
    res = evaluate(_result([_hit("o", "o.md", 0.9, **ONE_DAY)]), {}, CAL, late)
    assert res.hits[0].verdict == "ok", f"a one-day window was not ok at 23:00: {res.hits[0].verdict}"
    # Its evidence card, which refuses a window whose end is not after its start, accepts it.
    card = replace(
        evidence_card(TEA, tenant="t", generation="g", chunk="c", source="o.md"),
        card_id="", valid_from=start, valid_until=end,
    )
    assert card.valid_until == end


def test_e3_lint_reports_an_empty_window(tmp_path: Path) -> None:
    """Node tests/test_validity_empty_window.py::test_e3_lint_reports_an_empty_window.

    Red against master: fails on "lint did not report an empty window" (no issue at all).
    """
    (tmp_path / "d.md").write_text(_memo(EMPTY), encoding="utf-8")
    (tmp_path / "o.md").write_text(_memo(ONE_DAY), encoding="utf-8")
    issues = lint_corpus(tmp_path)
    assert [(i.file, i.code, i.level) for i in issues] == [("d.md", "invalid-date", "error")], (
        f"lint did not report an empty window: {issues}"
    )
    assert "empty validity window" in issues[0].message


@pytest.mark.parametrize("bad", [EMPTY, {"valid_until": "soonish"}], ids=["empty window", "malformed date"])
def test_e4_a_lite_refresh_refuses_it_as_it_refuses_a_malformed_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: dict[str, str]
) -> None:
    """Node tests/test_validity_empty_window.py::test_e4_a_lite_refresh_refuses_it_as_it_refuses_a_malformed_date.

    The indexer's fail-fast (`Indexer.index_path`), reached the way a lite user reaches it. The
    malformed-date case is the existing behaviour, green on master. Red against master, the empty
    window case: fails on "the refresh indexed a memo the trust gate cannot use" (REFRESHED).
    """
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    folder = tmp_path / "memory"
    _write_memos(folder, 2)
    (folder / "d.md").write_text(_memo(bad), encoding="utf-8")
    with LiteStore(tmp_path / ".recall" / "memory.db", dim=DIM) as store:
        report = refresh(store, [folder], ContentWordEmbedder())  # type: ignore[arg-type]
    assert report.outcome is Outcome.FAILED, f"the refresh indexed a memo the trust gate cannot use: {report.to_dict()}"
    assert "d.md" in (report.reason or "")


@requires_db
def test_e5_index_path_refuses_an_empty_window_with_its_filename(
    tmp_path: Path, make_store: Callable[[int], PgVectorStore]
) -> None:
    """Node tests/test_validity_empty_window.py::test_e5_index_path_refuses_an_empty_window_with_its_filename.

    As `tests/test_index.py::test_index_path_malformed_date_raises_with_filename`, for the empty
    window. Red against master: fails on "DID NOT RAISE".
    """
    (tmp_path / "d.md").write_text(_memo(EMPTY), encoding="utf-8")
    with pytest.raises(ValueError, match="d.md.*empty validity window"):
        Indexer(make_store(64), HashingEmbedder(dim=64)).index_path(tmp_path)


def test_e6_the_assistant_memo_writer_refuses_an_empty_window(tmp_path: Path) -> None:
    """Node tests/test_validity_empty_window.py::test_e6_the_assistant_memo_writer_refuses_an_empty_window.

    `recall_remember` on a lite store writes through `AssistantMemos.write`, and the next refresh
    indexes the folder, so a memo it wrote with an empty window would now fail that refresh. Red
    against master: fails on "the writer wrote an empty window" (it returned a memo id). Red,
    mutation removing the `validity_bounds` call from `MemoFolder.write`: the same line.
    """
    memos = AssistantMemos(tmp_path / "memos", index_root=tmp_path)
    refused = memos.write("The limit is 100.", kind="fact", **EMPTY)
    assert "refused" in refused, f"the writer wrote an empty window: {refused}"
    assert "empty validity window" in str(refused["refused"])
    assert not list((tmp_path / "memos").glob("*.md"))
    assert "memo_id" in memos.write("The limit is 100.", kind="fact", **ONE_DAY)


def test_e7_graph_expansion_rejects_an_empty_window_as_invalid() -> None:
    """Node tests/test_validity_empty_window.py::test_e7_graph_expansion_rejects_an_empty_window_as_invalid.

    `temporal_bounds` raising is what makes the expansion reject the neighbour as
    `invalid_temporal_metadata`. Red against master: fails on "DID NOT RAISE" (it returned the
    bounds, and the neighbour was rejected as not yet valid). With `validity_bounds` fixed and the
    new check in `temporal_bounds` removed, still red on the same line: its fallback re-read the
    bare dates at midnight and returned them inverted.
    """
    with pytest.raises(ValueError, match="empty validity window"):
        temporal_bounds(EMPTY["valid_from"], EMPTY["valid_until"])
    assert temporal_bounds(ONE_DAY["valid_from"], ONE_DAY["valid_until"]) == validity_bounds(ONE_DAY)
