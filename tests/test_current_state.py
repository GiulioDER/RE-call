from datetime import UTC, datetime

import pytest

from recall.current_state import project_current_state
from recall.store import resolve_supersession_candidates
from recall.types import Chunk


NOW = datetime(2026, 8, 24, tzinfo=UTC)


class Store:
    tenant = "tenant-a"
    generation_id = "gen-1"

    def __init__(self, chunks, candidates=(), unresolved=frozenset()):
        self._chunks = chunks
        self._candidates = dict(candidates)
        # Parameterised because every fake store in this suite used to hardcode an empty set,
        # which made `unresolved_supersession_reference` unreachable by construction. See the
        # two tests at the end of this file.
        self._unresolved = frozenset(unresolved)

    def iter_chunks(self, batch_size=1000):
        del batch_size
        return iter(self._chunks)

    def supersession_all(self):
        return {}, self._unresolved, self._candidates


def test_current_state_is_deterministic_and_generation_bound() -> None:
    store = Store(
        [
            Chunk("a", "a.md", "old", {"file": "a.md"}),
            Chunk("b", "b.md", "new", {"file": "b.md"}),
        ],
        {"a.md": [("b.md", NOW)]},
    )
    first = project_current_state(store, as_of=NOW)
    second = project_current_state(store, as_of=NOW)
    assert first == second
    assert first.generation_id == "gen-1"
    states = {record.source: record.state for record in first.records}
    assert states == {"a.md": "superseded", "b.md": "current"}


def test_current_state_fails_closed_on_multiple_live_successors() -> None:
    store = Store(
        [Chunk("a", "a.md", "old", {"file": "a.md"})],
        {"a.md": [("b.md", NOW), ("c.md", NOW)]},
    )
    record = project_current_state(store, as_of=NOW).records[0]
    assert record.state == "ambiguous"
    assert "multiple_live_successors" in record.diagnostics


def test_current_state_reports_validity_window_states() -> None:
    store = Store(
        [
            Chunk("future", "future.md", "future", {"file": "future.md", "valid_from": "2027-01-01"}),
            Chunk("past", "past.md", "past", {"file": "past.md", "valid_until": "2026-01-01"}),
        ]
    )
    states = {record.source: record.state for record in project_current_state(store, as_of=NOW).records}
    assert states == {"future.md": "not_yet_valid", "past.md": "expired"}


def test_current_state_fails_closed_on_malformed_supersession_metadata() -> None:
    store = Store([Chunk("bad", "bad.md", "bad", {"file": "bad.md", "supersedes": 7})])
    record = project_current_state(store, as_of=NOW).records[0]
    assert record.state == "invalid"
    assert "malformed_supersession_metadata" in record.diagnostics


def test_current_state_serving_bound_fails_closed_before_assembling_more_records() -> None:
    store = Store(
        [
            Chunk("a", "a.md", "a", {"file": "a.md"}),
            Chunk("b", "b.md", "b", {"file": "b.md"}),
        ]
    )
    with pytest.raises(ValueError, match="exceeds max_records"):
        project_current_state(store, as_of=NOW, max_records=1)


def test_current_state_rejects_an_unbounded_serving_limit() -> None:
    with pytest.raises(ValueError, match="<= 1000"):
        project_current_state(Store([]), as_of=NOW, max_records=1001)


def test_dependency_invalidation_never_masks_a_fail_closed_state() -> None:
    """A fail-closed verdict outranks `dependency_invalidated`, and the diagnostic still fires.

    `ambiguous` and `invalid` mean "this projection cannot be trusted about this source at all".
    Replacing either with `dependency_invalidated` swaps an admission of ignorance for a
    confident, specific explanation the reader will act on, and it silently defeats the two
    guarantees the neighbouring `fails_closed` tests exist to hold.

    The store below declares NO dependencies. The invalidation reason arrives anyway, naming the
    source as its own dependency because its base state is bad, which is precisely the case where
    relabelling is least informative.
    """
    store = Store(
        [Chunk("a", "a.md", "old", {"file": "a.md"})],
        {"a.md": [("b.md", NOW), ("c.md", NOW)]},
    )
    record = project_current_state(store, as_of=NOW).records[0]

    assert record.state == "ambiguous"
    # Narrowing the relabel must not lose the finding: it is still reported as a diagnostic, so
    # both facts survive and the more actionable one is the state.
    assert "dependency_invalidated" in record.diagnostics


def test_a_current_document_is_still_relabelled_when_a_dependency_fails() -> None:
    """The narrowing must not disable the feature: `current` is the state it exists to change.

    Without this, narrowing the relabel to `current` could be satisfied by never relabelling
    anything, and the suite would stay green while the feature did nothing.
    """
    store = Store(
        [
            Chunk("p", "prereq.md", "old", {"file": "prereq.md", "valid_until": "2026-08-01"}),
            Chunk(
                "d",
                "dependent.md",
                "text",
                {"file": "dependent.md", "recall_graph": {"depends_on": ["prereq.md"]}},
            ),
        ],
    )
    states = {r.source: r.state for r in project_current_state(store, as_of=NOW).records}

    assert states["prereq.md"] == "expired"
    assert states["dependent.md"] == "dependency_invalidated"


def test_current_state_flags_a_source_named_in_the_unresolved_set() -> None:
    """Arm one of the `unresolved_supersession_reference` guard, driven by the REAL producer.

    `resolve_supersession_candidates` is thoroughly tested and `_record`'s consumption of its
    `unresolved` output was not, which left the seam between them unwatched: every fake store in
    this suite returned `frozenset()`, so the branch was unreachable by construction rather than
    by oversight.

    The rows below are the shape that actually produces a non-empty set: two indexed files share
    the stem `a`, so a `supersedes: a.md` claim cannot be resolved to one of them without
    guessing, and both candidates are named. Neither candidate carries `supersedes` metadata of
    its own, so only `source in unresolved` can be responsible for the diagnostic.

    Fail direction, and why this is worth a test: with the guard gone these fall through to
    `current`, so a document something claims to supersede is served as current with nothing
    attached saying the lineage link is broken.
    """
    rows = [("dir1/a.md", None, NOW), ("dir2/a.md", None, NOW), ("c.md", "a.md", NOW)]
    _edges, unresolved, candidates = resolve_supersession_candidates(rows)
    assert unresolved == frozenset({"dir1/a.md", "dir2/a.md"})

    store = Store(
        [
            Chunk("1", "dir1/a.md", "one", {"file": "dir1/a.md"}),
            Chunk("2", "dir2/a.md", "two", {"file": "dir2/a.md"}),
            Chunk("3", "c.md", "claimant", {"file": "c.md", "supersedes": "a.md"}),
        ],
        candidates,
        unresolved,
    )
    records = {record.source: record for record in project_current_state(store, as_of=NOW).records}

    for source in ("dir1/a.md", "dir2/a.md"):
        assert records[source].state == "ambiguous"
        assert "unresolved_supersession_reference" in records[source].diagnostics


def test_current_state_flags_a_chunk_whose_supersedes_target_is_unresolved() -> None:
    """Arm two: the chunk's own `supersedes:` value is the thing that could not be resolved.

    The source is deliberately absent from `unresolved`, so arm one cannot fire and the
    diagnostic can only come from the `any(...)` over this source's chunks. Injected directly
    rather than produced, because the two arms compare different domains: `unresolved` holds
    root-relative file identifiers while the arm-two key is a STEM (`supersedes_key` strips the
    directory and the `.md`). They coincide only for a corpus holding an extensionless file whose
    name equals another file's stem, which is what `"a"` below stands for.
    """
    store = Store(
        [Chunk("c", "c.md", "claimant", {"file": "c.md", "supersedes": "a"})],
        (),
        frozenset({"a"}),
    )
    record = project_current_state(store, as_of=NOW).records[0]

    assert record.source == "c.md"
    assert record.state == "ambiguous"
    assert "unresolved_supersession_reference" in record.diagnostics
