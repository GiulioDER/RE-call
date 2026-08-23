"""Systems whose behaviour is known by construction, for the eval calibration fleet.

Nothing here ships in the wheel. See docs/EVAL_CALIBRATION_FLEET_DESIGN.md.
"""
from __future__ import annotations

from datetime import UTC, datetime

from recall.types import Chunk, ScoredChunk


class ScriptedEmbedder:
    """An embedder that records the last text it was asked to encode.

    The store needs the QUERY TEXT to decide what to return, and `query_dense` receives only a
    vector. Rather than encode the query into the vector and decode it back (which
    `tests/fakes.py:VectorKeyedFakeStore` does, at the cost of unreadable fixtures), the
    embedder and the store share this object. `HybridRetriever._retrieve_legs` calls
    `embed_query` BEFORE `query_dense` on every search, so `last_query` is always the query
    currently being served.

    `name` is set explicitly to satisfy the `Embedder` protocol, whose docstring documents it as
    identifying the backend, used in logging and evals. Two call sites read it, and it is worth
    being precise about which one does what.

    `_score_config` builds the returned `AblationResult` with
    `embedder=embedding_profile_id(embedder)` (`harness.py:167`), using this raw, unwrapped
    instance, never the `TimedEmbedder` wrapper. Without `name`, `embedding_profile_id` would
    fall through `profile` (also absent here) to `type(embedder).__name__`, so the label would
    read "ScriptedEmbedder", not "TimedEmbedder".

    The label "TimedEmbedder" is reachable, but through a different field entirely:
    `HybridRetriever` builds `RetrievalDiagnostics.embedding_profile` from `self._embedder`,
    which `_score_config` sets to the `TimedEmbedder` wrapper. `TimedEmbedder.name` is a property
    delegating to `self._inner.name`; if `ScriptedEmbedder` had no `name`, that property access
    would raise `AttributeError`, which `embedding_profile_id`'s own `getattr(obj, "name", None)`
    swallows, falling back to `type(embedder).__name__` on the wrapper this time. That is the
    one field that would actually say "TimedEmbedder", and the fleet never reads it.

    So the attribute is kept for a plainer reason: `ScriptedEmbedder` claims to satisfy the
    `Embedder` protocol, and an implementation missing a property the protocol documents is a
    fake that fails its own interface quietly.
    """

    dim = 4
    name = "scripted"

    def __init__(self) -> None:
        self.last_query: str | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if texts:
            self.last_query = texts[-1]
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class QueryKeyedStore:
    """A store whose dense leg is scripted per QUERY TEXT.

    `script` maps a query string to the rows it retrieves, in order, as `(chunk_id, score)`.
    `chunk_id` is the `file:ord` form `harness._key` rebuilds from chunk metadata, so a member
    writes the same string it puts in `relevant_ids`.

    Two things are controlled at once and both matter:

      * the ORDER of the ids, which drives precision, recall, MRR and nDCG;
      * the dense SCORES, which drive `gap_warning`. `HybridRetriever` computes it as
        `max(dense_scores) < 0.50` (`recall/guards.py`), so a member picks scores either side
        of 0.50 to drive it deterministically.

    The sparse legs return nothing. The fleet drives `_score_config` with `fusion="dense"`,
    which sets `use_sparse=False`, so RRF runs over a single list and preserves the scripted
    order exactly. Under a hybrid fusion the fused score would reorder the hits and every
    closed form in `members.py` would silently stop holding.
    """

    def __init__(
        self, embedder: ScriptedEmbedder, script: dict[str, list[tuple[str, float]]]
    ) -> None:
        self._embedder = embedder
        self._script = script

    @staticmethod
    def _chunk(chunk_id: str) -> Chunk:
        # `harness._key` rebuilds the id as f"{metadata['file']}:{metadata['ord']}" and raises
        # KeyError on the empty metadata `tests/fakes.py:FakeStore` supplies, so splitting the
        # id back into its two parts here is load-bearing, not cosmetic.
        file, _, ordinal = chunk_id.rpartition(":")
        return Chunk(
            id=chunk_id,
            source="fleet",
            text=f"text-{chunk_id}",
            metadata={"file": file, "ord": int(ordinal)},
        )

    def _scripted_rows(self) -> list[tuple[str, float]]:
        """Every row scripted for the query now in flight — this store's entire corpus.

        Shared by the two dense readers so they cannot drift: `query_dense` truncates to `k` and
        `top_cosine` does not, and if those ever disagreed about WHICH rows exist, a fleet member's
        closed form would move without its script changing.
        """
        query = self._embedder.last_query
        if query not in self._script:
            raise KeyError(
                f"no script for query {query!r}. A fleet member must script every query it "
                f"asks about: an unscripted query returning no rows would score as a total "
                f"retrieval failure, which is a fixture bug that reads exactly like a defect."
            )
        return self._script[query]

    def query_dense(self, vector, k, source=None):
        return [
            ScoredChunk(chunk=self._chunk(cid), score=score)
            for cid, score in self._scripted_rows()[:k]
        ]

    def top_cosine(self, vector):
        """The exact best cosine in scope, which for a scripted store is over the WHOLE script.

        Deliberately not `query_dense(vector, k=1)[0].score`. On the real store those two differ —
        `ORDER BY ... LIMIT 1` may be served by an approximate index while an aggregate cannot be,
        which is the defect `recall.eval.calibrate.measure_top_cosines` now avoids by calling this.
        A double that computed its maximum by taking the first of its own top-k would model the
        approximate reading rather than the exact one, and would go on passing if the real store
        regressed to it.

        `default=0.0` matches `PgVectorStore.top_cosine` on an empty scope: a scripted query with
        no rows scores zero, exactly as a query that retrieved nothing did before.
        """
        return max((score for _, score in self._scripted_rows()), default=0.0)

    def query_sparse(self, text, k, source=None, vec=None):
        return []

    def query_learned_sparse(self, weights, k, profile_id, source=None, vec=None):
        return []

    def newest_indexed_at(self):
        return datetime.now(UTC)


class QueryKeyedTrustStore(QueryKeyedStore):
    """`QueryKeyedStore` plus the surface `recall.trust.trusted_search` needs beyond a plain
    `HybridRetriever.search`.

    Reached only through `run_trust_eval`/`run_nearmiss_eval`'s injected `store_factory`
    (`recall/eval/harness.py`). `trusted_search` calls `store.supersession()` unconditionally
    whenever a search returns any hit at all (`recall/trust.py:738-744`, the `known_as_of is
    None` branch every fleet member takes) — a bare `QueryKeyedStore` has no such method, so
    without this subclass every trust-layer call on a scripted store raises `AttributeError`
    before a single fleet member could run.

    `supersession_edges` maps a superseded FILE (the `metadata["file"]` basename `_chunk` builds
    from a scripted chunk id, e.g. `"stale.md"` for `"stale.md:0"`) to its successor file, in
    the exact shape `recall.trust.resolve_successor` walks. Deliberately real, not a stub that
    always returns empty: `recall/trust.py:_verdict` reads this to decide `superseded` BEFORE it
    ever looks at the hit's score, so scripting an edge exercises the actual supersession
    detection a fleet member certifies, not merely a pass-through of the dense score.

    `touch_files` is a documented no-op, not an omission. `run_trust_eval`'s `touch_stale=True`
    default calls it to give stale documents a fresh `indexed_at` so the RECENCY arm can be
    fooled by a re-sync (`recall/eval/harness.py`'s `run_trust_eval` docstring) — but the chunks
    this store hands back never carry an `indexed_at` in the first place (`_chunk` builds a bare
    `Chunk`, and `QueryKeyedStore.query_dense` wraps it in a `ScoredChunk` with no `indexed_at`
    argument, so it defaults to `None` on every hit). There is nothing for a touch to move, so
    accepting the call and doing nothing is the honest behaviour; every fleet member built on
    this store calls `run_trust_eval(..., touch_stale=False)` and does not rely on it.
    """

    def __init__(
        self,
        embedder: ScriptedEmbedder,
        script: dict[str, list[tuple[str, float]]],
        *,
        supersession_edges: dict[str, str] | None = None,
    ) -> None:
        super().__init__(embedder, script)
        self._supersession_edges = dict(supersession_edges or {})

    def supersession(self) -> tuple[dict[str, str], frozenset[str]]:
        # `frozenset()`: no member scripts an AMBIGUOUS target (a basename two documents share),
        # so `unresolved` is always empty here — a real gap, named in test_fleet.py's roll-up.
        return dict(self._supersession_edges), frozenset()

    def touch_files(self, files: list[str]) -> int:
        return 0


class AlwaysEntailJudge:
    """An `EntailmentJudge` that agrees with every candidate — the entailment arms in
    `recall/eval/harness.py`'s `run_nearmiss_eval` (`ARM_STACKED`, `ARM_ENTAIL_ONLY`) require
    SOME judge instance to be constructed at all, even for a fleet member that only cares about
    `ARM_THRESHOLD`'s row, so this exists to satisfy `run_nearmiss_eval`'s required `judge`
    argument without crashing or demoting anything.

    Named after, and functionally identical to, `AcceptAll` in `tests/test_eval_nearmiss.py`,
    which the real (DB-backed) near-miss tests use to pin the exact same degeneracy this fleet
    exploits: with a judge that never disagrees, `ARM_STACKED` cannot differ from `ARM_THRESHOLD`
    on any published rate (`recall/entailment.py`'s `apply_entailment` only ever DEMOTES a hit,
    never promotes one), so a fleet member built on this judge can assert
    `ARM_STACKED == ARM_THRESHOLD` as a genuine, non-vacuous, derived property, not merely leave
    it unchecked.

    Declared blind spot, named again in `test_fleet.py`'s roll-up: no member built on this judge
    can show the entailment DEMOTION itself firing (`ok` -> `not_entailed`), because doing so
    needs a judge that disagrees on at least one candidate, and this one never does.
    """

    def judge(self, query: str, texts: list[str]) -> list[bool]:
        return [True] * len(texts)
