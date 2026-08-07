"""The splade arm of the latency split, end to end on a tiny corpus.

No checkpoint: a deterministic keyword encoder drives the same production retrieval path, so
this asserts the ARM is selectable and its guards hold, at a size a test suite can afford.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.store_latency_share import (  # noqa: E402
    LegSplit,
    _print_report,
    measure,
    to_markdown,
)
from recall.sparse import SparseProfile, backfill_learned_sparse  # noqa: E402
from recall.types import Chunk  # noqa: E402
from tests.conftest import requires_db  # noqa: E402

PROFILE_ID = "kw-arm-test"
VOCAB = {"aardvark": 7, "beta": 9, "gamma": 11, "delta": 13}


class KeywordSparseEncoder:
    def __init__(self) -> None:
        self.profile = SparseProfile(
            profile_id=PROFILE_ID, model_name="test/keyword",
            artifact_digest="sha256:test", dimension=30522, top_k=1000,
        )

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        return [
            {VOCAB[w]: 1.0 for w in text.lower().split() if w in VOCAB} for text in texts
        ]


class StubEmbedder:
    dim = 64
    name = "stub"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 64 for _ in texts]


@requires_db
def test_the_splade_arm_reports_a_learned_fire_rate_and_no_lexical_leg(make_store) -> None:
    """`splade` REPLACES ts_rank rather than adding to it.

    So under it the LEXICAL leg is the one asserted idle, and its fire rate must read null rather
    than 0.0. Zero is the issue #81 alarm value (a leg present but matching nothing), and a
    healthy splade run must not publish the alarm.
    """
    store = make_store(64)
    chunks = [
        Chunk(id=f"c{i}", source=f"/c/{i}.md", text=text, metadata={"file": f"{i}.md"})
        for i, text in enumerate(["aardvark facts", "beta prose", "gamma notes", "delta text"])
    ]
    store.upsert(chunks, [[0.1] * 64 for _ in chunks])
    encoder = KeywordSparseEncoder()
    backfill_learned_sparse(store, encoder)

    split = measure(
        store, StubEmbedder(),
        [{"query": "aardvark"}, {"query": "beta"}],
        candidate_k=10, reranker=None, n_chunks=store.count(), repeats=1,
        sparse_backend="splade", sparse_encoder=encoder,
    )

    assert split.sparse_backend == "splade"
    assert split.learned_sparse_fire_rate is not None
    assert split.learned_sparse_fire_rate > 0.0
    assert split.sparse_fire_rate is None
    assert split.max_nesting_violation_ms <= 0.0


@requires_db
def test_the_lexical_arm_still_reports_a_null_learned_fire_rate(make_store) -> None:
    """The mirror image, so neither column can be hard-coded to a constant."""
    store = make_store(64)
    chunks = [
        Chunk(id=f"c{i}", source=f"/c/{i}.md", text=text, metadata={"file": f"{i}.md"})
        for i, text in enumerate(["aardvark facts", "beta prose", "gamma notes", "delta text"])
    ]
    store.upsert(chunks, [[0.1] * 64 for _ in chunks])

    split = measure(
        store, StubEmbedder(),
        [{"query": "aardvark"}, {"query": "beta"}],
        candidate_k=10, reranker=None, n_chunks=store.count(), repeats=1,
        sparse_backend="lexical",
    )

    assert split.sparse_backend == "lexical"
    assert split.learned_sparse_fire_rate is None
    assert split.sparse_fire_rate is not None


def _a_split() -> LegSplit:
    """A minimal, self-consistent `LegSplit`. No DB: this pins a pure string-formatting bug."""
    return LegSplit(
        candidate_k=10, reranked=False, sparse_backend="splade", repeats=1, n_queries=1,
        n_chunks=4, embedder="stub",
        total_ms_mean=1.0, total_ms_p50=1.0, total_ms_p95=1.0,
        embed_ms_mean=0.1, dense_ms_mean=0.1, sparse_ms_mean=0.0,
        learned_sparse_ms_mean=0.5, learned_sparse_encode_ms_mean=0.3,
        fusion_ms_mean=0.05, rerank_ms_mean=0.0, meta_ms_mean=0.05, residual_ms_mean=0.0,
        store_ms_mean=0.5, store_ms_p50=0.5, store_share=0.5, total_ms_if_store_were_free=0.5,
        sparse_fire_rate=None, learned_sparse_fire_rate=1.0, truncated=False,
    )


def test_print_report_survives_a_console_that_cannot_encode_the_markdown() -> None:
    """Pin the traceback observed running this benchmark against a real SPLADE checkpoint.

    `to_markdown`'s scope caveat always carries a warning glyph, and Windows' default console
    codec is cp1252, which cannot encode it. The run had already written `splits.json` and
    `SPLIT.md` correctly (as UTF-8) before this final print raised, so the crash must not
    reach the caller: a successful measurement must still exit with its intended status.
    """
    md = to_markdown([_a_split()], "4 chunks, embedder `stub`, seed 0, 1 query x 1 repeat")
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252")

    _print_report(md, stream)
    stream.flush()

    rendered = buffer.getvalue().decode("cp1252")
    assert "store share" in rendered
    assert "splade" in rendered
