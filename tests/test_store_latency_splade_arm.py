"""The splade arm of the latency split, end to end on a tiny corpus.

No checkpoint: a deterministic keyword encoder drives the same production retrieval path, so
this asserts the ARM is selectable and its guards hold, at a size a test suite can afford.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import benchmarks.store_latency_share as store_latency_share  # noqa: E402
import recall.sparse as sparse  # noqa: E402
from benchmarks.store_latency_share import (  # noqa: E402
    LegSplit,
    _print_report,
    main,
    measure,
    to_markdown,
)
from recall.sparse import SparseProfile, backfill_learned_sparse  # noqa: E402
from recall.types import Chunk  # noqa: E402
from tests.conftest import TEST_DSN, requires_db  # noqa: E402

PROFILE_ID = "kw-arm-test"
#: "the" is here for `test_main_runs_end_to_end...` below, not for the two tests that follow it.
#: Every document `recall.eval.synthetic.generate()` writes carries the literal sentence "The
#: {aspect} for {subj} is ...", and every one of its answerable queries carries "what is the
#: {aspect} for {subj}" — so "the" is a token guaranteed present in both the corpus and the
#: queries that end-to-end test drives, without hand-crafting chunk text the way the two tests
#: below do. It does not change what those two tests exercise: neither their chunks nor their
#: queries contain the word "the".
VOCAB = {"aardvark": 7, "beta": 9, "gamma": 11, "delta": 13, "the": 101}


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


def _patch_from_pretrained(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for a 500 MB checkpoint download with the deterministic keyword encoder above.

    Patched on `recall.sparse.SpladeEncoder.from_pretrained` itself, the classmethod attribute,
    not on a name in `benchmarks.store_latency_share`'s namespace. Read: `main()`'s `if
    wants_learned:` block does a LAZY, function-local `from recall.sparse import (..., SpladeEncoder,
    ...)`, executed only when `main()` actually runs, and then calls `SpladeEncoder.from_pretrained(
    ...)`. That local import fetches whatever `recall.sparse.SpladeEncoder` currently IS, which is
    the same class object either way; patching the `from_pretrained` attribute on that class is
    what a call resolves at call time, regardless of whether the patch happened before or after
    the local import statement executed. A plain function assigned as a class attribute is called
    unbound when reached via the class (`SpladeEncoder.from_pretrained(...)`, not an instance), so
    no `staticmethod` wrapper is needed for the args `main()` passes to land in `*args, **kwargs`.
    """
    monkeypatch.setattr(
        sparse.SpladeEncoder, "from_pretrained",
        lambda *args, **kwargs: KeywordSparseEncoder(),
    )


@requires_db
def test_main_runs_end_to_end_across_both_backends_and_writes_the_artifacts(
    tmp_path, monkeypatch
) -> None:
    """Drive `main()` itself, not `measure()` directly.

    Both DB tests above, and the eventual guard-ablation anchor, call `measure()`. Nothing before
    this test ever drove `main()`'s own sequence: argument parsing, the host-quiet check, the
    encoder build, the corpus generate, the learned-sparse backfill, the two-backend sweep, the
    provenance write, and the final console print. The only thing that had ever run `main()` was
    one manual invocation against a real checkpoint, which found a crash no test caught: the final
    `print` of the markdown died under a Windows cp1252 console AFTER `splits.json` and `SPLIT.md`
    were already correctly on disk, so a fully successful run exited non-zero with a traceback.

    This does NOT stub `_print_report` or replace `sys.stdout` with anything cp1252-flavoured;
    `main()` prints through the real path to whatever `sys.stdout` pytest hands it, so a
    regression in that print is free to surface here exactly as it did against the real
    checkpoint. The cp1252-specific encode failure itself is pinned directly, and independently,
    by `test_print_report_survives_a_console_that_cannot_encode_the_markdown` above.
    """
    _patch_from_pretrained(monkeypatch)

    out = tmp_path / "out"
    argv = [
        "store_latency_share",
        "--dsn", TEST_DSN,
        "--embedder", "hashing",
        "--queries", "4",
        "--repeats", "1",
        "--candidate-k", "10",
        "--sparse-backend", "lexical",
        "--sparse-backend", "splade",
        "--allow-busy-host",
        "--sparse-device", "cpu",
        "--out", str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    exit_code = main()

    assert exit_code == 0

    payload = json.loads((out / "splits.json").read_text(encoding="utf-8"))
    splits = payload["splits"]
    assert len(splits) == 2

    by_backend = {s["sparse_backend"]: s for s in splits}
    assert set(by_backend) == {"lexical", "splade"}

    # `splade` REPLACES the ts_rank leg rather than adding to it, so the two rows are exact
    # mirrors of which fire-rate column is populated and which reads `null` (NOT 0.0 — that is
    # the issue #81 alarm value for a leg that is present but matching nothing).
    lexical = by_backend["lexical"]
    assert lexical["sparse_fire_rate"] is not None
    assert lexical["learned_sparse_fire_rate"] is None

    splade = by_backend["splade"]
    assert splade["sparse_fire_rate"] is None
    assert splade["learned_sparse_fire_rate"] is not None

    for split in splits:
        assert split["max_nesting_violation_ms"] <= 0.0

    # `host_load_per_core_before`/`_after` are legitimately `null` on Windows (no `getloadavg`),
    # so the keys' PRESENCE is what is asserted, not that a reading was taken.
    provenance = payload["_provenance"]
    for key in (
        "host_load_per_core_before", "host_load_per_core_after",
        "host_load_ceiling_per_core", "host_load_override",
    ):
        assert key in provenance

    assert payload["sparse_device"] is not None
    assert payload["sparse_device"]["requested"] == "cpu"
    assert payload["sparse_device"]["resolved"] == "cpu"

    split_md = (out / "SPLIT.md").read_text(encoding="utf-8")
    assert "lexical" in split_md
    assert "splade" in split_md


def test_main_reads_the_sparse_device_exactly_once(tmp_path, monkeypatch) -> None:
    """Pin the fix for two independent live `torch.cuda` queries in `main()`'s device block.

    `main()` calls `inspect_sparse_device(...)` once directly, to build the report it stamps into
    provenance, and then calls `resolve_sparse_device(...)`. Before the fix that second call
    queried `inspect_sparse_device` again internally: two separate live reads that were not
    guaranteed to agree near a VRAM threshold, so the reading that drove the decision and the
    reading that got published could describe two different moments. Counted here: 2 before the
    fix, 1 after.

    No DB is needed to reach the code under test: `main()`'s device block runs, and finishes,
    before it ever generates a corpus or opens a store, so `_throwaway_store` is replaced with a
    stub that refuses immediately, keeping this test fast and independent of `@requires_db`
    without touching anything the device block itself does.
    """
    _patch_from_pretrained(monkeypatch)

    calls = {"n": 0}
    real_inspect = sparse.inspect_sparse_device

    def counting_inspect(*args, **kwargs):
        calls["n"] += 1
        return real_inspect(*args, **kwargs)

    # Patched on the module attribute, not the function object: `resolve_sparse_device`'s
    # internal call is a bare-name lookup resolved against `recall.sparse`'s own globals at call
    # time, so overwriting the module attribute is what a fixed OR unfixed `resolve_sparse_device`
    # both actually consult.
    monkeypatch.setattr(sparse, "inspect_sparse_device", counting_inspect)

    def _refuse_the_db(*args, **kwargs):
        raise RuntimeError("this test's subject is the device block, not the store")

    monkeypatch.setattr(store_latency_share, "_throwaway_store", _refuse_the_db)

    argv = [
        "store_latency_share",
        "--dsn", "postgresql://unused:unused@127.0.0.1/unused",
        "--embedder", "hashing",
        "--queries", "1",
        "--repeats", "1",
        "--sparse-backend", "splade",
        "--allow-busy-host",
        "--sparse-device", "cpu",
        "--out", str(tmp_path / "out"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(RuntimeError, match="this test's subject"):
        main()

    assert calls["n"] == 1
