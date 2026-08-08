"""Unit tests for the learned sparse (SPLADE) encoder, its identity, and its pruning.

Nothing here touches Postgres or downloads a model. The store and retriever integration lives in
`tests/test_learned_sparse_store.py`, which is `requires_db`.
"""

from __future__ import annotations

import math

import pytest

from recall.sparse import SpladeEncoder, SparseProfile, prune_to_top_k, splade_weights


def test_prune_keeps_only_the_largest_weights() -> None:
    """Pruning is top-k by WEIGHT, not by term id and not by insertion order.

    The HNSW ceiling measured on pgvector 0.8.4 is 1000 non-zero elements, and a SPLADE passage
    expansion can exceed it. Which terms survive therefore decides what the index can retrieve,
    so dropping the wrong ones is a silent recall loss rather than an error.
    """
    weights = {5: 0.5, 9: 3.0, 12: 1.0, 30521: 0.25}

    assert prune_to_top_k(weights, 2) == {9: 3.0, 12: 1.0}


def test_prune_breaks_ties_by_term_id_not_by_insertion_order() -> None:
    """Equal weights must resolve the same way regardless of how the dict was built.

    SPLADE weights collide often enough for this to bite, and `sorted` is stable, so a tie is
    resolved by insertion order unless something else decides it. That would make the same text
    encode to two different stored vectors depending on iteration order upstream — and the
    profile fingerprint promises reproducibility, so a nondeterministic prune makes the identity
    a claim the encoder cannot honour.
    """
    ascending = {3: 2.0, 5: 1.0, 9: 1.0}
    descending = {3: 2.0, 9: 1.0, 5: 1.0}

    assert prune_to_top_k(ascending, 2) == prune_to_top_k(descending, 2) == {3: 2.0, 5: 1.0}


def _profile(**overrides: object) -> SparseProfile:
    base: dict[str, object] = {
        "profile_id": "splade-pp-en-v1",
        "model_name": "prithivida/Splade_PP_en_v1",
        "artifact_digest": "sha256:" + "a" * 8,
        "dimension": 30522,
        "top_k": 1000,
    }
    return SparseProfile(**(base | overrides))  # type: ignore[arg-type]


def test_prune_budget_changes_the_fingerprint() -> None:
    """The top-k budget is part of the encoder's identity, because it changes the stored vector.

    A corpus encoded at top-k 512 is not the same corpus as one encoded at top-k 1000: the
    vectors differ, so their scores differ. If the budget were outside the fingerprint, the two
    would share a cache key and an identity, and a re-encode at a new budget would be served
    from vectors built under the old one with nothing reporting a mismatch.
    """
    assert _profile(top_k=1000).fingerprint() != _profile(top_k=512).fingerprint()


def test_identical_profiles_fingerprint_identically() -> None:
    """Without this, the inequality above is satisfied by a fingerprint that is simply random.

    A hash that never collides proves nothing about what it distinguishes.
    """
    assert _profile().fingerprint() == _profile().fingerprint()


def test_padding_positions_do_not_contribute_terms() -> None:
    """Masked positions must be excluded BEFORE the max-pool, not merely zeroed after it.

    SPLADE pools with a max over the sequence. Padding carries real logits, and on a batch of
    mixed-length passages those logits are frequently the largest in their column. Forget the
    mask and every short passage acquires terms from whatever the padding decoded to — a bug
    that produces plausible vectors, sensible-looking scores, and silent recall damage, because
    nothing about the output looks wrong.

    Here position 1 is padding and its term 0 logit (10.0) is the largest number in the tensor.
    It must not appear in the result.
    """
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[[0.0, 5.0, -1.0], [10.0, 0.0, 0.0]]])
    attention_mask = torch.tensor([[1, 0]])

    weights = splade_weights(logits, attention_mask)

    assert list(weights) == [{1: pytest.approx(math.log1p(5.0))}]


@pytest.fixture
def tiny_splade(tmp_path):
    """A REAL BertForMaskedLM and a REAL tokenizer, both tiny and randomly initialised.

    Not a mock. The point is to exercise the actual tokenize -> forward -> pool -> prune path,
    including a genuinely ragged batch, without a 500MB download. Random weights are fine
    because none of these tests assert a retrieval quality, only the mechanics.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"tok{i}" for i in range(25)]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(tmp_path / "vocab.txt"))
    config = transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16, max_position_embeddings=32,
    )
    torch.manual_seed(0)
    model = transformers.BertForMaskedLM(config).eval()
    return tokenizer, model, len(vocab)


def test_encoder_never_returns_more_terms_than_the_prune_budget(tiny_splade) -> None:
    """The budget is not advisory: pgvector REFUSES an over-budget vector at insert time.

    An encoder that returns whatever the model produced pushes that failure to the database,
    where it surfaces as a mid-load crash after an arbitrary number of rows have been written.
    The budget belongs here, where it is also part of the identity.
    """
    tokenizer, model, vocab_size = tiny_splade
    encoder = SpladeEncoder(
        tokenizer=tokenizer, model=model,
        profile=SparseProfile(
            profile_id="tiny", model_name="tiny", artifact_digest="sha256:test",
            dimension=vocab_size, top_k=2,
        ),
    )

    encoded = encoder.encode(["tok1 tok2 tok3 tok4", "tok5"])

    assert [len(row) for row in encoded] == [2, 2]


def test_encoder_refuses_a_budget_above_the_hnsw_limit(tiny_splade) -> None:
    """Reject an impossible budget at construction, not at row 40,000 of a load.

    pgvector's HNSW index caps a `sparsevec` at 1000 non-zero elements and raises on INSERT.
    A profile built at top_k=1200 is therefore not a slow configuration, it is one that cannot
    be stored — and discovering that partway through a 366k-passage load costs the whole load.
    """
    tokenizer, model, vocab_size = tiny_splade

    with pytest.raises(ValueError, match="HNSW"):
        SpladeEncoder(
            tokenizer=tokenizer, model=model,
            profile=SparseProfile(
                profile_id="tiny", model_name="tiny", artifact_digest="sha256:test",
                dimension=vocab_size, top_k=1001,
            ),
        )


def test_noncommercial_weights_must_be_opted_into_explicitly() -> None:
    """`naver/splade-v3` is CC-BY-NC-SA-4.0 and RE-call is MIT.

    The refusal is checked BEFORE any download, so this test needs no network: an accidental
    non-commercial dependency should cost a clear error, not 500MB and a licence violation
    discovered later. `Splade_PP_en_v1` (apache-2.0) is what you get if you do not choose.
    """
    with pytest.raises(ValueError, match="cc-by-nc-sa-4.0"):
        SpladeEncoder.from_pretrained("naver/splade-v3")


def test_the_default_model_is_the_permissively_licensed_one() -> None:
    """Stated as an executable fact, so a later edit to DEFAULT_MODEL cannot quietly make a
    non-commercial model the default that everyone installs without choosing it."""
    from recall.sparse import DEFAULT_MODEL, KNOWN_MODELS

    assert KNOWN_MODELS[DEFAULT_MODEL].license_id == "apache-2.0"
    assert KNOWN_MODELS[DEFAULT_MODEL].is_commercial_ok


def test_every_recorded_model_can_be_attributed() -> None:
    """CC BY-NC-SA 4.0 requires credit, a licence link, and a statement of changes.

    A licence id alone does not discharge that, so every entry must carry all three. Checked for
    every model rather than the two we happen to run, because the next one added is the one that
    will be missing a field.
    """
    from recall.sparse import KNOWN_MODELS, attribution_notice

    for name, entry in KNOWN_MODELS.items():
        assert entry.creator and entry.license_url and entry.source_url and entry.changes, name
        notice = attribution_notice(name)
        assert entry.creator in notice
        assert entry.license_url in notice
        assert entry.changes in notice


def test_a_noncommercial_model_is_attributed_as_noncommercial() -> None:
    """The notice has to say the thing that constrains the reader, not just name a licence.

    Someone reading a benchmark result needs to know this checkpoint is not what ships and that
    redistributing the derived vectors would carry the ShareAlike obligation.
    """
    from recall.sparse import attribution_notice

    notice = attribution_notice("naver/splade-cocondenser-ensembledistil")

    assert "NON-COMMERCIAL" in notice
    assert "ShareAlike" in notice
    assert "cc-by-nc-sa-4.0" in notice
    assert "https://creativecommons.org/licenses/by-nc-sa/4.0/" in notice
    assert "not shipped with RE-call" in notice


def test_attribution_is_refused_for_an_unrecorded_model() -> None:
    """Better to fail than to emit a result with no credit line attached to it."""
    from recall.sparse import attribution_notice

    with pytest.raises(ValueError, match="no licence recorded"):
        attribution_notice("someone/unrecorded-checkpoint")


def test_every_model_is_documented_in_the_third_party_notice() -> None:
    """The repo-level attribution must not drift from the registry.

    A model added to `KNOWN_MODELS` and forgotten here is an undischarged Attribution obligation
    that nothing else would catch, since the code keeps working perfectly.
    """
    from pathlib import Path

    from recall.sparse import KNOWN_MODELS

    notice = Path(__file__).resolve().parent.parent / "docs" / "THIRD_PARTY_MODELS.md"
    text = notice.read_text(encoding="utf-8")

    for name, entry in KNOWN_MODELS.items():
        assert name in text, f"{name} is in KNOWN_MODELS but undocumented in {notice.name}"
        assert entry.license_url in text, f"{name}'s licence link is missing from {notice.name}"


def test_encoder_sends_inputs_to_the_models_device(tiny_splade) -> None:
    """Inputs must follow the MODEL's device, never a hardcoded one.

    Without this, `from_pretrained` loads to CPU, `encode` feeds CPU tensors, and a rented GPU
    sits idle while the whole corpus runs on the instance's CPU. Nothing errors: you get correct
    vectors, slowly, and a bill. That is the failure this pins.

    No GPU is available in CI, and the first version of this test asserted
    `encoder.device == torch.device("cpu")` on a CPU-only box. Mutating the property to a
    hardcoded `cpu` left it GREEN, so it pinned nothing: the two sides of the assertion agreed
    for a reason unrelated to the code being right.

    Moving the model to torch's `meta` device fixes that. `meta` is not `cpu`, so a hardcoded
    device now disagrees with the model and the test fails. No forward pass is needed, which is
    the point: this is about where inputs are ROUTED, not about computing anything.
    """
    torch = pytest.importorskip("torch")
    tokenizer, model, vocab_size = tiny_splade
    encoder = SpladeEncoder(
        tokenizer=tokenizer, model=model,
        profile=SparseProfile(
            profile_id="tiny", model_name="tiny", artifact_digest="sha256:test",
            dimension=vocab_size, top_k=5,
        ),
    )
    assert encoder.device == torch.device("cpu")

    model.to("meta")

    assert encoder.device == next(model.parameters()).device
    assert encoder.device.type == "meta"


def test_sparse_ef_search_widens_far_enough_to_fill_k() -> None:
    """The multiplier that stops the learned sparse leg silently under-returning.

    MEASURED on the real 183,408-row index, four corpora sharing one sidecar, k=100:

        ef_search   40 (pgvector default) ->   6 candidates
        ef_search  100                    ->  19
        ef_search  400                    ->  72
        ef_search 1000                    -> 100

    The tenant/table/profile filters discard most of what the HNSW walk visits, so the scan
    exhausts its budget long before k survivors. A 10x multiplier is what closed the gap.

    This unit test pins the ARITHMETIC and the cap. It deliberately does NOT claim to reproduce
    the truncation: that needs an index large enough for Postgres to choose HNSW over a seq scan,
    and a test at that scale would be slow and flaky. A version of this written at 150 rows
    passed against the buggy code, which is why the evidence above is a measurement rather than
    an assertion.
    """
    from recall.store import _HNSW_EF_SEARCH_MAX, sparse_ef_search

    assert sparse_ef_search(100) == 1000
    assert sparse_ef_search(20) == 200
    # Never below pgvector's own default: widening must only ever widen.
    assert sparse_ef_search(1) == 40
    # Capped at what pgvector accepts, rather than raising on a legal k.
    assert sparse_ef_search(500) == _HNSW_EF_SEARCH_MAX
