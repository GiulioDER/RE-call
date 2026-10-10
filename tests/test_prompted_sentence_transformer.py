"""`st-prompted:<model>` sends each model's published prompts; `st:<model>` still sends raw text.

The sentence-transformers library is replaced by a stub that records what it was asked to encode,
so these run without the `rerank` extra, a download or a model. Every test drives the same helpers
the index and the search path call: `embed_with_cache(..., purpose="passage")` and `embed_query`.
"""

from __future__ import annotations

import math
import sys
import types

import pytest

from recall.embedding_prompts import PUBLISHED_PROMPTS, prompts_for

NANO = "voyageai/voyage-4-nano"
BGE = "BAAI/bge-small-en-v1.5"
OPT_IN = {"RECALL_ACCEPT_REMOTE_MODEL_CODE": "1"}


class _Head:
    modality_config: dict | None = None


class _StubSentenceTransformer:
    calls: list[tuple[list[str], object]] = []
    inits: list[dict] = []
    width = 8
    head = _Head()

    def __init__(self, model, **kwargs):
        type(self).inits.append({"model": model, **kwargs})
        self._dim = kwargs.get("truncate_dim") or type(self).width

    def __getitem__(self, index):
        return type(self).head

    def get_sentence_embedding_dimension(self):
        return self._dim

    def encode(self, texts, **kwargs):
        type(self).calls.append((list(texts), kwargs.get("prompt", "<no prompt kwarg>")))
        # Deliberately not unit length, so normalisation is the embedder's job, observably.
        return [[float(len(t) + 1)] + [2.0] * (self._dim - 1) for t in texts]


@pytest.fixture
def stub_st(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _StubSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    _StubSentenceTransformer.calls = []
    _StubSentenceTransformer.inits = []
    _StubSentenceTransformer.head = _Head()
    return _StubSentenceTransformer


def _prompts_on_the_index_and_search_paths(spelling: str, env: dict) -> list[tuple[list[str], object]]:
    from recall.cache import embed_with_cache
    from recall.embeddings import embed_query, resolve_embedder

    embedder = resolve_embedder(spelling, env)
    _StubSentenceTransformer.calls = []
    embed_with_cache(embedder, ["a passage", "another passage"], None, purpose="passage")
    embed_query(embedder, "a question")
    return _StubSentenceTransformer.calls


@pytest.mark.parametrize("model", sorted(PUBLISHED_PROMPTS))
def test_st_prompted_sends_the_published_prompts_on_both_paths(stub_st, model):
    """Passages go out with the document prompt and queries with the query prompt, exactly.

    Invariant: for every model in the table, the index path encodes passages with
    ``prompt=<document prompt>`` (an empty string when the model has none, passed explicitly so a
    model config's ``default_prompt_name`` cannot add one) and the search path encodes the query
    with ``prompt=<query prompt>``. Failure mode caught: the measured +0.14 to +0.17 MRR@10 is
    lost silently, because a prompt that is dropped or swapped still builds, indexes and serves.

    Red proof, recorded 2026-10-11 on branch claude/local-embedder-prompts, two mutations, each
    run separately against `PromptedSentenceTransformerEmbedder` in
    `recall/embedding_providers/local.py`: (1) `embed_query` encoding with
    ``self._prompts.document`` failed all six cases at the assertion on the query element;
    (2) `_encode` passing ``prompt=prompt or None`` failed the five models without a document
    prompt at the passage element, recording ``None`` where ``""`` was required, while
    voyage-4-nano (which has one) passed. Restored, all six pass.
    """
    p = prompts_for(model)
    assert _prompts_on_the_index_and_search_paths(f"st-prompted:{model}", OPT_IN) == [
        (["a passage", "another passage"], p.document),
        (["a question"], p.query),
    ]


def test_st_spelling_still_sends_raw_text(stub_st):
    """`st:<model>` is unchanged: no prompt reaches the model on either path.

    Invariant: stores built with ``st:<model>`` hold vectors of raw text, and their queries must
    keep matching them. Failure mode caught: "fixing" `st:` in place, which would change every
    query vector while the embedder's name, the only thing the serving gate and calibration
    compare, stayed the same.

    Red proof, recorded 2026-10-11 on branch claude/local-embedder-prompts:
    `SentenceTransformerEmbedder.embed` passing ``prompt="query: "`` failed at the assertion,
    recording ``"query: "`` where no prompt keyword was required. Restored, passes.
    """
    assert _prompts_on_the_index_and_search_paths(f"st:{BGE}", {}) == [
        (["a passage", "another passage"], "<no prompt kwarg>"),
        (["a question"], "<no prompt kwarg>"),
    ]


def test_the_prompted_identity_never_matches_the_raw_one(stub_st):
    """Name, profile fingerprint and cache key all differ from `st:<model>`'s.

    Invariant: a store built with raw text and one built with prompts cannot be mistaken for each
    other by any identity RE-call checks. The name matters most, because the serving gate,
    calibration v2 and the lite store compare names only. Failure mode caught: vectors from the
    two encodings answering through one store with no error anywhere.

    Red proof, recorded 2026-10-11 on branch claude/local-embedder-prompts:
    `PromptedSentenceTransformerEmbedder` naming itself ``f"st:{model}"`` failed at the first
    assertion. Restored, passes.
    """
    from recall.cache import cache_key
    from recall.embeddings import embedding_profile, resolve_embedder

    raw = resolve_embedder(f"st:{BGE}", {})
    prompted = resolve_embedder(f"st-prompted:{BGE}", {})
    assert prompted.name != raw.name
    assert prompted.name == f"st-prompted:{BGE}@published-v1"
    assert embedding_profile(prompted).fingerprint() != embedding_profile(raw).fingerprint()
    assert cache_key(embedding_profile(prompted), prompted.dim, "same text", "passage") != \
        cache_key(embedding_profile(raw), raw.dim, "same text", "passage")


def test_a_model_without_pinned_prompts_is_refused(stub_st):
    """`st-prompted:` refuses a model the table does not hold, and says which it does.

    Red proof, recorded 2026-10-11 on branch claude/local-embedder-prompts: `prompts_for`
    returning ``PromptSet("none", "", "", "main")`` for an unknown model failed at
    `pytest.raises`. Restored, passes.
    """
    from recall.embeddings import resolve_embedder

    with pytest.raises(ValueError, match="no published prompts pinned"):
        resolve_embedder("st-prompted:someone/unknown-model", {})


def test_remote_code_models_need_the_opt_in_and_load_pinned(stub_st):
    """voyage-4-nano runs repository code, so it needs the opt-in, and loads at its pinned revision.

    Red proof, recorded 2026-10-11 on branch claude/local-embedder-prompts: removing the
    `_require_remote_model_code_opt_in` call from `PromptedSentenceTransformerEmbedder.__init__`
    failed at `pytest.raises`. Restored, passes.
    """
    from recall.embeddings import resolve_embedder

    with pytest.raises(ValueError, match="RECALL_ACCEPT_REMOTE_MODEL_CODE"):
        resolve_embedder(f"st-prompted:{NANO}", {})
    resolve_embedder(f"st-prompted:{NANO}", OPT_IN)
    init = stub_st.inits[-1]
    assert (init["revision"], init["trust_remote_code"], init["truncate_dim"]) == (
        PUBLISHED_PROMPTS[NANO].revision, True, 1024)


def test_chat_template_routing_is_refused(stub_st):
    """A sentence-transformers version that routes text through a chat template is refused.

    Failure mode caught: from sentence-transformers 5.4, a model with a chat template has its
    text wrapped in the template and the prompt sent as a system turn, so "the published prompt"
    would silently be something else.

    Red proof, recorded 2026-10-11 on branch claude/local-embedder-prompts: removing the
    ``modality_config`` check failed at `pytest.raises`. Restored, passes.
    """
    from recall.embeddings import resolve_embedder

    stub_st.head.modality_config = {"text": {}, "message": {}}
    with pytest.raises(ValueError, match="chat template"):
        resolve_embedder(f"st-prompted:{BGE}", {})


def test_truncated_vectors_come_back_unit_length(stub_st):
    """voyage-4-nano is kept at 1,024 of its 2,048 dimensions and renormalised after truncation.

    Red proof, recorded 2026-10-11 on branch claude/local-embedder-prompts: `_encode` returning
    the library's vectors without dividing by their norm failed at the norm assertion. Restored,
    passes.
    """
    from recall.embeddings import resolve_embedder

    embedder = resolve_embedder(f"st-prompted:{NANO}", OPT_IN)
    vector = embedder.embed_query("a question")
    assert embedder.dim == len(vector) == 1024
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-9)


def test_the_labelled_harness_measures_the_spelling_it_was_given(stub_st):
    """`recall.eval.labelled` resolves `st-prompted:` instead of falling back to bge-small.

    Failure mode caught: the harness used to build the default fastembed model for any spelling
    it did not know, and report the result under the name it was asked for.

    Red proof, recorded 2026-10-11 on branch claude/local-embedder-prompts: restoring the
    unconditional `FastEmbedEmbedder()` fallback in `_make_embedder` failed at the isinstance
    assertion. Restored, passes.
    """
    from recall.embedding_providers.local import PromptedSentenceTransformerEmbedder
    from recall.eval.labelled import _make_embedder

    assert isinstance(_make_embedder(f"st-prompted:{BGE}"), PromptedSentenceTransformerEmbedder)
