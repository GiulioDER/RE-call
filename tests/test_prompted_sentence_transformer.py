"""`st-prompted:<model>` sends each model's published prompts; `st:<model>` still sends raw text.

The sentence-transformers library is replaced by a stub that records what it was asked to encode,
so these run without the `rerank` extra, a download or a model. Every test drives the same helpers
the index and the search path call: `embed_with_cache(..., purpose="passage")` and `embed_query`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
import sys
import types
from pathlib import Path

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
    ``self._prompts.document`` failed all five cases at the assertion on the query element;
    (2) `_encode` passing ``prompt=prompt or None`` failed the four models without a document
    prompt at the passage element, recording ``None`` where ``""`` was required, while
    voyage-4-nano (which has one) passed. Restored, all five pass.
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


def test_a_local_path_shadowing_the_model_id_is_refused(stub_st, tmp_path, monkeypatch):
    """A directory named like the hub id would be loaded instead of the pinned revision.

    Failure mode caught: sentence-transformers prefers an existing local path to the hub, so a
    `voyageai/voyage-4-nano` directory in the working directory would bypass the revision pin and,
    with remote code allowed, run whatever code that directory holds.

    Red proof, recorded 2026-10-11 on branch claude/local-embedder-prompts: removing the
    ``Path(model).exists()`` check from `PromptedSentenceTransformerEmbedder.__init__` failed at
    `pytest.raises`. Restored, passes.
    """
    from recall.embeddings import resolve_embedder

    (tmp_path / NANO).mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="shadow the pinned hub revision"):
        resolve_embedder(f"st-prompted:{NANO}", OPT_IN)
    assert stub_st.inits == []


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


#: A digest of every field of each pinned prompt set, keyed by model and version and transcribed by
#: hand. The prompt text is not part of the embedder's identity: its name and profile carry the
#: version in place of the text, so an entry edited in place keeps the identity of the vectors
#: written under the old prompt. Changing an entry means a new version string and a new row here.
_PROMPT_SET_DIGESTS = {
    ("BAAI/bge-large-en-v1.5", "published-v1"): "18f15574500e9bed",
    ("BAAI/bge-small-en-v1.5", "published-v1"): "a170099a4754064a",
    ("Qwen/Qwen3-Embedding-0.6B", "published-v1"): "4f8244473bb35840",
    ("Snowflake/snowflake-arctic-embed-l-v2.0", "published-v1"): "6db055d3da786682",
    ("voyageai/voyage-4-nano", "published-v1"): "1c732685f8443311",
}


def _prompt_set_digest(prompts) -> str:
    return hashlib.sha256(repr(dataclasses.astuple(prompts)).encode()).hexdigest()[:16]


@pytest.mark.parametrize("model", sorted(PUBLISHED_PROMPTS))
def test_a_changed_prompt_set_needs_a_new_version(model):
    """Each pinned prompt set still matches the digest recorded for its model and version.

    Failure mode caught: a query or document prompt edited without a version bump keeps the
    embedder's name, profile id and fingerprint, so a store built under the old prompt keeps
    serving as trusted while queries are encoded under the new one.

    Red proof, recorded 2026-10-27 on branch claude/cca-pr910-descending: editing `_BGE_QUERY` in
    `recall/embedding_prompts.py` (dropping "relevant") with the version left at published-v1
    failed both bge cases at the digest assertion. Restored, passes.
    """
    prompts = PUBLISHED_PROMPTS[model]
    key = (model, prompts.version)
    assert key in _PROMPT_SET_DIGESTS, f"{model} {prompts.version}: record its digest here"
    assert _prompt_set_digest(prompts) == _PROMPT_SET_DIGESTS[key], (
        f"{model}: a field changed under version {prompts.version!r}; give the entry a new version "
        "(or, if a field was added to PromptSet itself, re-transcribe every digest here)"
    )


def test_every_prompted_model_has_a_licence_row():
    """docs/MODEL_LICENSES.md has a row for every model `st-prompted:` can load.

    Failure mode caught: a model added to PUBLISHED_PROMPTS without its licence read off the model
    card, which is how four of them shipped.

    It reads only its own section, because `BAAI/bge-small-en-v1.5` also appears in the default
    stack's table, which would satisfy a whole-file search with the row here deleted.

    Red proof, recorded 2026-10-27 on branch claude/cca-pr910-descending: deleting the
    bge-small-en-v1.5 row from this section (the default stack's row left in place), and separately
    emptying the Qwen3-Embedding-0.6B licence cell, each failed at the `not missing` assertion
    naming that model. Restored, passes.
    """
    doc = Path(__file__).resolve().parents[1] / "docs" / "MODEL_LICENSES.md"
    text = doc.read_text(encoding="utf-8")
    start = text.index("## Local embedders with published prompts")
    end = text.find("\n## ", start + 1)
    section = text[start:] if end == -1 else text[start:end]
    missing = sorted(
        model for model in PUBLISHED_PROMPTS
        if not re.search(rf"^\| `{re.escape(model)}` \| [^|\s][^|]*\|", section, re.M)
    )
    assert not missing, f"docs/MODEL_LICENSES.md has no licence row for {missing}"
