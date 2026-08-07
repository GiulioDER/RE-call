"""Learned sparse (SPLADE) retrieval: encoder, identity, and pruning.

See `docs/superpowers/specs/2026-08-06-learned-sparse-splade-design.md`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Protocol
import hashlib


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"

if TYPE_CHECKING:  # pragma: no cover - typing only, torch is an optional dependency
    import torch


@dataclass(frozen=True)
class SparseProfile:
    """Immutable identity for every input that can change a stored sparse vector.

    Deliberately NOT an extension of `EmbeddingProfile`. That class documents fingerprint
    stability as a contract and is pinned by a test transcribing its encoding independently, so a
    field added there re-partitions every dense cache in existence. Learned sparse gets its own
    identity standing beside it.
    """

    profile_id: str
    model_name: str
    artifact_digest: str
    dimension: int
    top_k: int
    pooling: str = "max"
    activation: str = "log1p_relu"
    dependencies: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_id or not self.model_name or not self.artifact_digest:
            raise ValueError("sparse profile identity fields must be non-empty")
        if self.dimension < 1:
            raise ValueError("sparse profile dimension must be positive")
        if self.top_k < 1:
            raise ValueError("sparse profile top_k must be positive")

    def fingerprint(self) -> str:
        """SHA256 over the COMPLETE identity, as durable cache and provenance key material.

        `top_k` is in here because pruning changes the stored vector: a corpus encoded at 512 is
        not the corpus encoded at 1000, and without the budget in the key the two would share an
        identity and a cache entry.

        Fields are NUL-terminated before hashing. The terminators are what make the
        concatenation unambiguous; without them ``("ab", "c")`` and ``("a", "bc")`` hash alike.
        """
        digest = hashlib.sha256()
        parts = [
            "sparse-profile-fingerprint-v1",
            self.profile_id,
            self.model_name,
            self.artifact_digest,
            str(self.dimension),
            str(self.top_k),
            self.pooling,
            self.activation,
        ]
        for name, pinned_version in self.dependencies:
            parts.extend((name, pinned_version))
        for part in parts:
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()


def prune_to_top_k(weights: dict[int, float], top_k: int) -> dict[int, float]:
    """The `top_k` largest-weight entries of `weights`.

    pgvector's HNSW index accepts at most 1000 non-zero elements per `sparsevec` (measured on
    0.8.4; the type itself accepts 16000), and a SPLADE passage expansion can exceed that. So a
    vector is pruned before it is stored, and which terms survive decides what the index can
    retrieve.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if len(weights) <= top_k:
        return dict(weights)
    # Tie broken by term id, ascending. `sorted` is stable, so without the second key equal
    # weights resolve by dict INSERTION ORDER: the same text would store two different vectors
    # depending on how the caller happened to build the mapping, and the profile fingerprint
    # promises reproducibility it could not then deliver.
    ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
    return dict(ranked[:top_k])


def splade_weights(logits: "torch.Tensor", attention_mask: "torch.Tensor") -> list[dict[int, float]]:
    """SPLADE term weights for a batch: ``max_over_sequence(log1p(relu(logits)) * mask)``.

    `logits` is the MLM head output, shaped (batch, sequence, vocabulary); `attention_mask` is
    (batch, sequence). Returns one ``{term_id: weight}`` mapping per input, zeros dropped.

    🔑 The mask is applied BEFORE the max-pool, and that ordering is the whole correctness
    question. Padding positions carry real logits, and across a batch of mixed-length passages
    they are routinely the largest value in their vocabulary column. Pooling first and masking
    after cannot undo that: the padding's term has already won the max. The result would be
    plausible vectors with plausible scores and silently corrupted recall on every short passage
    in the batch.
    """
    import torch

    activated = torch.log1p(torch.relu(logits))
    masked = activated * attention_mask.unsqueeze(-1).to(activated.dtype)
    pooled = masked.max(dim=1).values
    out: list[dict[int, float]] = []
    for row in pooled:
        nonzero = torch.nonzero(row, as_tuple=False).flatten()
        out.append({int(term): float(row[term]) for term in nonzero})
    return out


#: Learned sparse encoders that ship as named, supported choices.
#:
#: ⚠️ `naver/splade-v3` is CC-BY-NC-SA-4.0 — NON-COMMERCIAL. RE-call is MIT, so it is never the
#: default; it is here because it is the model MTRAGEval rank 3 used and the comparable number is
#: worth being able to produce. `Splade_PP_en_v1` is apache-2.0 and is what you get if you do not
#: choose. Weights are downloaded by the user at runtime and are not vendored by this package.
KNOWN_MODELS: dict[str, str] = {
    "prithivida/Splade_PP_en_v1": "apache-2.0",
    "naver/splade-v3": "cc-by-nc-sa-4.0",
    # SPLADE++ EnsembleDistil. Same BERT MLM architecture and 30522 vocabulary as the two above,
    # so it is a drop-in for this encoder, and stronger than the default (MRR@10 38.3 vs 37.22).
    # It exists here because `naver/splade-v3` is a GATED repo: reaching it needs an approved
    # HuggingFace account, not merely the licence flag. This is the closest ungated substitute
    # for the "is our checkpoint the weak link?" control.
    "naver/splade-cocondenser-ensembledistil": "cc-by-nc-sa-4.0",
}

DEFAULT_MODEL = "prithivida/Splade_PP_en_v1"

#: pgvector refuses more than this many non-zero elements in an HNSW-indexed `sparsevec`.
#: Measured on 0.8.4, not read from documentation: the type itself accepts 16000, the INDEX
#: accepts 1000, and exceeding it raises at INSERT.
HNSW_MAX_NONZERO = 1000


class SpladeEncoder:
    """Encodes text to SPLADE term weights, pruned to the profile's budget.

    The tokenizer and model are injected rather than loaded in `__init__`, so the encode path is
    testable against a real (tiny) `BertForMaskedLM` without a download. `from_pretrained` is the
    production constructor.
    """

    def __init__(self, *, tokenizer: Any, model: Any, profile: SparseProfile,
                 max_length: int = 512) -> None:
        if profile.top_k > HNSW_MAX_NONZERO:
            raise ValueError(
                f"top_k={profile.top_k} exceeds pgvector's HNSW limit of {HNSW_MAX_NONZERO} "
                f"non-zero elements; an over-budget vector is rejected at INSERT, so the load "
                f"would fail partway through rather than here."
            )
        self._tokenizer = tokenizer
        self._model = model
        self._profile = profile
        self._max_length = max_length

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = DEFAULT_MODEL,
        *,
        top_k: int = HNSW_MAX_NONZERO,
        revision: str | None = None,
        accept_noncommercial_license: bool = False,
        max_length: int = 512,
        device: str | None = None,
    ) -> "SpladeEncoder":
        """Load a published SPLADE checkpoint.

        The licence is checked BEFORE the download. RE-call is MIT, and a non-commercial
        checkpoint arriving by default is a licence violation nobody chose — so
        `naver/splade-v3` has to be opted into by name, in code, with this flag.
        """
        license_id = KNOWN_MODELS.get(model_name)
        if license_id is None:
            raise ValueError(
                f"unknown learned sparse model {model_name!r}; known models are "
                f"{sorted(KNOWN_MODELS)}. Record it in KNOWN_MODELS with its licence first — an "
                f"unrecorded licence is exactly what this check exists to prevent."
            )
        if license_id != "apache-2.0" and not accept_noncommercial_license:
            raise ValueError(
                f"{model_name} is licensed {license_id}, which is not compatible with RE-call's "
                f"MIT distribution for commercial use. Pass accept_noncommercial_license=True to "
                f"use it anyway (benchmark reproduction), or keep the default {DEFAULT_MODEL}."
            )

        from transformers import AutoModelForMaskedLM, AutoTokenizer

        # `trust_remote_code=False` is stated rather than relied upon. It IS the current default,
        # but this is a path that fetches and executes third-party artifacts named by a string,
        # and a default is a decision someone else can change. Writing it down also makes the
        # intent reviewable instead of implicit.
        loader_args = {"revision": revision, "trust_remote_code": False}
        tokenizer = AutoTokenizer.from_pretrained(model_name, **loader_args)
        model = AutoModelForMaskedLM.from_pretrained(model_name, **loader_args).eval()

        # `from_pretrained` loads to CPU. Default to CUDA when it is there, because the ONLY
        # reason to run this on rented hardware is the GPU, and silently leaving the model on
        # CPU turns an expensive box into a slow one with no error to show for it.
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        # `revision` first: it is what the CALLER pinned, and a resolved commit hash that
        # disagrees with it would mean the pin did not take. "unpinned" is a real value, not a
        # placeholder — it records that this encoder's weights are NOT reproducible, which is a
        # fact a published benchmark number needs to carry rather than hide behind a default.
        resolved = revision or getattr(model.config, "_commit_hash", None) or "unpinned"
        profile = SparseProfile(
            profile_id=model_name.replace("/", "__"),
            model_name=model_name,
            artifact_digest=str(resolved),
            dimension=int(model.config.vocab_size),
            top_k=top_k,
            dependencies=(("transformers", _package_version("transformers")),),
        )
        return cls(tokenizer=tokenizer, model=model, profile=profile, max_length=max_length)

    @property
    def profile(self) -> SparseProfile:
        return self._profile

    @property
    def device(self) -> "torch.device":
        """The device the MODEL is on, which is where inputs have to go.

        Read off the model rather than stored, so it stays correct if a caller moves the model
        after construction. Hardcoding CPU here is the expensive mistake: `from_pretrained`
        loads to CPU, so a rented GPU would sit idle while the corpus encoded on the instance's
        CPU. Nothing would error. The vectors would be correct, the run would be ~100x slower,
        and the only symptom is the bill.
        """
        return next(self._model.parameters()).device

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        """One pruned ``{term_id: weight}`` mapping per input, in input order."""
        import torch

        if not texts:
            return []
        encoded = self._tokenizer(
            texts, padding=True, truncation=True,
            max_length=self._max_length, return_tensors="pt",
        )
        device = self.device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self._model(**encoded).logits
        weights = splade_weights(logits, encoded["attention_mask"])
        return [prune_to_top_k(row, self._profile.top_k) for row in weights]


class SparseEncoderProtocol(Protocol):
    """What the indexing helpers actually need from an encoder.

    Stated as a protocol rather than as `SpladeEncoder`, because the tests drive these helpers
    with a deterministic keyword encoder and that is a feature: it keeps the corpus path testable
    without a 500 MB download, and it keeps `torch` out of a lexical-only install.
    """

    @property
    def profile(self) -> SparseProfile: ...

    def encode(self, texts: list[str]) -> list[dict[int, float]]: ...


@dataclass(frozen=True)
class SparseIndexResult:
    """What one indexing pass wrote, and what it could not write.

    `empty_ids` is not a warning to be discarded. It is the ONLY explanation an operator will get
    for `assert_sparse_coverage` finding fewer sidecar rows than chunks, so it is returned rather
    than logged.
    """

    written: int
    empty_ids: list[str]


def store_sparse_vectors(
    store: Any,
    encoder: SparseEncoderProtocol,
    items: Iterable[tuple[str, str]],
    *,
    batch_size: int = 32,
    progress: Callable[[int], None] | None = None,
) -> SparseIndexResult:
    """Encode `(chunk_id, text)` pairs and write them to the learned sparse sidecar.

    The profile id is read off `encoder.profile`, never taken as a separate argument. Vectors
    filed under a name a different model produced score plausibly instead of failing, which is
    precisely what the profile column exists to prevent, so the caller is not given the chance.

    A chunk that encodes to an EMPTY vector is skipped and its id returned. `upsert_sparse`
    refuses an empty mapping (the table's CHECK requires nnz > 0), and it is right to, but that
    refusal belongs at the corpus level where an operator can act on it: see
    `assert_sparse_coverage`. One term-free passage must not kill a whole index.

    `progress` receives the running written count after each batch, so a caller can print
    something during a CPU encode that takes tens of minutes.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}; nothing would be encoded")

    profile_id = encoder.profile.profile_id
    written = 0
    empty_ids: list[str] = []
    batch: list[tuple[str, str]] = []

    def _flush_batch() -> None:
        nonlocal written
        if not batch:
            return
        vectors = encoder.encode([text for _, text in batch])
        payload: dict[str, dict[int, float]] = {}
        for (chunk_id, _text), weights in zip(batch, vectors, strict=True):
            if weights:
                payload[chunk_id] = weights
            else:
                empty_ids.append(chunk_id)
        if payload:
            written += store.upsert_sparse(profile_id, payload)
        batch.clear()
        if progress is not None:
            progress(written)

    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            _flush_batch()
    _flush_batch()

    return SparseIndexResult(written=written, empty_ids=empty_ids)


class SparseCoverageError(RuntimeError):
    """Fewer sidecar rows than chunks. The retrieval leg would answer, thinly and silently."""


def assert_sparse_coverage(
    store: Any, profile_id: str, *, empty_ids: "Iterable[str]" = ()
) -> None:
    """Refuse a corpus whose sidecar is not complete under `profile_id`.

    This is the corpus-level half of the empty-vector decision made in `store_sparse_vectors`,
    and it is the reason skipping a row there is safe. A partially encoded corpus does not error
    on query: the learned leg simply retrieves from the fraction that exists, and the result is
    indistinguishable from a corpus where those passages genuinely did not match.

    `empty_ids` is what `store_sparse_vectors` returned. It does not suppress the refusal, it
    EXPLAINS it: an operator who can see that the missing chunks were term-free can proceed,
    where "1 of 2" alone cannot be told apart from a broken encoder.
    """
    encoded = store.sparse_row_count(profile_id)
    total = store.count()
    if encoded == total:
        return
    message = (
        f"learned sparse sidecar holds {encoded} of {total} chunks under profile "
        f"{profile_id!r}. A query would retrieve from the encoded fraction and report nothing, "
        f"so no result from this corpus may be quoted."
    )
    named = list(empty_ids)
    if named:
        shown = ", ".join(named[:10])
        more = f" (and {len(named) - 10} more)" if len(named) > 10 else ""
        message += f" {len(named)} chunk(s) encoded to an empty vector: {shown}{more}."
    raise SparseCoverageError(message)
