"""Learned sparse (SPLADE) retrieval: encoder, identity, and pruning."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import closing
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

        ⚠️ This stability guarantee covers the CACHE KEY, not sidecar coverage. The sparse
        sidecar (`recall.store.PgVectorStore.sparse_covered_sources`) is keyed on `profile_id`
        alone, which is not this fingerprint: `top_k` and the checkpoint's pinned `revision`
        (folded into `artifact_digest`) can both change without `profile_id` changing, and the
        sidecar has no way to notice. See that method's docstring for what that costs.
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
@dataclass(frozen=True)
class ModelLicense:
    """What a checkpoint is licensed under, and what attributing it correctly requires.

    The SPDX id alone is not enough to COMPLY. CC BY-NC-SA 4.0 obliges a user to give appropriate
    credit, link the licence, and indicate whether changes were made, so those three have to be
    data we can emit next to a result, not a remark in a comment that never reaches an artifact.
    `changes` is the third of those and is the one most easily forgotten, because "we only ran
    inference" still has to be stated rather than assumed.
    """

    creator: str
    license_id: str
    license_url: str
    source_url: str
    changes: str

    @property
    def is_commercial_ok(self) -> bool:
        return self.license_id == "apache-2.0"


#: Every checkpoint this encoder will load, with what using it obliges. An unrecorded licence is
#: refused rather than assumed permissive; see `SpladeEncoder.from_pretrained`.
KNOWN_MODELS: dict[str, ModelLicense] = {
    "prithivida/Splade_PP_en_v1": ModelLicense(
        creator="Prithivi Da",
        license_id="apache-2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        source_url="https://huggingface.co/prithivida/Splade_PP_en_v1",
        changes="No modification. Used for inference only, to produce term weights.",
    ),
    "naver/splade-v3": ModelLicense(
        creator="Naver Corporation",
        license_id="cc-by-nc-sa-4.0",
        license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
        source_url="https://huggingface.co/naver/splade-v3",
        changes="No modification. Used for inference only, to produce term weights.",
    ),
    # SPLADE++ EnsembleDistil. Same BERT MLM architecture and 30522 vocabulary as the two above,
    # so it is a drop-in for this encoder, and stronger than the default (MRR@10 38.3 vs 37.22).
    # It exists here because `naver/splade-v3` is a GATED repo: reaching it needs an approved
    # HuggingFace account, not merely the licence flag. This is the closest ungated substitute
    # for the "is our checkpoint the weak link?" control.
    "naver/splade-cocondenser-ensembledistil": ModelLicense(
        creator="Naver Corporation",
        license_id="cc-by-nc-sa-4.0",
        license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
        source_url="https://huggingface.co/naver/splade-cocondenser-ensembledistil",
        changes=(
            "No modification to the model. Weights were loaded unchanged and used for inference "
            "only; the pruning to a top-k budget is applied to this run's OUTPUT vectors, not to "
            "the checkpoint."
        ),
    ),
}


def attribution_notice(model_name: str) -> str:
    """The credit line a result computed with `model_name` has to carry.

    Written next to any artifact the model contributed to, so the obligation travels with the
    number rather than living in a source file the reader never opens. Covers the Attribution
    term (credit, licence link, changes stated); NonCommercial is enforced upstream by
    `from_pretrained`'s gate, and ShareAlike binds only if the vectors are redistributed, which
    is why the notice names them.
    """
    entry = KNOWN_MODELS.get(model_name)
    if entry is None:
        raise ValueError(f"no licence recorded for {model_name!r}; cannot attribute it")
    lines = [
        f"Term weights in this artifact were produced with {model_name}",
        f"  Creator: {entry.creator}",
        f"  Source:  {entry.source_url}",
        f"  Licence: {entry.license_id} ({entry.license_url})",
        f"  Changes: {entry.changes}",
    ]
    if not entry.is_commercial_ok:
        lines += [
            "  Use:     NON-COMMERCIAL. Benchmark reproduction and research only. This checkpoint",
            "           is not shipped with RE-call and is not the default; RE-call's default is",
            f"           {DEFAULT_MODEL} (apache-2.0).",
            "  ShareAlike: if the derived sparse vectors are redistributed, they must carry this",
            "           same licence. They are not redistributed by this benchmark.",
        ]
    return "\n".join(lines)

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
        entry = KNOWN_MODELS.get(model_name)
        if entry is None:
            raise ValueError(
                f"unknown learned sparse model {model_name!r}; known models are "
                f"{sorted(KNOWN_MODELS)}. Record it in KNOWN_MODELS with its licence first — an "
                f"unrecorded licence is exactly what this check exists to prevent."
            )
        if not entry.is_commercial_ok and not accept_noncommercial_license:
            raise ValueError(
                f"{model_name} is licensed {entry.license_id}, which is not compatible with "
                f"RE-call's MIT distribution for commercial use. Benchmark reproduction and "
                f"research ARE permitted by that licence: pass accept_noncommercial_license=True "
                f"to proceed, and the result must carry attribution_notice({model_name!r}). "
                f"Otherwise keep the default {DEFAULT_MODEL}."
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


def _validated_batch_size(batch_size: Any) -> int:
    """The one place a batch size is checked, shared by both entry points.

    Shared rather than duplicated because `backfill_learned_sparse` derives TWO values from this
    argument — the `iter_chunks` FETCH size and the encode batch — and the moment each validates
    its own copy they are free to disagree about what the caller asked for.

    The `isinstance` half is not decoration. `None` is the realistic mistake (an unset option
    threaded through) and a bare `batch_size < 1` answers it with `TypeError: '<' not supported
    between instances of 'NoneType' and 'int'`, which names neither the argument nor the rule.
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(
            f"batch_size must be >= 1, got {batch_size!r}; nothing would be encoded"
        )
    return batch_size


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

    The iterator is CLOSED on the way out, and that matters most on the failing path. `items` is
    routinely a generator over `store.iter_chunks()`, a server-side cursor holding one pooled
    connection open inside a transaction for its whole scan. If `encoder.encode` or
    `store.upsert_sparse` raises mid-batch, a bare `for` loop abandons that generator suspended
    mid-yield, still holding the connection, and cleanup falls to CPython refcounting finalising
    the orphan. That is usually prompt but is NOT guaranteed: anything retaining the traceback —
    a retry loop storing the exception, a caller logging `sys.exc_info()`, a test framework's
    `excinfo` — keeps this frame alive and `items` with it, and the connection stays checked out
    for as long as that reference lives. Closing here is what makes the release deterministic
    rather than a property of the garbage collector.

    This belongs at this level because this function is the ACTUAL consumer: it is what advances
    the iterator, so it is what owes it a close. A caller wrapping its own generator is welcome
    to, and `backfill_learned_sparse` does, but that protects only that caller.
    """
    batch_size = _validated_batch_size(batch_size)

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

    items_iter = iter(items)
    try:
        for item in items_iter:
            batch.append(item)
            if len(batch) >= batch_size:
                _flush_batch()
        _flush_batch()
    finally:
        # `items` is an `Iterable`, not a generator: a list's iterator has no `close`, and the
        # signature deliberately accepts both. `contextlib.closing` would demand the method.
        close = getattr(items_iter, "close", None)
        if close is not None:
            close()

    return SparseIndexResult(written=written, empty_ids=empty_ids)


class SparseCoverageError(RuntimeError):
    """The sidecar disagrees with the corpus under a profile, in either direction.

    Fewer sidecar rows than chunks: the retrieval leg would answer, thinly and silently. More
    sidecar rows than chunks: the sidecar holds rows for chunks that no longer exist, because it
    keys its parent as a column value rather than a relation, so nothing cascades when a chunk is
    removed. Both are real faults and both raise; only the message differs.
    """


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

    The two counts are separate round trips (`sparse_row_count`, then `count()`), so this assumes
    indexing has quiesced: a concurrent indexer running between the two queries can make them
    disagree transiently. No locking is added for that; the caller is responsible for calling
    this only once writes have settled.
    """
    encoded = store.sparse_row_count(profile_id)
    total = store.count()
    if encoded == total:
        return
    if encoded < total:
        message = (
            f"learned sparse sidecar holds {encoded} of {total} chunks under profile "
            f"{profile_id!r}. A query would retrieve from the encoded fraction and report "
            f"nothing, so no result from this corpus may be quoted."
        )
        named = list(empty_ids)
        if named:
            shown = ", ".join(named[:10])
            more = f" (and {len(named) - 10} more)" if len(named) > 10 else ""
            message += f" {len(named)} chunk(s) encoded to an empty vector: {shown}{more}."
        raise SparseCoverageError(message)
    raise SparseCoverageError(
        f"learned sparse sidecar holds {encoded} rows under profile {profile_id!r}, more than "
        f"the {total} chunks in the corpus. The sidecar keys its parent chunk table as a column "
        f"value, not a relation, so nothing cascades when a chunk row is removed: these are "
        f"orphaned rows for chunks that no longer exist. The erasure paths (`replace_sources`, "
        f"`delete_sources`, `delete_sources_across`, `generations.forget`) now scrub the "
        f"sidecar in the same transaction, so on a current build the likely cause is rows "
        f"orphaned BEFORE that fix landed, or a scrub bypassed by direct SQL. This still "
        f"refuses, because "
        f"a sidecar that disagrees with the corpus is a real fault: at least {encoded - total} "
        f"sidecar row(s) are orphaned. This compares counts, not id sets, so an overcount is "
        f"not evidence that coverage is complete: a separately unencoded chunk can still be "
        f"hiding inside it."
    )


def backfill_learned_sparse(
    store: Any,
    encoder: SparseEncoderProtocol,
    *,
    batch_size: int = 32,
    progress: Callable[[int], None] | None = None,
) -> SparseIndexResult:
    """Encode every chunk already in `store` into the learned sparse sidecar.

    This is the path that reaches corpora indexed before `Indexer` could write the sidecar, which
    is every corpus that exists today. It streams `store.iter_chunks()`, a server-side cursor
    that excludes the dense vector, so a corpus larger than memory is fine.

    `batch_size` is validated ONCE, here, and the single validated value is then both the cursor's
    FETCH size and the encode batch. It used to be clamped with `max(batch_size, 1)` for the
    cursor and passed through raw to `store_sparse_vectors`, which left one argument feeding two
    separately-derived meanings: nothing today can tell them apart, because every value the clamp
    would change is a value the other side rejects, but that is a coincidence of the current
    validation order rather than a property, and it costs nothing to not depend on it.

    IDEMPOTENT, not resumable. `upsert_sparse` is ON CONFLICT DO UPDATE, so re-invoking simply
    re-encodes. Skipping ids already present would need a `store.sparse_ids(profile_id)` this
    store does not have, and at the corpus sizes this serves it would buy nothing. That is a
    decision, not an oversight.

    `store.iter_chunks()` holds a pool-borrowed connection open inside an explicit
    `conn.transaction()`, with a named server-side cursor bound to it, for the entire life of the
    generator. This is the one call site that drives that generator through fallible work,
    `encoder.encode` and `store.upsert_sparse`, rather than materialising it or reading plain
    attributes. If either raises mid-batch, the generator would otherwise be abandoned mid-yield,
    still holding its transaction and its pooled connection open, reclaimed only whenever CPython
    happens to garbage-collect it, which is not promptly once a traceback is retained anywhere up
    the stack. `closing` forces the generator's own cleanup (rolling back the transaction and
    releasing the connection) on the way out, success or failure, so this function owns the
    resource it created instead of leaving that to its caller.

    On the connection mode `PgVectorStore` itself recommends as the default (no `pool_size`, no
    `shared_pool`), that held connection is the SAME one `store.upsert_sparse` writes through:
    `_with_retry` runs each write directly on `self._direct`, which is the very connection
    `iter_chunks()` is holding inside its open `conn.transaction()`. So on that path the writes do
    not commit independently, they JOIN the reader's transaction: this whole function, the read
    cursor and every write it drives, is ONE all-or-nothing transaction held open for the duration
    of the encode. A failure on chunk 900 of 1000 rolls back chunks 1 through 899 as well, not
    only the batch that failed. (Pooled and shared-pool stores borrow a SEPARATE connection per
    `upsert_sparse` call, so this does not apply there.)
    """
    batch_size = _validated_batch_size(batch_size)
    with closing(store.iter_chunks(batch_size=batch_size)) as chunks:
        return store_sparse_vectors(
            store,
            encoder,
            ((chunk.id, chunk.text) for chunk in chunks),
            batch_size=batch_size,
            progress=progress,
        )


#: What BERT-base fp32 inference needs with headroom for activations at batch 32. Stated as a
#: number rather than computed, and exposed as a parameter, because a caller running a larger
#: checkpoint or a bigger batch has a different answer and should not have to edit this file.
DEFAULT_REQUIRED_VRAM_MB = 2048

SPARSE_DEVICES = ("auto", "cpu", "cuda")


class SparseDeviceError(RuntimeError):
    """A device was asked for by name and cannot be used."""


@dataclass(frozen=True)
class DeviceReport:
    """Everything the device decision was made from, kept so it can be printed and stamped.

    `learned_sparse_encode_ms_mean` is a transformer forward pass, so its value on CPU and on GPU
    are measurements of different things. An artifact that does not record which one it was cannot
    be compared against another, which is why this whole object reaches provenance rather than
    only the resolved string.
    """

    requested: str
    resolved: str
    torch_cuda_build: str | None
    device_name: str | None
    capability: tuple[int, int] | None
    supported_architectures: tuple[str, ...]
    free_vram_mb: int | None
    refusal: str | None


def device_refusal(
    *,
    cuda_build: str | None,
    device_count: int,
    capability: tuple[int, int] | None,
    arch_list: tuple[str, ...],
    free_vram_mb: int | None,
    required_vram_mb: int = DEFAULT_REQUIRED_VRAM_MB,
) -> str | None:
    """Why CUDA cannot be used, or `None` when it can.

    A PURE function over the facts, deliberately. Every branch is then reachable from a test on a
    box with no GPU and no CUDA build, which is the only way this guard gets shown FIRING rather
    than shown running. The collector that gathers these facts has no logic in it.

    The checks are ordered so each names its own fix. A CPU-only wheel and an absent card need
    different actions, and telling someone with a working card that they have no GPU sends them
    to the wrong one.
    """
    if cuda_build is None:
        return (
            "torch is a CPU-only build (torch.version.cuda is None), so no GPU is reachable "
            "regardless of what hardware is present. Install a CUDA build of torch."
        )
    if device_count < 1:
        return (
            f"torch is built against CUDA {cuda_build} but reports no CUDA device. The driver, "
            f"the container's device mapping or CUDA_VISIBLE_DEVICES is where to look."
        )
    if capability is not None:
        arch = f"sm_{capability[0]}{capability[1]}"
        if arch_list and arch not in arch_list:
            return (
                f"this card is compute capability {capability[0]}.{capability[1]} ({arch}) and "
                f"the installed torch was not built for it. It carries {', '.join(arch_list)}. "
                f"A wheel without the architecture does not decline politely, so this refuses "
                f"here instead. Install a torch build listing {arch}, or pass --sparse-device cpu."
            )
    if free_vram_mb is not None and free_vram_mb < required_vram_mb:
        return (
            f"only {free_vram_mb} MiB of VRAM is free and this needs about {required_vram_mb} "
            f"MiB. Encoding would fail partway through a corpus rather than here."
        )
    return None


def inspect_sparse_device(
    requested: str = "auto", required_vram_mb: int = DEFAULT_REQUIRED_VRAM_MB
) -> DeviceReport:
    """Read the device facts off torch and apply `device_refusal` to them.

    `requested="cpu"` short-circuits before importing torch. Otherwise asking for CPU on a box
    with no torch would import torch in order to decide it did not need torch.
    """
    if requested == "cpu":
        return DeviceReport(
            requested=requested, resolved="cpu", torch_cuda_build=None, device_name=None,
            capability=None, supported_architectures=(), free_vram_mb=None, refusal=None,
        )

    try:
        import torch
    except ImportError:
        return DeviceReport(
            requested=requested, resolved="cpu", torch_cuda_build=None, device_name=None,
            capability=None, supported_architectures=(), free_vram_mb=None,
            refusal="torch is not installed; the learned sparse path needs the `sparse` extra",
        )

    cuda_build = torch.version.cuda
    device_count = torch.cuda.device_count() if cuda_build else 0
    name = None
    capability = None
    free_vram_mb = None
    arch_list: tuple[str, ...] = ()
    if device_count:
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        arch_list = tuple(torch.cuda.get_arch_list())
        # `mem_get_info` returns (free, total) in bytes. FREE rather than total: another process
        # holding the card is the common case on a shared box, and total would say yes to a card
        # with nothing left to give.
        free_bytes, _total = torch.cuda.mem_get_info(0)
        free_vram_mb = int(free_bytes // (1024 * 1024))

    refusal = device_refusal(
        cuda_build=cuda_build, device_count=device_count, capability=capability,
        arch_list=arch_list, free_vram_mb=free_vram_mb, required_vram_mb=required_vram_mb,
    )
    return DeviceReport(
        requested=requested, resolved="cpu" if refusal else "cuda",
        torch_cuda_build=cuda_build, device_name=name, capability=capability,
        supported_architectures=arch_list, free_vram_mb=free_vram_mb, refusal=refusal,
    )


def resolve_sparse_device(
    requested: str = "auto",
    required_vram_mb: int = DEFAULT_REQUIRED_VRAM_MB,
    *,
    report: DeviceReport | None = None,
) -> str:
    """The device string for `SpladeEncoder.from_pretrained`, refusing a named GPU it cannot use.

    `auto` means "use it if it is there", so a refusal is information and the answer is `cpu`.
    `cuda` is a STATEMENT about the run, and answering `cpu` to it would make that statement false
    while producing correct vectors roughly a hundred times more slowly, with nothing to show for
    it. See the note on `SpladeEncoder.device`.

    `report`, when given, is used AS IS instead of calling `inspect_sparse_device` again. Without
    this, a caller that also wants the report for its own provenance (as `store_latency_share.py`
    does) ends up taking two separate live `torch.cuda` readings: one to build the report it
    stamps into the artifact, one taken here to decide. Near a VRAM threshold on a real GPU those
    two reads are not guaranteed to agree, so the reading that drove the decision and the reading
    that gets published could describe two different moments. Passing the report through makes
    them the SAME reading. Omitted, behaviour is unchanged: one fresh read, exactly as before.
    """
    if requested not in SPARSE_DEVICES:
        raise ValueError(f"device must be one of {SPARSE_DEVICES}, got {requested!r}")
    if report is None:
        report = inspect_sparse_device(requested, required_vram_mb=required_vram_mb)
    if requested == "cuda" and report.refusal:
        raise SparseDeviceError(f"--sparse-device cuda was requested but {report.refusal}")
    return report.resolved
