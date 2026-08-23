"""Embedder and reranker factories for the MCP server.

Moved out of `recall_mcp.service` unchanged (which re-exports every name here). The logger
deliberately keeps the `mcp.service` channel so nothing downstream of the log stream changes.
"""

from __future__ import annotations

import os
import threading

from recall.embedding_registry import (
    RegisteredProfile,
    find_registered_profile,
    registered_profile_ids,
)
from recall.embeddings import (
    Embedder,
    FastEmbedEmbedder,
    HashingEmbedder,
    REMOTE_MODEL_CODE_OPT_IN,
    resolve_embedder,
)
from recall.observability import get_logger
from recall.profiles import RetrievalAdmission, RetrievalProfile, resolve_retrieval_profile
from recall.rerank import (
    COREB_CODE_RERANKER_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_REVISION,
    KNOWN_RERANKER_REVISIONS,
    RERANKER_MODEL_ALIASES,
    Reranker,
)

_log = get_logger("mcp.service")


HASHING_DIM = 64  # offline HashingEmbedder width; matches the eval/test default


#: Which `RECALL_EMBEDDER` spellings may be paired with a registered profile of each backend.
#:
#: Local profiles keep accepting ONLY `fastembed`, including the qwen3 entry, because that is the
#: pairing every existing deployment already has written down and narrowing or widening it would
#: be an unannounced config break. Hosted profiles take the spellings `resolve_embedder` already
#: understands for the same providers, so an operator moving from an unregistered hosted embedder
#: to a registered one changes one variable and not two.
_PROFILE_EMBEDDER_SPELLINGS: dict[str, frozenset[str]] = {
    "fastembed": frozenset({"fastembed"}),
    "qwen3": frozenset({"fastembed"}),
    "voyage": frozenset({"voyage"}),
    "openai-compat": frozenset({"openai", "openrouter"}),
}


def _warn_if_rejected(entry: RegisteredProfile) -> None:
    """Log the measured verdict when a profile that was rejected is being loaded anyway."""
    if not entry.rejected:
        return
    record = entry.rejection
    assert record is not None  # `rejected` is exactly `rejection is not None`
    _log.warning(
        "embedding profile %s was REJECTED on %s (%s) and is being loaded anyway; "
        "the measured reason was %s",
        entry.profile_id,
        record.decided_on,
        record.reason,
        ", ".join(f"{k}={v}" for k, v in record.measurements),
    )


def make_embedder(name: str, env: dict[str, str] | None = None) -> Embedder:
    """Return the embedder backend by name.

    Registered profiles belong to `recall.embedding_registry`; this boundary resolves those first
    so profile identity stays single-sourced. Without `RECALL_EMBED_PROFILE`, the MCP server
    accepts the shared `recall.embeddings.resolve_embedder` spellings, including explicit cloud
    and research-model aliases.

    `RECALL_EMBED_PROFILE` now names a LOCAL or a HOSTED profile, and the two need different
    inputs, which is why the branch is on the entry rather than on the variable:

    ===============  ==========================  ======================================
    profile kind     RECALL_EMBEDDER             also required
    ===============  ==========================  ======================================
    fastembed/qwen3  ``fastembed``               artifact path env + RECALL_MODEL_SHA256
    voyage           ``voyage``                  VOYAGE_API_KEY
    openai-compat    ``openai`` / ``openrouter`` OPENROUTER_API_KEY
    ===============  ==========================  ======================================

    A hosted profile is deliberately NOT asked for `RECALL_MODEL_CACHE` or `RECALL_MODEL_SHA256`.
    Requiring them was what made an earlier attempt at this feature unreachable in production:
    there is no artifact tree to point at and no bytes to hash, so the only values an operator
    could have supplied would have been invented ones.
    """
    values = dict(os.environ) if env is None else env
    profile = values.get("RECALL_EMBED_PROFILE", "").strip()
    entry = find_registered_profile(profile) if profile else None
    if profile and entry is None:
        # Kept as an env-facing message: the operator set a variable, and naming the
        # variable is more useful than naming the registry they have never heard of.
        raise ValueError(
            f"unknown RECALL_EMBED_PROFILE: {profile!r} "
            f"(registered: {', '.join(registered_profile_ids())})"
        )
    if entry is not None and name not in _PROFILE_EMBEDDER_SPELLINGS[entry.backend]:
        # Spelled as `RECALL_EMBEDDER=<value>` per accepted value rather than as a list, because
        # that is the form an operator pastes and the form the existing test for the local case
        # pins. A backend with two spellings reads "... =openai or RECALL_EMBEDDER=openrouter".
        accepted = " or ".join(
            f"RECALL_EMBEDDER={spelling}"
            for spelling in sorted(_PROFILE_EMBEDDER_SPELLINGS[entry.backend])
        )
        raise ValueError(
            f"RECALL_EMBED_PROFILE={profile!r} is a {entry.backend} profile and needs {accepted}"
        )
    if name == "hashing":
        return HashingEmbedder(dim=HASHING_DIM)
    if entry is not None and entry.hosted:
        _warn_if_rejected(entry)
        # The key is read HERE rather than left to the embedder class's own env lookup, because
        # `make_embedder` accepts an explicit `env` mapping and the class would consult the real
        # process environment instead, so a caller passing a key in `env` would get a confusing
        # "needs an API key" from a call it had just supplied one to.
        return entry.build(api_key=values.get(entry.api_key_env) or None)
    if name == "fastembed":
        if not profile:
            return FastEmbedEmbedder()
        assert entry is not None  # `profile` non-empty and unknown ids already raised above
        artifact_digest = values.get("RECALL_MODEL_SHA256", "")
        artifact_path = values.get(entry.artifact_path_env, "")
        if not artifact_path or not artifact_digest:
            raise ValueError(
                f"profile {profile!r} requires {entry.artifact_path_env} and RECALL_MODEL_SHA256"
            )
        _warn_if_rejected(entry)
        return entry.build(artifact_path=artifact_path, artifact_digest=artifact_digest)
    try:
        return resolve_embedder(name, env=values)
    except ValueError as exc:
        if "unknown embedder" not in str(exc):
            raise
        raise ValueError(
            f"unknown embedder: {name!r} (use 'fastembed', 'hashing', or any "
            "recall.embeddings resolver spelling)"
        ) from exc


def make_profile_embedder(
    profile_id: str, *, shadow: bool = False, env: dict[str, str] | None = None
) -> Embedder:
    """Construct one registered profile, with optional shadow-specific artifact settings."""
    values = dict(os.environ if env is None else env)
    values["RECALL_EMBED_PROFILE"] = profile_id
    # The spelling is DERIVED from the profile, not fixed at "fastembed". `recall.enterprise_cli`
    # calls this with `route.active.embedding_profile`, which is whatever profile the active
    # generation was built under, so hard-coding one backend meant a tenant on a hosted generation
    # could never be served through this path. `sorted(...)[0]` only to make the choice
    # deterministic where a backend accepts more than one spelling; both reach the same builder.
    entry = find_registered_profile(profile_id)
    spelling = "fastembed"
    if entry is not None and entry.hosted:
        spelling = sorted(_PROFILE_EMBEDDER_SPELLINGS[entry.backend])[0]
    if shadow:
        mappings = {
            "RECALL_SHADOW_MODEL_CACHE": "RECALL_MODEL_CACHE",
            "RECALL_SHADOW_MODEL_SHA256": "RECALL_MODEL_SHA256",
            "RECALL_SHADOW_QWEN_MODEL_PATH": "RECALL_QWEN_MODEL_PATH",
        }
        for source, target in mappings.items():
            if source in values:
                values[target] = values[source]
    return make_embedder(spelling, values)


#: Cross-encoder reranking, opt-in via `RECALL_RERANK`.
#:
#: Measured on LOCOMO at n=1,536 (FINDINGS §11): hit@5 **0.671 -> 0.777**, intervals disjoint from
#: the baseline through k=10 — the largest single retrieval gain in this project, and roughly twice
#: the best embedder effect. It closes 57% of the distance to the candidate pool's own ceiling.
#:
#: OFF by default because it costs ~1,050 ms per query on CPU. A memory server that silently
#: quadrupled every query's latency to improve a benchmark would be choosing for the operator.
#: Worth enabling when a human is waiting on the answer; leave it off for high-volume automated
#: retrieval or constrained hardware.
_RERANK_TRUE = frozenset({"1", "true", "yes", "on"})
_RERANK_FALSE = frozenset({"", "0", "false", "no", "off"})


def resolve_reranker(env: dict[str, str] | None = None) -> tuple[str, str | None] | None:
    """`(model, revision)` for the configured reranker, or None when it is off.

    `revision` is None for a cloud model, which has no Hub reference to pin. That is a real
    difference in guarantee, not a missing value, and the type says so rather than hiding it
    behind an empty string.

    Returns a spec rather than an instance so the decision can be tested without importing torch.

    `ms-marco-MiniLM-L-6-v2` is the default because it was *measured* to be the right choice, not
    because it was already there: `bge-reranker-base`, with 12x the parameters and four years newer,
    is statistically indistinguishable at **6.3x** the per-query cost. Reranker selection here is
    about task match — short query against short passage — not model size.

    An unparseable flag is REFUSED rather than read as "off". An operator who asked for reranking
    and silently got an unreranked server would have no way to notice: the failure is fast, quiet
    and looks exactly like success.
    """
    import os as _os

    source = env if env is not None else _os.environ
    raw = source.get("RECALL_RERANK", "").strip().lower()
    if raw in _RERANK_FALSE:
        return None
    if raw not in _RERANK_TRUE:
        raise ValueError(
            f"RECALL_RERANK={raw!r} is not a boolean. Use one of {sorted(_RERANK_TRUE)} to enable "
            f"or leave it unset. Refused rather than treated as off, because a server that quietly "
            f"ignored the flag would look identical to one that honoured it."
        )

    model = source.get("RECALL_RERANK_MODEL")
    if not model:
        return (DEFAULT_RERANKER_MODEL, DEFAULT_RERANKER_REVISION)
    model = RERANKER_MODEL_ALIASES.get(model, model)
    revision = source.get("RECALL_RERANK_REVISION")

    # A cloud model has no Hub reference, so it has no revision to pin. The requirement below is a
    # Hub property, and applying it here would make the Voyage reranker unselectable while looking
    # like a safety check.
    #
    # ⚠️ The guarantee genuinely differs and is not being papered over. `rerank-2.5` is a name
    # resolved on Voyage's side: its weights can change under us in a way a pinned Hub revision
    # cannot. That is a real, smaller guarantee, recorded here so a reader comparing the two
    # rerankers can see it. It is a reason to know what you are choosing, not a reason to refuse.
    if model == "voyage" or model.startswith("voyage:"):
        if revision:
            raise ValueError(
                f"RECALL_RERANK_MODEL={model!r} has no Hub revision to pin, so "
                f"RECALL_RERANK_REVISION={revision!r} cannot be honoured. Accepting it would put a "
                "pin in every trace that pins nothing, which asserts a guarantee that does not "
                "exist. Unset RECALL_RERANK_REVISION for a cloud reranker."
            )
        return (model, None)

    if not revision:
        revision = KNOWN_RERANKER_REVISIONS.get(model)
        if not revision:
            raise ValueError(
                "RECALL_RERANK_MODEL requires RECALL_RERANK_REVISION unless it names a built-in "
                "pinned reranker. An unpinned Hub reference is mutable, and the shipped revision "
                "pin belongs to the shipped weights only — reusing it would name the wrong "
                "artifact in every trace."
            )
    return (model, revision)


def _positive_env(values: dict[str, str], name: str, default: int) -> int:
    raw = values.get(name, "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be positive")
    return parsed


def _remote_model_code_enabled(values: dict[str, str]) -> bool:
    return values.get(REMOTE_MODEL_CODE_OPT_IN, "").strip().lower() in {"1", "true", "yes", "on"}


def _require_remote_model_code_enabled(values: dict[str, str], model: str) -> None:
    if not _remote_model_code_enabled(values):
        raise ValueError(
            f"{model} requires {REMOTE_MODEL_CODE_OPT_IN}=1 because its Transformers loader "
            "executes model repository code"
        )


def _validate_quality_reranker_config(values: dict[str, str]) -> tuple[str, str]:
    """`(model_path, digest)` for the quality profile, or raise. No model is loaded.

    The digest must EQUAL the pin. Accepting whatever the operator typed would make the
    `local_files_only` verification self-referential: it would prove the tree matches the hash of
    itself, which every tree does. The check that means something compares it against a value
    chosen elsewhere.
    """
    from recall.rerank import PINNED_RERANKER_SHA256

    # Normalise ONCE, compare the normalised value, and return the normalised value. Validating
    # one string and handing the loader a different one re-opens the hole this check closed:
    # `verify_artifact` rejects on LENGTH before it lowercases, so a digest that is correct
    # except for a trailing newline (what a dotenv literal block or a padded `.env` line
    # produces) passed startup and then raised on the first search.
    model_path = values.get("RECALL_RERANK_PATH", "").strip()
    digest = values.get("RECALL_RERANK_SHA256", "").strip().lower()
    if not model_path or not digest:
        raise ValueError("quality profile requires RECALL_RERANK_PATH and RECALL_RERANK_SHA256")
    if digest != PINNED_RERANKER_SHA256:
        raise ValueError(
            f"RECALL_RERANK_SHA256 does not match the reranker pinned to the quality profile "
            f"(expected {PINNED_RERANKER_SHA256}). A different artifact tree is a different "
            f"model and needs its own registered experiment, not a reused profile."
        )
    return model_path, digest


def _new_reranker(env: dict[str, str] | None = None) -> "Reranker | None":  # pragma: no cover
    """Instantiate the configured reranker, or None. Imports torch only when actually enabled."""
    values = dict(os.environ) if env is None else env
    profile = resolve_retrieval_profile(values)
    if profile.name == "fast":
        return None
    if profile.name == "quality":
        model_path, digest = _validate_quality_reranker_config(values)
        from recall.rerank import CrossEncoderReranker

        return CrossEncoderReranker(
            model=model_path,
            revision=None,
            local_files_only=True,
            artifact_sha256=digest,
            inference_threads=profile.inference_threads,
        )
    if profile.name == "code":
        rerank_values = dict(values)
        rerank_values.setdefault("RECALL_RERANK", "1")
        rerank_values.setdefault("RECALL_RERANK_MODEL", "coreb-code")
        spec = resolve_reranker(rerank_values)
        assert spec is not None
        model, revision = spec
        if model != COREB_CODE_RERANKER_MODEL:
            raise ValueError("the code retrieval profile requires RECALL_RERANK_MODEL=coreb-code")
        _require_remote_model_code_enabled(values, "coreb-code")
        from recall.rerank import QwenYesNoReranker

        return QwenYesNoReranker(
            model=model,
            revision=revision,
            inference_threads=profile.inference_threads,
            batch_size=_positive_env(values, "RECALL_RERANK_BATCH_SIZE", 4),
            trust_remote_code=True,
        )
    spec = resolve_reranker(values)
    if spec is None:
        return None
    model, revision = spec
    if model == COREB_CODE_RERANKER_MODEL:
        raise ValueError("coreb-code requires RECALL_RETRIEVAL_PROFILE=code")
    from recall.rerank import CrossEncoderReranker

    model, revision = spec

    # Voyage primary, local cross-encoder fallback. The fallback is not politeness: a Voyage outage
    # would otherwise take retrieval down entirely, and reranking is the largest single measured
    # retrieval gain. `FallbackReranker` counts and logs every fallback, so a run cannot silently
    # measure a blend of two rerankers — the confound named in this branch's pre-registration.
    #
    # The fallback is built EAGERLY, alongside the primary, rather than on first failure. Building a
    # cross-encoder downloads and loads weights; doing that at the moment Voyage is already failing
    # turns one outage into a cold start under load, which is when the process can least afford it.
    if model == "voyage" or model.startswith("voyage:"):
        from recall.rerank import FallbackReranker, reranker_from_name

        return FallbackReranker(
            primary=reranker_from_name(model),
            fallback=CrossEncoderReranker(
                model=DEFAULT_RERANKER_MODEL, revision=DEFAULT_RERANKER_REVISION
            ),
        )

    return CrossEncoderReranker(model=model, revision=revision)


#: ONE reranker per worker process, built once, keyed by the resolved profile.
#:
#: `lru_cache` was not enough and the difference is not academic. A cache lookup is not a
#: construction lock: N threads arriving on a cold cache all miss, all call the factory, and all
#: load their own copy of a cross-encoder. That is hundreds of megabytes per surplus copy, at the
#: one moment the process is least able to afford it — a cold start under load. The lock makes
#: "one per worker" a property of the code rather than of the arrival pattern.
#:
#: Keyed by profile rather than stored in a single slot so a process whose profile changes (only
#: tests do this; production selects one profile per process) cannot be served a reranker built
#: for the other one.
_RERANKER_LOCK = threading.Lock()
#: Maps profile name to the built reranker, or to a `(type, args)` description of the failure.
#:
#: FAILURES are cached too. Caching only successes meant a bad artifact re-ran the full
#: tree-SHA256 over a several-hundred-megabyte model directory on EVERY client search, while
#: holding both this lock and an admission running slot: a configuration error turned into a
#: self-inflicted disk-and-CPU load that grew with traffic.
#:
#: A DESCRIPTION rather than the exception object, and this is not fastidiousness. Re-raising one
#: instance appends the current frame to its `__traceback__` every time, and each retained frame
#: pins its locals — which on this path include the caller's QUERY TEXT and the store. That would
#: be an unbounded memory leak that also retains user text for the process lifetime, on the very
#: path the caching was added to make cheap. Caching the instance and clearing its traceback at
#: each raise would preserve more state, but two threads raising one shared object race on
#: `__traceback__`; a fresh instance per raise cannot.
#:
#: ⚠️ Known fidelity limit: `(type, args)` does not round-trip the `OSError` family exactly. A bad
#: `RECALL_RERANK_PATH` reports the offending path on the FIRST failure and drops it (along with
#: `filename` / `winerror`) on cached repeats. The error class and the reason survive; the path
#: does not. Accepted because the first occurrence is the diagnostic one and thread safety is not.
_RERANKERS: dict[str, "Reranker | None | tuple[type[Exception], tuple[object, ...]]"] = {}


def _reset_reranker_cache() -> None:
    """Drop the per-process reranker. For tests — a server should never need this."""
    with _RERANKER_LOCK:
        _RERANKERS.clear()


def _build_reranker(
    profile: RetrievalProfile | None = None, env: dict[str, str] | None = None
) -> "Reranker | None":
    if env is not None:  # explicit environment: an ad-hoc instance, never the shared one
        return _new_reranker(env)
    name = (profile or resolve_retrieval_profile()).name
    with _RERANKER_LOCK:
        if name not in _RERANKERS:
            # `Exception`, deliberately not `BaseException`. A configuration error is
            # deterministic and caching its verdict is right; a `KeyboardInterrupt` or a
            # `SystemExit` arriving during a cold build says nothing about the artifact, and
            # caching it would turn a transient event into a process-lifetime outage.
            try:
                _RERANKERS[name] = _new_reranker()
            except Exception as exc:
                _RERANKERS[name] = (type(exc), exc.args)
                raise
        cached = _RERANKERS[name]
    if isinstance(cached, tuple):
        failure_type, failure_args = cached
        raise failure_type(*failure_args)
    return cached


_ADMISSION_LOCK = threading.Lock()
_ADMISSIONS: dict[tuple[str, int, int, int], RetrievalAdmission] = {}


def _admission(profile: RetrievalProfile) -> RetrievalAdmission:
    """The admission queue for one profile.

    Keyed on `queue_identity` rather than on the whole profile, and built under a lock rather
    than memoised with `lru_cache`, for the two reasons this module already argues for the
    reranker. A cache lookup is not a construction lock, so concurrent cold-start callers could
    each receive their own full-capacity queue; and an `lru_cache` keyed on the whole profile
    made `inference_threads` part of a queue's identity, so two profiles that mean the same queue
    got two of them. Both defects raise the enforced concurrency bound silently, which is the one
    direction a bound must never move on its own.

    Fast and quality still hold SEPARATE budgets: saturating one cannot shed requests on the
    other. In production only one profile is resolved, so this is one queue.
    """
    key = profile.queue_identity
    with _ADMISSION_LOCK:
        existing = _ADMISSIONS.get(key)
        if existing is None:
            existing = _ADMISSIONS[key] = RetrievalAdmission(profile)
        return existing
