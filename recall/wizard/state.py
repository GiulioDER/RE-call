"""Resumable state: what a previous run of this install already finished.

Building one corpus takes minutes. Measured on this repository, roughly seven for a 1,793-chunk
generation with fastembed on CPU, and the wizard drives three corpora. So a run that dies on the
second one and then rebuilds the first from scratch is not merely slow, it is the difference between
an install a user retries and an install a user abandons.

Worse than slow: every re-run creates a NEW generation and copies every chunk row and embedding into
it, so a crash loop grows the database without bound. Resuming is a storage fix as much as a time
fix.

**What identifies "already done" is a digest, not a timestamp.** A recorded outcome is only reusable
if the thing that produced it would produce the same result now, so the digest covers the inputs that
invalidate previous work and deliberately excludes the ones that do not. The two lists are
`DIGEST_FIELDS` and `IGNORED_FIELDS` below, each with its reasons, and `test_wizard_state.py`
asserts that every config field is in one of them.

**The state is written after EVERY corpus, not at the end of the plan.** Writing once at the end
would lose exactly the case this module exists for: the run that did not reach the end.

⚠️ This does NOT detect a corpus whose FILES changed. The digest covers the configuration, not the
content, so editing a document and re-running reuses the old generation. That is a real limitation
and not a hidden one: `--fresh` discards the state, and the corpus fingerprint inside the generation
is what a caller should compare when it needs content-level truth. Detecting it here would mean
walking and hashing every corpus before deciding to skip it, which costs a large fraction of what
the skip saves.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from recall.atomic_write import atomic_write_bytes
from recall.lineage import canonical_sha256

if TYPE_CHECKING:  # pragma: no cover (import cycle: `from_dict` imports LegacyIndex at call time)
    from recall.wizard.headless import HeadlessConfig, LegacyIndex

from recall.wizard.pipeline import CorpusOutcome

__all__ = ["WizardState", "config_digest", "load_state", "save_state"]

#: Config fields that invalidate previously completed work. Named explicitly rather than "everything
#: except X", so adding a field to `HeadlessConfig` is a decision here rather than a silent one, and
#: `test_wizard_state.py` asserts every field is classified.
DIGEST_FIELDS = (
    "dsn",
    # Selects the database when `dsn` is absent, so a changed location is a different store and
    # nothing recorded against the old one is reusable.
    "data_root",
    "embedder",
    "corpus_version",
    "project",
    "docs_root",
    "code_root",
    "memory_root",
)

#: Fields deliberately excluded: they decide who connects or where configuration is written, not
#: what is built. `project_root` is safe to exclude because the wiring is rewritten on every run
#: regardless of what was reused, so a changed root still produces corrected server blocks.
IGNORED_FIELDS = ("migration_dsn", "serving_role", "fact_write_dsn", "project_root")


def config_digest(config: HeadlessConfig) -> str:
    """A digest over the config fields that would invalidate a completed corpus.

    `Path` is rendered with `as_posix()` rather than `str()`, so a state file written on Windows and
    read on Linux compares equal for the same logical root instead of silently discarding the whole
    run over a separator.
    """
    payload: dict[str, Any] = {}
    for name in DIGEST_FIELDS:
        value = getattr(config, name)
        payload[name] = value.as_posix() if isinstance(value, Path) else value
    # `RECALL_EMBED_PROFILE` takes precedence over the configured embedder when it is set, so a run
    # under another profile built different vectors. Added only when set, so the digest of every
    # existing run without one is unchanged and its state stays resumable.
    if profile := os.environ.get("RECALL_EMBED_PROFILE", "").strip():
        payload["embed_profile"] = profile
    return canonical_sha256(payload)


@dataclass(frozen=True)
class WizardState:
    """Corpora a previous run finished, and the config digest they were finished under."""

    digest: str
    outcomes: tuple[CorpusOutcome, ...] = field(default_factory=tuple)
    indexed: tuple[LegacyIndex, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        object.__setattr__(self, "indexed", tuple(self.indexed))

    def outcome_for(self, tenant: str) -> CorpusOutcome | None:
        """The recorded outcome for `tenant`, or None.

        Only a PROMOTED outcome is returned. A degraded corpus is deliberately not reusable: it was
        left unpromoted precisely so it could be retried, and reusing it would make a re-run a no-op
        for the one corpus the user is re-running to fix.
        """
        for outcome in self.outcomes:
            if outcome.tenant == tenant and outcome.promoted:
                return outcome
        return None

    def indexed_for(self, tenant: str) -> LegacyIndex | None:
        for entry in self.indexed:
            if entry.tenant == tenant:
                return entry
        return None

    def with_outcome(self, outcome: CorpusOutcome) -> WizardState:
        kept = tuple(o for o in self.outcomes if o.tenant != outcome.tenant)
        return replace(self, outcomes=(*kept, outcome))

    def with_indexed(self, entry: LegacyIndex) -> WizardState:
        kept = tuple(e for e in self.indexed if e.tenant != entry.tenant)
        return replace(self, indexed=(*kept, entry))

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {
            "schema_version": 1,
            "digest": self.digest,
            "outcomes": [asdict(o) for o in self.outcomes],
            # The JSON shape is the one the old tuple produced, so existing files still load.
            "indexed": [[e.tenant, e.files, e.chunks] for e in self.indexed],
        }

    @classmethod
    def from_dict(cls, raw: Any) -> WizardState | None:
        """Returns None for anything unreadable, because a bad state file must not stop an install.

        A corrupt or future state file is not an error the operator should have to fix before
        installing: the worst case of ignoring it is doing work that was already done, and the worst
        case of trusting it is skipping work that was not.
        """
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            return None
        digest = raw.get("digest")
        if not isinstance(digest, str) or not digest:
            return None
        if not isinstance(raw.get("outcomes", []), list) or not isinstance(
            raw.get("indexed", []), list
        ):
            return None
        from recall.wizard.headless import LegacyIndex

        known = {f.name for f in fields(CorpusOutcome)}
        outcomes: list[CorpusOutcome] = []
        for item in raw.get("outcomes") or ():
            if not isinstance(item, dict):
                return None
            if any(
                not isinstance(item.get(name, False), bool)
                for name in ("certified", "promoted", "unverified_embedder")
            ):
                return None
            try:
                outcomes.append(CorpusOutcome(**{k: v for k, v in item.items() if k in known}))
            except (TypeError, ValueError):
                return None
        indexed: list[LegacyIndex] = []
        for item in raw.get("indexed") or ():
            if not (isinstance(item, list) and len(item) == 3):
                return None
            tenant, files, chunks = item
            if not isinstance(tenant, str) or not isinstance(files, int) or not isinstance(chunks, int):
                return None
            indexed.append(LegacyIndex(tenant=tenant, files=files, chunks=chunks))
        return cls(digest=digest, outcomes=tuple(outcomes), indexed=tuple(indexed))


def load_state(path: Path, *, digest: str) -> WizardState:
    """The recorded state when it matches `digest`, otherwise an empty one.

    A mismatch is the normal case after the operator changes the embedder or a root, and it is
    handled by starting over rather than by refusing: the recorded generations are still in the
    database and still reachable through `recall generation list`, so nothing is lost that a person
    cannot find.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # ValueError covers JSONDecodeError and a file that is not UTF-8 alike.
        return WizardState(digest=digest)
    state = WizardState.from_dict(raw)
    if state is None or state.digest != digest:
        return WizardState(digest=digest)
    return state


def save_state(path: Path, state: WizardState) -> None:
    """Write atomically, because the state file is written repeatedly mid-run.

    A partial write is worse than no write: `from_dict` would reject it and the whole run's progress
    would be discarded on the next attempt, which is the failure this module exists to prevent.

    The shared writer also fsyncs before the swap and creates a new file 0600: the digest is taken
    over the DSN, password included, so the file is no more public than the config beside it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, (json.dumps(state.to_dict(), indent=2) + "\n").encode("utf-8"))
