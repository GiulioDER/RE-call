"""Give a locally-downloaded embedder an identity `generation build` will accept as immutable.

`EmbedderIdentity` requires a provider revision or an artifact digest. Nothing in this tree pins a
revision for the default embedder, so without this a generation could only be built
`--unverified-development` and recorded `unverified_reason: "explicit development build"` in place
of any real provenance.

⚠️ **This does not unblock anything, and an earlier version of this docstring claimed it did.**
Measured directly: a verified and an unverified generation over one corpus, both promoted, both
searched under `RECALL_ENV=production`, both refused identically with `CALIBRATION_MISSING`.
`verified` is not consulted on the serving path. `require_production_identity` does refuse an
unverified pipeline, but only when BUILDING under `RECALL_ENV=production`, and that path is
unreachable for a local corpus because the `file://` manifest check fires first. So this buys real
provenance and a future S3 build, not a fix.

**Digesting what was downloaded is a better identity than a Hub revision, not a substitute for
one.** fastembed serves `BAAI/bge-small-en-v1.5` out of `qdrant/bge-small-en-v1.5-onnx-q`, a
quantised ONNX repackaging, so a commit SHA for the BAAI repo would name bytes nobody loads. The
same asymmetry runs the other way and is the subtler trap: the snapshot SHA in the cache path is a
commit in the *qdrant* repo, so recording it under the BAAI name produces a revision nobody can
resolve. Both are kept, attributed to the repository each belongs to, and `lineage_revision` hands
out the revision only when the two agree.

Every function here follows the wizard probe convention: it returns rather than raises. The
consumer's fallback is `--unverified-development` plus a message, which is worse than a verified
build and much better than an install that dies on a diagnostic.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recall.observability import get_logger

_log = get_logger("wizard.identity")

#: fastembed lays its cache out as `<cache>/models--<org>--<repo>/snapshots/<revision>/`. The
#: directory name under `snapshots` is the provider's published revision, so it is read rather than
#: guessed. A cache layout without it still yields a usable identity from the digest alone.
_SNAPSHOT_PARENT = "snapshots"

#: Where to look for the artifact directory on a resolved embedder, in order. These reach into
#: fastembed's internals, which is why every one is optional and failure is not an error: the
#: attribute names are not ours and can move between releases.
_MODEL_DIR_PATHS: tuple[tuple[str, ...], ...] = (
    ("_model", "model", "_model_dir"),
    ("_model", "model", "model_dir"),
    ("_model", "_model_dir"),
    ("model", "_model_dir"),
    ("_model_dir",),
)


@dataclass(frozen=True)
class ArtifactIdentity:
    """What a locally-provisioned embedder can prove about itself."""

    provider: str
    #: The model as the caller asked for it, e.g. `BAAI/bge-small-en-v1.5`.
    model: str
    #: The repository the cached bytes actually came from, e.g. `qdrant/bge-small-en-v1.5-onnx-q`.
    #: Usually NOT the same as `model`, which is the whole reason this field exists.
    source_repo: str | None
    #: The snapshot revision, which belongs to `source_repo` and not necessarily to `model`.
    revision: str | None
    artifact_digest: str
    path: Path

    @property
    def lineage_revision(self) -> str | None:
        """The revision only when it can be resolved against the recorded model.

        fastembed serves `BAAI/bge-small-en-v1.5` out of `qdrant/bge-small-en-v1.5-onnx-q`, so the
        snapshot SHA is a commit in the qdrant repo and does not exist in the BAAI one. Writing it
        into a lineage record under the BAAI name produces a revision nobody can resolve — the
        same false-provenance mistake as pinning a Hub SHA for weights nobody loads, made from the
        other direction. The digest alone already makes the identity immutable, so `None` here
        costs nothing.
        """
        if self.revision and self.source_repo and self.source_repo == self.model:
            return self.revision
        return None


def _dig(obj: Any, attrs: tuple[str, ...]) -> Any | None:
    for attr in attrs:
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def _model_directory(embedder: Any) -> Path | None:
    for attrs in _MODEL_DIR_PATHS:
        found = _dig(embedder, attrs)
        if not found:
            continue
        try:
            path = Path(str(found))
        except Exception:
            continue
        if path.is_dir():
            return path
    return None


#: A HuggingFace commit id. Checked so that a user-provisioned `.../snapshots/latest/` does not
#: get promoted into a "published revision": `EmbedderIdentity` accepts any non-empty string and
#: applies no format check of its own, so the validation has to happen here or nowhere.
_REVISION = re.compile(r"^[0-9a-f]{40}$")

#: HuggingFace names a cache entry `models--<org>--<repo>`.
_CACHE_PREFIX = "models--"


def _revision_from(path: Path) -> str | None:
    """The published revision, when the cache layout names a plausible one."""
    try:
        if path.parent.name != _SNAPSHOT_PARENT:
            return None
        return path.name if _REVISION.match(path.name) else None
    except Exception:
        return None


def _source_repo_from(path: Path) -> str | None:
    """The repository the snapshot belongs to, read from `.../models--<org>--<repo>/snapshots/…`."""
    try:
        entry = path.parent.parent.name
        if not entry.startswith(_CACHE_PREFIX):
            return None
        return entry[len(_CACHE_PREFIX) :].replace("--", "/") or None
    except Exception:
        return None


def _artifact_digest(directory: Path) -> str:
    """Digest a HuggingFace snapshot: content-addressed, symlink-following, OS-independent.

    Deliberately NOT `recall.embeddings.artifact_tree_sha256`, for two reasons that only appear on
    the platform this is deployed on:

    * **It refuses a symlink leaving its root**, and on Linux and macOS every file in an HF
      snapshot IS such a symlink, pointing at `../../blobs/<sha>`. Windows falls back to real
      copies, which is why the first version of this module measured fine locally and would have
      returned `None` on every Linux install. That containment check is right where it lives — it
      guards operator-provisioned enterprise artifacts — and wrong for a cache whose layout is
      symlinks by design.
    * **It sorts `Path` objects**, and `PureWindowsPath` comparison is case-folded while POSIX is
      not, so `README.md` sorts differently on the two and identical bytes produce two digests.
      Ordering here is by the POSIX-form relative path, which is the same everywhere.

    Following symlinks means a hostile cache could point at an unrelated file. That is acceptable
    here and not in the enterprise path: this reads a cache the user's own process just populated,
    and the digest is recorded rather than served.
    """
    files = sorted(
        (p for p in directory.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(directory).as_posix(),
    )
    if not files:
        raise ValueError(f"model artifact has no files: {directory}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\x00")
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):  # noqa: B023
                digest.update(block)
    return digest.hexdigest()


def artifact_identity_for(embedder: Any) -> ArtifactIdentity | None:
    """What `embedder` can prove about the weights it loaded, or `None`.

    `None` is a real answer, not an error: `hashing` has no weights at all, and `voyage` and
    `openai` keep theirs on somebody else's machine. The caller acts on it by building
    `--unverified-development` and saying so, rather than by failing.
    """
    try:
        # Guarded on the class, because `provider` below is written verbatim into a lineage record
        # that is treated as immutable evidence. The attribute chains probed above are generic
        # (`_model_dir` is not a fastembed-only name), so without this a SentenceTransformer-backed
        # embedder that grew a model directory would be stamped with fastembed provenance.
        if type(embedder).__name__ != "FastEmbedEmbedder":
            return None

        directory = _model_directory(embedder)
        if directory is None:
            return None

        # Raises on an empty directory, which means "not an artifact I can name" — exactly `None`.
        digest = _artifact_digest(directory)

        model = str(getattr(embedder, "name", "") or "")
        if not model:
            return None

        return ArtifactIdentity(
            provider="fastembed",
            model=model,
            source_repo=_source_repo_from(directory),
            revision=_revision_from(directory),
            artifact_digest=digest,
            path=directory,
        )
    except Exception as exc:  # noqa: BLE001 - a probe returns, it does not raise
        _log.warning("could not resolve an artifact identity (%s)", type(exc).__name__)
        return None
