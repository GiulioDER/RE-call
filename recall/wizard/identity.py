"""Give a locally-downloaded embedder an identity `generation build` will accept as immutable.

`EmbedderIdentity` requires a provider revision or an artifact digest; without one, a generation
can only be built `--unverified-development`, and `generations.py` refuses `allow_unverified` under
`RECALL_ENV=production`. Nothing in this tree pins a revision for the default embedder, so the
wizard could build a generation only in the mode its own design does not serve `docs` and `code`
in.

Both halves of the fix were already here and simply never joined up: `artifact_tree_sha256` digests
a provisioned model directory, and an artifact digest is immutable on its own.

**Digesting what was downloaded is a better identity than a Hub revision, not a substitute for
one.** fastembed fetches `qdrant/bge-small-en-v1.5-onnx-q`, a quantised ONNX repackaging, so a
commit SHA for `BAAI/bge-small-en-v1.5` would have named bytes nobody loads. The digest names the
bytes; the snapshot directory, where the cache layout provides one, names the published revision
they came from. Both are recorded, because they answer different questions.

Every function here follows the wizard probe convention: it returns rather than raises. The
consumer's fallback is `--unverified-development` plus a message, which is worse than a verified
build and much better than an install that dies on a diagnostic.
"""

from __future__ import annotations

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
    model: str
    revision: str | None
    artifact_digest: str
    path: Path


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


def _revision_from(path: Path) -> str | None:
    """The published revision, when the cache layout names one. `None` is not a failure."""
    try:
        if path.parent.name == _SNAPSHOT_PARENT and path.name:
            return path.name
    except Exception:
        return None
    return None


def artifact_identity_for(embedder: Any) -> ArtifactIdentity | None:
    """What `embedder` can prove about the weights it loaded, or `None`.

    `None` is a real answer, not an error: `hashing` has no weights at all, and `voyage` and
    `openai` keep theirs on somebody else's machine. The caller acts on it by building
    `--unverified-development` and saying so, rather than by failing.
    """
    try:
        from recall.embeddings import artifact_tree_sha256

        directory = _model_directory(embedder)
        if directory is None:
            return None

        # Raises on a directory with no files, and on a symlink escaping its root. Both mean "this
        # is not an artifact I can name", which is exactly `None`.
        digest = artifact_tree_sha256(directory)

        model = str(getattr(embedder, "name", "") or "")
        if not model:
            return None

        return ArtifactIdentity(
            # `fastembed` is the only provider that lands weights locally through
            # `resolve_embedder` today. Named explicitly rather than derived from the class, so a
            # renamed class cannot silently change a recorded lineage identity.
            provider="fastembed",
            model=model,
            revision=_revision_from(directory),
            artifact_digest=digest,
            path=directory,
        )
    except Exception as exc:  # noqa: BLE001 - a probe returns, it does not raise
        _log.warning("could not resolve an artifact identity (%s)", type(exc).__name__)
        return None
