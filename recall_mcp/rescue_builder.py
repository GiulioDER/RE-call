"""The host MCP server builds a missing atomic rescue artifact in the background.

On a desktop install (the Docker stack) the app's uploads and recalibrations run in containers
that cannot reach the host artifact root, so they promote a generation, or publish a calibration,
with no rescue artifact. The host servers run with RECALL_ATOMIC_RESCUE_WHEN_MISSING=skip, so a
search then answers without the rescue (`recall.trust`); this builds what is missing so the next
searches have it again (docs/DESKTOP_ATOMIC_RESCUE_DESIGN.md).

After each retrieval, at most once per CHECK_INTERVAL_S, it asks whether an artifact serves the
active generation beside its published calibration, and when none does starts one daemon thread
that builds it (`recall.generation_rescue.build_artifact`, the build `recall generation rescue`
runs) and then removes superseded artifacts. A lock file under the root, taken without waiting,
lets one process build at a time: several servers of one tenant (Claude Code, Codex, each open
session) see the same gap, and the others leave it to the holder. A failed build is retried after
a backoff, MIN_BACKOFF_S doubling to MAX_BACKOFF_S, so a broken build does not run on every search.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from recall.generation_rescue import artifact_manifest, build_artifact

if TYPE_CHECKING:
    from recall.embeddings import Embedder
    from recall.generation_store import GenerationStore

__all__ = ["RescueBuilder"]

_log = logging.getLogger(__name__)

CHECK_INTERVAL_S = 30.0
MIN_BACKOFF_S = 60.0
MAX_BACKOFF_S = 600.0


def _artifact_serves(root: Path, binding: Mapping[str, str], calibration_id: str, embedder: Embedder) -> bool:
    """Whether an artifact under `root` serves this generation beside `calibration_id`, verified by
    the serving loader (its cache makes a repeated check cheap)."""
    from recall.generation_rescue import _serves

    lineage = {
        "generation_id": binding.get("generation_id"),
        "calibration_id": calibration_id,
        "pipeline_fingerprint": binding.get("pipeline_fingerprint"),
        "corpus_fingerprint": binding.get("corpus_fingerprint"),
    }
    return _serves(artifact_manifest(root, str(binding["generation_id"]), calibration_id), lineage, embedder)


class RescueBuilder:
    """Builds the atomic rescue artifact one host server's searches are missing."""

    def __init__(
        self,
        store: GenerationStore,
        embedder: Embedder,
        root: Path,
        *,
        build: Callable[..., str] = build_artifact,
        prune: Callable[[str, str, Path], int] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store, self._embedder, self._root = store, embedder, root
        self._build, self._prune, self._clock = build, prune, clock
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_check = -math.inf
        self._retry_at: dict[tuple[str, str], tuple[float, float]] = {}  # key -> (retry at, backoff)
        self._stopped = False

    @classmethod
    def from_env(
        cls, store: object, embedder: Embedder, env: Mapping[str, str], **kwargs: object
    ) -> RescueBuilder | None:
        """A builder for a PostgreSQL store whose rescue is active with `skip` and a root; else None.

        The settings are the tenant's, resolved as its searches resolve them, so the builder and the
        search agree. A Lite store builds its artifact after each write instead (`recall.lite.atomic`).
        """
        from recall._env import tenant_retrieval_environment
        from recall.generation_store import GenerationStore

        if not isinstance(store, GenerationStore):
            return None
        values = tenant_retrieval_environment(env, store.tenant)
        root = values.get("RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT", "").strip()
        if (
            values.get("RECALL_ATOMIC_RESCUE_MODE", "off").strip().lower() != "active"
            or values.get("RECALL_ATOMIC_RESCUE_WHEN_MISSING", "refuse").strip().lower() != "skip"
            or not root
        ):
            return None
        return cls(store, embedder, Path(root).expanduser(), **kwargs)  # type: ignore[arg-type]

    def after_retrieval(self) -> None:
        """Start a build when the active generation has no artifact. Never raises; returns at once."""
        try:
            with self._lock:
                now = self._clock()
                if self._stopped or now - self._last_check < CHECK_INTERVAL_S:
                    return
                self._last_check = now
                if self._thread is not None and self._thread.is_alive():
                    return
            resolution = self._store.resolve_calibration()
            if resolution.artifact is None:
                return  # strict search refuses an uncalibrated generation before the rescue
            binding = self._store.generation_binding()
            key = (str(binding["generation_id"]), str(resolution.artifact.calibration_id))
            with self._lock:
                retry_at, _backoff = self._retry_at.get(key, (-math.inf, 0.0))
                if now < retry_at:
                    return
            if _artifact_serves(self._root, binding, key[1], self._embedder):
                return
            with self._lock:
                if self._stopped:
                    return
                self._thread = threading.Thread(
                    target=self._run, args=(key,), name="recall-rescue-build", daemon=True
                )
                self._thread.start()
        except Exception as exc:  # BROAD-CATCH: fail-open  # a rescue problem must never fail a search
            _log.warning("atomic rescue build check failed: %s", exc)

    def stop(self, timeout_s: float = 10.0) -> None:
        """Start no more builds, and let a running one finish before the store closes."""
        with self._lock:
            self._stopped = True
            thread = self._thread
        if thread is not None:
            thread.join(timeout_s)

    def _run(self, key: tuple[str, str]) -> None:
        from recall.lite.writer_lock import _try_lock, _unlock

        generation_id = key[0]
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            with open(self._root / f".rescue-{generation_id}.lock", "ab") as fh:
                if not _try_lock(fh):
                    return  # another process builds it; its artifact serves this server too
                try:
                    report = self._build(
                        self._store._dsn, self._store.tenant, generation_id, self._embedder, self._root
                    )
                    _log.info("atomic rescue for %s: %s", generation_id, report)
                    if self._prune is not None:
                        self._prune(self._store._dsn, self._store.tenant, self._root)
                finally:
                    _unlock(fh)
        except Exception as exc:  # BROAD-CATCH: fail-open  # logged and retried after the backoff
            _log.warning("atomic rescue build for %s failed: %s", generation_id, exc)
            with self._lock:
                _retry, backoff = self._retry_at.get(key, (-math.inf, 0.0))
                backoff = MIN_BACKOFF_S if backoff == 0.0 else min(backoff * 2, MAX_BACKOFF_S)
                self._retry_at[key] = (self._clock() + backoff, backoff)
            return
        with self._lock:
            self._retry_at.pop(key, None)
