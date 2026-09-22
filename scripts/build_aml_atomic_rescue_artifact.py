"""Build a generation-bound atomic rescue artifact for one hosted AML tenant.

The hosted service binds an artifact to ``describe_corpus`` of the store it searches, under the
scope id it searches with (the tenant, or the specialist tenant on a Context route), and to the
calibration id and pipeline fingerprint in its own environment. This builder opens that store
through the service's own construction path, so the identities it writes are the ones the
service will ask for, and it refuses rather than guesses when anything disagrees.

Sessions are rebuilt from the stored raw windows alone: content-only windows are exact slices of
one word sequence, so every window must equal its recorded ``word_start:word_end`` slice and every
segment must be present. The atomizer then segments that sequence (``recall.atomizer``) and each
view is embedded with the scope's own embedder, behind the service's embedding lock.

Run it after the tenant's corpus is frozen: any later Add changes ``corpus_sha256`` and the
service will fail closed on this artifact, which is the intended behaviour.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recall.atomizer import ATOMIZER_STRATEGIES, window_views  # noqa: E402
from recall.types import Chunk  # noqa: E402


class BuildRefusal(RuntimeError):
    """The tenant cannot yield a trustworthy artifact; nothing was written."""


@dataclass(frozen=True)
class RebuiltSession:
    session: str
    text: str
    window_size: int
    window_stride: int
    chunk_ids: tuple[str, ...]


def _int(metadata: Mapping[str, object], name: str) -> int:
    value = metadata.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BuildRefusal(f"raw window lacks a non-negative integer {name}")
    return value


def rebuild_sessions(chunks: Iterable[Chunk]) -> list[RebuiltSession]:
    """Reassemble each session's word sequence from its raw content-only windows."""

    by_session: dict[str, dict[int, Chunk]] = defaultdict(dict)
    for chunk in chunks:
        metadata = chunk.metadata
        if metadata.get("record_type") != "raw":
            continue
        session = metadata.get("source_session_id")
        if not isinstance(session, str) or not session:
            raise BuildRefusal("raw window lacks source_session_id")
        segment = _int(metadata, "segment")
        previous = by_session[session].get(segment)
        if previous is not None and previous.text != chunk.text:
            raise BuildRefusal(f"session {session!r} has two different windows at segment {segment}")
        by_session[session][segment] = chunk

    rebuilt: list[RebuiltSession] = []
    for session in sorted(by_session):
        windows = by_session[session]
        ordered = [windows[index] for index in sorted(windows)]
        counts = {_int(chunk.metadata, "segment_count") for chunk in ordered}
        sizes = {_int(chunk.metadata, "word_window_size") for chunk in ordered}
        strides = {_int(chunk.metadata, "word_window_stride") for chunk in ordered}
        if len(counts) != 1 or len(sizes) != 1 or len(strides) != 1:
            raise BuildRefusal(f"session {session!r} mixes window geometries")
        count, size, stride = counts.pop(), sizes.pop(), strides.pop()
        if sorted(windows) != list(range(count)):
            raise BuildRefusal(f"session {session!r} is missing window segments")
        if not 0 < stride <= size:
            raise BuildRefusal(f"session {session!r} has an invalid window stride")
        words: list[str] = []
        for chunk in ordered:
            start = _int(chunk.metadata, "word_start")
            end = _int(chunk.metadata, "word_end")
            tokens = chunk.text.split()
            if end - start != len(tokens) or start > len(words):
                raise BuildRefusal(f"session {session!r} window offsets are inconsistent")
            overlap = words[start:]
            if tokens[: len(overlap)] != overlap:
                raise BuildRefusal(f"session {session!r} windows disagree on their overlap")
            words.extend(tokens[len(overlap) :])
        rebuilt.append(
            RebuiltSession(
                session,
                " ".join(words),
                size,
                stride,
                tuple(chunk.id for chunk in ordered),
            )
        )
    return rebuilt


def plan_views(
    sessions: Sequence[RebuiltSession], strategy: str
) -> tuple[list[str], list[dict[str, object]]]:
    """Return the view texts and the artifact metadata rows, in one deterministic order."""

    texts: list[str] = []
    rows: list[dict[str, object]] = []
    for rebuilt in sessions:
        for view in window_views(
            rebuilt.text,
            window_size=rebuilt.window_size,
            window_stride=rebuilt.window_stride,
            strategy=strategy,  # type: ignore[arg-type]
        ):
            texts.append(view.text)
            rows.append(
                {
                    "chunk_id": rebuilt.chunk_ids[view.parent_segment],
                    "source": rebuilt.session,
                    "parent_ordinal": view.parent_segment,
                    "view_ordinal": view.view_ordinal,
                }
            )
    if not rows:
        raise BuildRefusal("the tenant yields no atomic views")
    return texts, rows


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise BuildRefusal(f"{name} must be set exactly as the service unit sets it")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", required=True, help="the AML user id whose tenant to build")
    parser.add_argument("--atomizer", choices=ATOMIZER_STRATEGIES, required=True)
    parser.add_argument(
        "--specialist-profile",
        help="build the Context specialist scope instead of the primary Code4 scope",
    )
    parser.add_argument("--dry-run", action="store_true", help="report counts, embed nothing")
    args = parser.parse_args()

    import numpy as np

    from recall.atomic_rescue import (
        load_atomic_rescue_artifact,
        resolve_atomic_rescue_manifest,
        write_atomic_rescue_artifact,
    )
    from recall.embeddings import embed_passages, embedding_profile, embedding_profile_id
    from recall.store import PgVectorStore
    from recall_aml.__main__ import _resolve_hosted_embedders
    from recall_aml.config import HostedSettings
    from recall_aml.identity import specialist_tenant, tenant_for
    from recall_aml.storage import PgHostedRepository, describe_corpus
    from recall_aml.variants import variant

    settings = HostedSettings.from_env()
    behavior = variant(settings.variant_name)
    if not behavior.atomic_rescue:
        raise BuildRefusal(f"variant {behavior.name} does not carry atomic rescue")
    artifact_root = _required("RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT")
    calibration_id = _required("RECALL_AML_ATOMIC_RESCUE_CALIBRATION_ID")
    pipeline_fingerprint = _required("RECALL_AML_ATOMIC_RESCUE_PIPELINE_FINGERPRINT")

    embedder, specialists = _resolve_hosted_embedders(settings, behavior)
    base = PgVectorStore(
        settings.database_url,
        embedder.dim,
        table=settings.table,
        tenant="aml_service_readiness",
        generation_id=settings.generation_id,
    )
    repository = PgHostedRepository(base, embedder, None, specialist_embedders=specialists)
    tenant = tenant_for(args.user_id)
    if args.specialist_profile:
        if args.specialist_profile not in specialists:
            raise BuildRefusal(f"variant has no specialist {args.specialist_profile!r}")
        store = repository.specialist_store(tenant, args.specialist_profile)
        scope_id = specialist_tenant(tenant, args.specialist_profile)
        scope_embedder: Any = specialists[args.specialist_profile]
    else:
        store = repository.tenant_store(tenant)
        scope_id = tenant
        scope_embedder = embedder

    corpus = describe_corpus(store)
    generation_id = str(corpus["generation_id"])
    corpus_fingerprint = str(corpus["corpus_sha256"])
    chunks = list(store.iter_chunks(batch_size=256))
    sessions = rebuild_sessions(chunks)
    texts, rows = plan_views(sessions, args.atomizer)
    report: dict[str, object] = {
        "variant": behavior.name,
        "scope": "specialist" if args.specialist_profile else "primary",
        "generation_id": generation_id,
        "corpus_sha256": corpus_fingerprint,
        "raw_windows": sum(len(item.chunk_ids) for item in sessions),
        "sessions": len(sessions),
        "atomizer": args.atomizer,
        "views": len(rows),
        "parents": len({row["chunk_id"] for row in rows}),
    }
    manifest_path = resolve_atomic_rescue_manifest(
        artifact_root, generation_id, scope_id=scope_id, corpus_fingerprint=corpus_fingerprint
    )
    report["manifest"] = str(manifest_path)
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if manifest_path.parent.exists():
        raise BuildRefusal(f"an artifact already exists for this lineage: {manifest_path.parent}")

    vectors: list[list[float]] = []
    for offset in range(0, len(texts), 64):
        vectors.extend(embed_passages(scope_embedder, texts[offset : offset + 64]))
    matrix: Any = np.asarray(vectors, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    source_commit = os.environ.get("RECALL_SOURCE_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    written = write_atomic_rescue_artifact(
        manifest_path.parent,
        matrix=matrix,
        views=rows,
        generation_id=generation_id,
        calibration_id=calibration_id,
        pipeline_fingerprint=pipeline_fingerprint,
        corpus_fingerprint=corpus_fingerprint,
        embedding_profile=embedding_profile_id(scope_embedder),
        embedding_fingerprint=embedding_profile(scope_embedder).fingerprint(),
        ordinary_chunk_count=len(chunks),
        source_commit=source_commit,
    )
    artifact = load_atomic_rescue_artifact(written)
    artifact.assert_lineage(
        generation_id=generation_id,
        calibration_id=calibration_id,
        pipeline_fingerprint=pipeline_fingerprint,
        corpus_fingerprint=corpus_fingerprint,
        embedder=scope_embedder,
    )
    report["written"] = str(written)
    report["load_ms"] = round(artifact.load_ms, 3)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BuildRefusal as refusal:
        raise SystemExit(f"REFUSED: {refusal}") from None
