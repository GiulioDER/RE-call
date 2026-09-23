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
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recall.atomizer import ATOMIZER_STRATEGIES  # noqa: E402
# The reconstruction lives beside the Add-time builder so the two cannot drift; re-exported here
# for this script's callers and tests.
from recall_aml.atomic_views import (  # noqa: E402,F401
    BuildRefusal,
    RebuiltSession,
    plan_views,
    rebuild_sessions,
)


def served_corpus_fingerprint(
    base: Mapping[str, object], graph: Mapping[str, object] | None
) -> str:
    """The corpus identity the hosted service serves, and so binds an artifact to.

    Mirrors ``HostedService._corpus_status``: with a graph sidecar the served
    ``corpus_sha256`` is the digest of the raw corpus digest and the scope's graph corpus
    digest, not the raw store's own digest. Measured 2026-09-23 on the VPS3 C8 instance: the
    store said ``f5aa3cdf…`` while every search reported ``40df3177…``, so an artifact bound to
    the store's digest would have failed closed on every query.
    """

    from recall_aml.identity import canonical_digest

    if graph is None:
        return str(base["corpus_sha256"])
    return canonical_digest([base["raw_corpus_sha256"], graph["corpus_sha256"]])


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
    parser.add_argument(
        "--expect-served-corpus-sha256",
        help="refuse unless the derived identity equals the X-Recall-Corpus-SHA256 the service serves",
    )
    args = parser.parse_args()

    import numpy as np

    from recall.atomic_rescue import (
        load_atomic_rescue_artifact,
        resolve_atomic_rescue_manifest,
        write_atomic_rescue_artifact,
    )
    from recall.embeddings import embed_passages, embedding_profile, embedding_profile_id
    from recall.pool import SharedPool
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
    # The service reads tenants through views on one shared pool; a store without one refuses
    # for_tenant(), because a per-connection tenant could read as the wrong tenant.
    pool = SharedPool(settings.database_url, min_size=1, max_size=2, statement_timeout_ms=25_000)
    base = PgVectorStore(
        settings.database_url,
        embedder.dim,
        table=settings.table,
        tenant="aml_service_readiness",
        generation_id=settings.generation_id,
        shared_pool=pool,
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
    graph = describe_corpus(repository.graph_store(scope_id)) if behavior.graph_sidecar else None
    corpus_fingerprint = served_corpus_fingerprint(corpus, graph)
    expected = args.expect_served_corpus_sha256
    if expected is not None and expected != corpus_fingerprint:
        raise BuildRefusal(
            f"derived served corpus {corpus_fingerprint} differs from the served {expected}"
        )
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
