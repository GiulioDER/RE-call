"""Count PostgreSQL statements and time `trusted_search` on a generation store.

Pre-registered in `docs/preregistrations/2026-09-23-store-round-trips.md`.

``build`` creates, builds, validates and promotes one generation of ``--chunks`` single-chunk
objects under a fresh tenant, with a hashing embedder (distinct vectors per text; identical
vectors are pathological for HNSW). It prints the tenant id.

``search`` opens a `GenerationStore` on that tenant whose connection carries
``-c log_statement=all``, checks the instrument against a known answer, runs ``--searches``
trusted searches, and reports statements per search (from the database's own log, by backend
pid) and wall time per search. It measures whatever RE-call code is importable, so the same
script measures both arms: run it from each checkout.

    python scripts/measure_store_round_trips.py build --dsn "$RECALL_TEST_DSN" --chunks 2000
    python scripts/measure_store_round_trips.py search --dsn "$RECALL_TEST_DSN" \\
        --container recall-sess-xxxx --tenant <t> --searches 100

Needs a session database container you own: it reads that container's log with `docker logs`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import time
import uuid
from io import BytesIO
from urllib.parse import quote

BOUNDARY = "SELECT 'recall-measure-boundary'"
QUERIES = [f"how does component {i} handle erasure and calibration state {i % 7}" for i in range(50)]


class _S3:
    def __init__(self, objects: dict[tuple[str, str, str], bytes]) -> None:
        self.objects = objects

    def get_object(self, **kwargs):  # type: ignore[no-untyped-def]
        data = self.objects[(kwargs["Bucket"], kwargs["Key"], kwargs["VersionId"])]
        return {"Body": BytesIO(data), "ContentLength": len(data), "VersionId": kwargs["VersionId"]}


class _Embedder:
    """A hashing embedder named as the pipeline identity expects."""

    def __init__(self) -> None:
        from recall.embeddings import HashingEmbedder

        self._inner = HashingEmbedder(dim=64)

    dim = 64
    name = "measure-model"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed([text])[0]


def _pipeline():  # type: ignore[no-untyped-def]
    from recall.lineage import ChunkerIdentity, EmbedderIdentity, PipelineIdentity

    return PipelineIdentity(
        EmbedderIdentity("fixture", "measure-model", 64, revision="measure-model-commit"),
        ChunkerIdentity("paragraph-pack", 1, {"max_chars": 800, "overlap": 80}),
        fts_configuration={"language": "english", "schema_version": 1},
    )


def _build(dsn: str, chunks: int) -> str:
    from recall.generations import GenerationManager
    from recall.lineage import IndexManifestV1, ManifestObjectV1
    from recall.manifest import S3Allowlist, S3ObjectReader

    tenant = "measure-" + uuid.uuid4().hex[:10]
    objects: dict[tuple[str, str, str], bytes] = {}
    entries = []
    for index in range(chunks):
        data = (
            f"Memo {index}: component {index % 97} records erasure, calibration and "
            f"generation state number {index} for the round-trip measurement."
        ).encode()
        key = f"corpora/{tenant}/memo-{index:05d}.md"
        objects[("approved", key, "v1")] = data
        entries.append(
            ManifestObjectV1(
                f"s3://approved/{key}", "v1", "text/markdown", len(data),
                hashlib.sha256(data).hexdigest(),
            )
        )
    manifest = IndexManifestV1(tenant, "corpus-v1", tuple(entries))
    manager = GenerationManager(dsn, tenant, actor="measure", environment="test")
    generation = manager.create(manifest, _pipeline())
    reader = S3ObjectReader(_S3(objects), S3Allowlist.parse("approved/corpora/"))
    manager.build(generation.generation_id, reader, _Embedder(), lambda text: [text])
    manager.validate(generation.generation_id)
    manager.promote(generation.generation_id, unsafe_development=True)
    return tenant


def _log_lines(container: str, since: str) -> list[str]:
    completed = subprocess.run(
        ["docker", "logs", "--since", since, container],
        capture_output=True, text=True, check=True,
    )
    return (completed.stdout + completed.stderr).splitlines()


def _statements(lines: list[str], pid: int) -> list[str]:
    marker = re.compile(rf"\[{pid}\].*LOG:\s+(statement|execute [^:]*): (.*)")
    out = []
    for line in lines:
        match = marker.search(line)
        if match:
            out.append(match.group(2).strip())
    return out


def _search(dsn: str, container: str, tenant: str, searches: int) -> dict[str, object]:
    from recall.generation_store import GenerationStore
    from recall.trust import trusted_search
    from recall.trust_policy import TrustPolicy

    separator = "&" if "?" in dsn else "?"
    logged = f"{dsn}{separator}options={quote('-c log_statement=all')}"
    store = GenerationStore(logged, 64, tenant=tenant)
    embedder = _Embedder()
    policy = TrustPolicy.development()
    try:
        row = store._with_retry(lambda conn: conn.execute("SELECT pg_backend_pid()").fetchone())
        assert row is not None
        pid = int(row[0])
        since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 1))

        # Known answer first: five statements must log as exactly five.
        store._with_retry(lambda conn: conn.execute(BOUNDARY))
        for _ in range(5):
            store._with_retry(lambda conn: conn.execute("SELECT 1"))
        store._with_retry(lambda conn: conn.execute(BOUNDARY))

        timings: list[float] = []
        for index in range(searches):
            query = QUERIES[index % len(QUERIES)]
            started = time.perf_counter()
            trusted_search(store, embedder, query, k=5, policy=policy)
            timings.append((time.perf_counter() - started) * 1000.0)
            store._with_retry(lambda conn: conn.execute(BOUNDARY))
        time.sleep(1.0)  # let the container flush its log
        statements = _statements(_log_lines(container, since), pid)
    finally:
        store.close()

    groups: list[list[str]] = [[]]
    for statement in statements:
        if "recall-measure-boundary" in statement:
            groups.append([])
        else:
            groups[-1].append(statement)
    # groups[0]: before the first boundary; groups[1]: the known answer; then one per search.
    known = groups[1] if len(groups) > 1 else []
    if len(known) != 5 or any(s != "SELECT 1" for s in known):
        raise SystemExit(f"instrument failed its known-answer check: {known!r}")
    per_search = [len(group) for group in groups[2 : 2 + searches]]
    if len(per_search) != searches:
        raise SystemExit(f"expected {searches} search groups in the log, found {len(per_search)}")
    quartiles = statistics.quantiles(timings, n=4)
    import recall

    return {
        # Which checkout was measured: the arm is whatever `recall` resolved to.
        "recall_file": recall.__file__,
        "tenant": tenant,
        "searches": searches,
        "statements_first_search": per_search[0],
        "statements_steady_state": sorted(set(per_search[1:])),
        "statements_steady_state_median": statistics.median(per_search[1:]),
        "first_search_statements": groups[2],
        "steady_search_statements": groups[3] if len(groups) > 3 else [],
        "ms_median": statistics.median(timings),
        "ms_iqr": [quartiles[0], quartiles[2]],
        "ms_all": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    build = sub.add_parser("build")
    build.add_argument("--dsn", required=True)
    build.add_argument("--chunks", type=int, default=2000)
    search = sub.add_parser("search")
    search.add_argument("--dsn", required=True)
    search.add_argument("--container", required=True)
    search.add_argument("--tenant", required=True)
    search.add_argument("--searches", type=int, default=100)
    args = parser.parse_args()
    if args.mode == "build":
        print(_build(args.dsn, args.chunks))
        return
    print(json.dumps(_search(args.dsn, args.container, args.tenant, args.searches), indent=2))


if __name__ == "__main__":
    main()
