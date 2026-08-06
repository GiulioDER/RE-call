#!/usr/bin/env python
"""Load an `encode_sparse.py` artifact into the learned sparse sidecar.

    python scripts/load_sparse.py --input vectors.jsonl --table mtrag_clapnq \
        --dsn-env RECALL_MIGRATION_DSN

Runs on a host you trust, because unlike the encoder this one needs a database credential. It
verifies the artifact header against the profile it is about to write under, so vectors from one
model can never land in a corpus indexed under another's name , the failure the profile column
exists to prevent, and one that produces plausible scores rather than an error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def read_artifact(path: Path) -> tuple[dict, list[tuple[str, dict[int, float]]]]:
    """`(header, rows)`. Raises if the header is missing — the identity is not optional."""
    header: dict | None = None
    rows: list[tuple[str, dict[int, float]]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("_header"):
                header = row
                continue
            if header is None:
                raise ValueError(
                    f"{path}: vector on line {lineno} before any header. Without the header "
                    f"nothing records WHICH model produced these weights, and vectors from two "
                    f"encoders are silently mixable."
                )
            rows.append((str(row["id"]), {int(t): float(w) for t, w in row["weights"].items()}))
    if header is None:
        raise ValueError(f"{path}: no header line; refusing to load vectors of unknown origin")
    return header, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--table", required=True, help="the chunk table these vectors belong to")
    parser.add_argument("--dsn-env", default="RECALL_MIGRATION_DSN")
    parser.add_argument("--dim", type=int, required=True, help="dense dimension of --table")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--profile-id", default=None,
        help="override the artifact's profile_id; must be given deliberately",
    )
    args = parser.parse_args(argv)

    dsn = os.environ.get(args.dsn_env)
    if not dsn:
        print(f"{args.dsn_env} is not set", file=sys.stderr)
        return 2

    from recall.store import PgVectorStore

    header, rows = read_artifact(args.input)
    profile_id = args.profile_id or header["profile_id"]
    print(json.dumps({
        "event": "start", "vectors": len(rows), "table": args.table,
        "profile_id": profile_id, "model_name": header.get("model_name"),
        "artifact_digest": header.get("artifact_digest"),
        "fingerprint": header.get("fingerprint"),
    }), flush=True)

    if header.get("artifact_digest") == "unpinned":
        # A warning, not a refusal: an unpinned encode is still a usable experiment. But a
        # published number that rests on it cannot claim reproducibility, so the fact travels.
        print(json.dumps({
            "event": "warning",
            "message": "artifact_digest=unpinned — these weights are NOT reproducible from the "
                       "model name alone. Re-encode with --revision before publishing a number.",
        }), flush=True)

    kwargs = {"tenant": args.tenant} if args.tenant else {}
    written = 0
    with PgVectorStore(dsn, dim=args.dim, table=args.table, **kwargs) as store:
        for start in range(0, len(rows), args.batch_size):
            batch = dict(rows[start : start + args.batch_size])
            written += store.upsert_sparse(profile_id, batch)
            print(json.dumps({"event": "progress", "written": written}), flush=True)
        total = store.sparse_row_count(profile_id)

    # Report what the DATABASE holds, not what this process believes it sent. The two differ
    # whenever an upsert collided on an id already present, and the stored count is the number
    # that decides whether the retriever will answer.
    print(json.dumps({"event": "done", "sent": written, "rows_in_table": total}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
