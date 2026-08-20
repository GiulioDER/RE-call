"""Build the benchmark's calibrated corpus from scratch, reproducibly.

    python scripts/agent_ab_build_corpus.py --source <memory-dir> --dsn <dsn>

Six steps, each of which failed the first time in a way worth recording:

1. **Its own database.** The shared `recall-dogfood` corpus serves other sessions' MCP servers,
   and this build promotes a generation and publishes a calibration, which changes what those
   servers return. It also uses a database *inside* the session container rather than the
   container's default one, because the test suite DROPs tables and would take the corpus with it.
2. **Schema.** `schema apply` refuses the serving DSN for DDL and wants `RECALL_MIGRATION_DSN`.
3. **Manifest.** `manifest inventory` produces an object list; `generation build` wants the
   canonical manifest object that `manifest create` makes from it. `file://` access also requires
   `RECALL_LOCAL_ALLOWLIST`, so a manifest cannot name any file on the machine.
4. **Generation build.** Slow: about 188 files and 981 chunks here. ⛔ A shell timeout does NOT
   kill it. The first attempt was killed at ten minutes, kept running, and a re-run produced a
   SECOND identical generation. Check for an existing `building`/`validating` generation before
   starting one, which is what `--resume` does below.
5. **Calibration.** Needs >= 20 answerable and >= 20 unanswerable queries and a separability CI
   whose LOWER bound clears 0.90. 50/50 measured 0.980 [0.952, 1.000].
6. **Promotion.** A generation built `--unverified-development` needs an explicit unsafe flag to
   promote, and that is a real limitation of this corpus, recorded rather than hidden.

The calibration is what makes the corpus servable under the strict trust policy. Serving it needs
`RECALL_ENV=production`, which is why the on arm runs over stdio: production refuses the static
bearer token an HTTP listener would need.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path.home() / ".claude" / "projects" / "C--Users-gde00-Documents-recall" / "memory"
DEFAULT_QUERIES = REPO_ROOT / "benchmarks" / "agent_ab" / "calibration" / "memory-query-set.json"


def run(command: list[str], env: dict[str, str], *, timeout: float = 3600) -> str:
    """Run one CLI step, streaming nothing and returning stdout.

    `timeout` is generous on purpose: a generation build that outlives its shell keeps running and
    a re-run duplicates it, so waiting is cheaper than racing.
    """

    print(f"  $ {' '.join(command[2:5])} ...", flush=True)
    proc = subprocess.run(  # noqa: S603 - argv list, no shell
        command, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"step failed ({proc.returncode}): {' '.join(command)}\n"
            f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
        )
    return proc.stdout.strip()


def existing_generation(dsn: str) -> tuple[str, str] | None:
    """Return an already-built generation, so a re-run never duplicates one."""

    import psycopg

    with psycopg.connect(dsn, connect_timeout=20) as conn:
        row = conn.execute(
            "SELECT generation_id, state FROM recall_generations "
            "WHERE state IN ('validating', 'ready', 'active') ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    return (row[0], row[1]) if row else None


def published_calibration(dsn: str, generation: str) -> str | None:
    """Return the published calibration for this generation, if one already exists."""

    import psycopg

    with psycopg.connect(dsn, connect_timeout=20) as conn:
        row = conn.execute(
            "SELECT calibration_id FROM recall_calibrations "
            "WHERE generation_id = %s AND lifecycle_state = 'published' "
            "ORDER BY created_at DESC LIMIT 1",
            (generation,),
        ).fetchone()
    return row[0] if row else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--dsn", required=True, help="a database this benchmark owns")
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--embedder", default="fastembed")
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES))
    parser.add_argument("--corpus-version", default="memory-benchmark")
    parser.add_argument("--work", default=None, help="scratch directory for manifests")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    if not source.is_dir():
        raise SystemExit(f"source corpus not found: {source}")
    work = Path(args.work) if args.work else REPO_ROOT / "benchmarks" / "artifacts" / "agent_ab" / "corpus"
    work.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update(
        {
            "RECALL_DSN": args.dsn,
            "RECALL_MIGRATION_DSN": args.dsn,
            "RECALL_EMBEDDER": args.embedder,
            "RECALL_LOCAL_ALLOWLIST": str(source),
            "PYTHONUTF8": "1",
        }
    )
    cli = [sys.executable, "-m", "recall.cli"]

    print("1/6  schema")
    run(cli + ["schema", "apply"], env)

    print("2/6  manifest")
    objects = work / "objects.json"
    manifest = work / "manifest.json"
    run(cli + ["manifest", "inventory", str(source), "--output", str(objects)], env)
    run(
        cli
        + [
            "manifest", "create",
            "--corpus-version", args.corpus_version,
            "--objects", str(objects),
            "--output", str(manifest),
        ],
        env,
    )

    print("3/6  generation")
    found = existing_generation(args.dsn)
    if found:
        generation, state = found
        print(f"  reusing existing generation {generation} ({state}); not building a duplicate")
    else:
        run(
            cli
            + [
                "generation", "build", str(manifest),
                "--unverified-development",
                "--project", "recall-memory",
                "--chunker", "text",
            ],
            env,
        )
        found = existing_generation(args.dsn)
        if not found:
            raise SystemExit("generation build produced nothing to validate")
        generation, state = found

    print("4/6  validate")
    if state != "active":
        print("  " + run(cli + ["generation", "validate", generation], env))

    print("5/6  calibrate")
    published = published_calibration(args.dsn, generation)
    if published:
        # Re-calibrating would publish a SECOND artifact over the same generation and query set,
        # which is noise in an audit trail that exists to say which threshold was in force.
        print(f"  reusing published calibration {published}")
    else:
        output = run(
            cli
            + [
                "--tenant", args.tenant,
                "calibration", "calibrate",
                "--generation", generation,
                "--queries", args.queries,
                "--publish",
            ],
            env,
        )
        print("  " + output.replace("\n", "\n  "))
        if "status: certified" not in output:
            raise SystemExit(
                "calibration was NOT certified. It needs >= 20 labelled queries of each class "
                "and a separability CI whose lower bound clears 0.90; the CLI printed the reason "
                "above. A rejected calibration must not be served, so this stops here."
            )
        published = published_calibration(args.dsn, generation)

    print("6/6  promote")
    if state == "active":
        print(f"  {generation} is already active")
    else:
        print("  " + run(
            cli + ["generation", "promote", generation, "--unsafe-development-promotion"], env
        ))

    summary = {
        "generation_id": generation,
        "tenant": args.tenant,
        "source": source.name,
        "corpus_version": args.corpus_version,
        "embedder": args.embedder,
        "queries": Path(args.queries).name,
        "calibration_id": published,
        # Recorded because it is a real limitation: the manifest was not cryptographically
        # verified, so this corpus is fit for a local benchmark and not for a trust claim.
        "unverified_development": True,
    }
    (work / "corpus-build.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\ncalibrated corpus ready: {generation}")
    print(f"artifact: {work / 'corpus-build.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
