#!/usr/bin/env bash
# Run this checkout's test suite on LINUX, from a Windows host, matching CI's `floor` job.
#
# Why this exists: two bugs shipped to master in one afternoon because a test was verified on
# Windows/3.14 and CI runs Linux/3.11. One was version-dependent (`Path.resolve()` stats path
# prefixes on 3.11 and 3.12 but not 3.14, so an `os.stat` stub that called `resolve()` recursed
# without bound). The other was PLATFORM-dependent (`local_path_for` refuses a UNC authority on
# POSIX through `_unc_supported`, so an unconditional `is not None` assertion passed here and
# failed there). A local 3.11 venv catches the first. Only Linux catches the second.
#
#   scripts/linux-check.sh tests/test_mcp_ingest_routing.py -q     # no database, DB tests skip
#   scripts/linux-check.sh --db tests/ -q                          # dedicated pgvector, full run
#
# ⛔ **`--db` starts its OWN pgvector container and never reuses the one `session-db.sh`
# started.** The suite DROPs tables, so two runs sharing one database drop each other's mid-flight
# and report failures that describe the other run's timing. That is the collision documented in
# CLAUDE.md, and it is easy to cause by accident here because a Linux run and a Windows run of the
# same checkout look independent and are not. This container publishes no port, so it cannot
# collide with a session container even by accident.
set -euo pipefail

IMAGE="python:3.11-slim"
VENV_VOLUME="recall-venv311-linux"
# (no user-defined network: the test container joins the DB container's namespace instead)
DB_NAME="recall-linux-db"
# `pwd -W` gives `C:/...` under Git Bash, which is what docker needs. Plain `pwd` gives `/c/...`,
# which docker reads as a RELATIVE path and silently appends to the cwd, producing a mount error
# naming a path that does not exist. Keep the two apart.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && { pwd -W 2>/dev/null || pwd; })"

WANT_DB=0
if [ "${1:-}" = "--down" ]; then
  # Only ever this script's OWN resources, named as constants above. Never a `session-db.sh`
  # container and never another checkout's: one of those belongs to somebody's running suite.
  docker rm -f "$DB_NAME" >/dev/null 2>&1 && echo "removed $DB_NAME" || echo "no $DB_NAME"
  echo "kept volume $VENV_VOLUME (delete with: docker volume rm $VENV_VOLUME)"
  exit 0
fi
if [ "${1:-}" = "--db" ]; then WANT_DB=1; shift; fi
[ $# -gt 0 ] || set -- tests/ -q

# The venv lives in a named volume, not in the repo: a Linux venv inside a Windows-mounted tree
# would collide with the host's own .venv, and building it into the mount writes root-owned
# artefacts onto the host filesystem.
if ! docker volume inspect "$VENV_VOLUME" >/dev/null 2>&1; then
  echo "linux-check: building $VENV_VOLUME (once, a few minutes)" >&2
  docker volume create "$VENV_VOLUME" >/dev/null
  # Dependency SPECS only, never `pip install -e .`: building the project would write *.egg-info
  # and build/ into the mounted Windows tree, which a concurrent host test run is reading.
  # The source itself arrives at runtime via PYTHONPATH, so it is always the live working tree.
  MSYS_NO_PATHCONV=1 docker run --rm -v "$REPO:/w:ro" -v "$VENV_VOLUME:/venv" "$IMAGE" bash -c '
    set -e
    pip install --quiet uv
    python - <<PY > /tmp/deps.txt
import tomllib, pathlib
p = tomllib.loads(pathlib.Path("/w/pyproject.toml").read_text(encoding="utf-8"))["project"]
specs = list(p.get("dependencies", []))
for extra in ("dev", "mcp", "documents"):
    specs += p["optional-dependencies"][extra]
print("\n".join(s for s in specs if not s.strip().lower().startswith("recall")))
PY
    uv venv /venv --python /usr/local/bin/python3.11 >/dev/null
    # `--resolution lowest-direct` is what CI floor uses: the oldest version of each DIRECT
    # dependency, transitives resolved normally. Plain `lowest` would test other projects
    # constraint hygiene rather than this one.
    uv pip install --python /venv/bin/python --resolution lowest-direct -r /tmp/deps.txt
  ' >&2
fi

DOCKER_ARGS=(--rm -v "$REPO:/w:ro" -v "$VENV_VOLUME:/venv" -w /w
             -e PYTHONPATH=/w -e PYTHONDONTWRITEBYTECODE=1)

if [ "$WANT_DB" = "1" ]; then
  if ! docker ps --format '{{.Names}}' | grep -qx "$DB_NAME"; then
    docker rm -f "$DB_NAME" >/dev/null 2>&1 || true
    echo "linux-check: starting $DB_NAME" >&2
    # No published port. The test container joins this container's network namespace, so nothing
    # needs to be exposed to the host — and nothing can collide with a `session-db.sh` container.
    docker run -d --name "$DB_NAME" \
      -e POSTGRES_USER=recall -e POSTGRES_PASSWORD=recall -e POSTGRES_DB=recall \
      pgvector/pgvector:pg16 >/dev/null
    # ⚠️ **The timeout FAILS, it does not fall through.** `cmd && break` is exempt from `set -e`
    # (verified: the loop survives a failing probe, which is what makes the wait work at all), so a
    # database that never comes up would leave this loop quietly and hand pytest a dead DSN. Every
    # DB test would then fail describing connection errors rather than the real cause, which is the
    # silent-failure shape this project keeps paying for. Say it, and show the container's own log.
    ready=0
    for _ in $(seq 1 60); do
      if docker exec "$DB_NAME" pg_isready -U recall >/dev/null 2>&1; then ready=1; break; fi
      sleep 1
    done
    if [ "$ready" != "1" ]; then
      echo "linux-check: $DB_NAME did not accept connections within 60s; its last log lines:" >&2
      docker logs --tail 20 "$DB_NAME" >&2 || true
      exit 1
    fi
  fi
  # ⛔ **The database is reached as `127.0.0.1`, and that is a correctness requirement, not a
  # style choice.** Sharing the DB container's network namespace makes its port genuinely local to
  # the test process, which is also exactly how CI reaches it (a service on localhost:5432).
  #
  # The obvious alternative — a user-defined network and `@recall-linux-db:5432` — trips TWO
  # independent non-local guards, and each would have to be switched off to proceed:
  #   1. `conftest.py::_reject_unsafe_test_dsn` refuses a non-local DSN, because the suite DROPs
  #      tables (`RECALL_ALLOW_REMOTE_TEST_DB`);
  #   2. `recall/store.py` refuses the published `recall:recall` credentials against a non-local
  #      host (`RECALL_ALLOW_INSECURE_DSN`).
  # Needing two security escapes to run a test suite is the design telling you it is wrong. Both
  # guards stay ARMED here, which also means this script keeps testing them.
  DOCKER_ARGS+=(--network "container:$DB_NAME"
                -e "RECALL_TEST_DSN=postgresql://recall:recall@127.0.0.1:5432/recall")
fi

# `-p no:cacheprovider`: the tree is mounted READ-ONLY, so pytest must not try to write .pytest_cache.
MSYS_NO_PATHCONV=1 exec docker run "${DOCKER_ARGS[@]}" "$IMAGE" \
  /venv/bin/python -m pytest -p no:cacheprovider "$@"
