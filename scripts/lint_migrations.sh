#!/usr/bin/env bash
# Lint NEW schema migrations with squawk, the way recall's own runner will execute them.
#
# Why only new ones: every migration through FROZEN_THROUGH is already applied somewhere and pinned
# by recall/migrations/checksums.json, so a finding there can be learned from but never fixed. On
# 2026-10-01 those 25 files carried four findings in the two hazard classes this gate exists for:
# a constraint added without NOT VALID (0014, 0025), which scans the table under a write lock, and
# a non-concurrent CREATE INDEX on an existing table inside a transaction (0014, 0023), which blocks
# writes for the whole build. A new migration with either is refused.
#
# The runner's semantics, which squawk cannot see on its own:
#   * a file marked `-- recall:transactional` runs inside one transaction, so it is linted with
#     --assume-in-transaction; a `-- recall:concurrent-index` file runs outside one.
#   * the runner sets lock_timeout itself and sets statement_timeout = 0 on purpose (an HNSW build
#     can take hours), so squawk's two timeout rules are excluded in .squawk.toml.
#
# Needs `squawk` on PATH (pip install squawk-cli). Exits 0 when there is nothing new to lint.
set -euo pipefail

FROZEN_THROUGH=25
here="$(cd "$(dirname "$0")/.." && pwd)"
cd "$here"

transactional=()
concurrent=()
for file in recall/migrations/sql/*.sql; do
  number=$((10#$(basename "$file" | cut -d_ -f1)))
  [ "$number" -le "$FROZEN_THROUGH" ] && continue
  if head -n 1 "$file" | grep -q '^-- recall:transactional'; then
    transactional+=("$file")
  else
    concurrent+=("$file")
  fi
done

if [ ${#transactional[@]} -eq 0 ] && [ ${#concurrent[@]} -eq 0 ]; then
  echo "no migrations after $(printf '%04d' "$FROZEN_THROUGH"); nothing to lint"
  exit 0
fi

status=0
if [ ${#transactional[@]} -gt 0 ]; then
  squawk --config .squawk.toml --assume-in-transaction "${transactional[@]}" || status=1
fi
if [ ${#concurrent[@]} -gt 0 ]; then
  squawk --config .squawk.toml "${concurrent[@]}" || status=1
fi
exit "$status"
