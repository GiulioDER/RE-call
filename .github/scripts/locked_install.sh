#!/usr/bin/env bash
# Install the hashed, frozen uv.lock export for the given extras, then this checkout without its
# dependencies. Every CI job that installs from the lock calls this, so a change to the recipe
# lands in one place rather than in one copy per job.
#
# Usage: bash .github/scripts/locked_install.sh <extra> [<extra> ...]
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <extra> [<extra> ...]" >&2
  exit 2
fi

extras=()
for extra in "$@"; do
  extras+=(--extra "$extra")
done

requirements="${RUNNER_TEMP:?RUNNER_TEMP is unset}/locked-requirements.txt"
uv export --frozen "${extras[@]}" --no-emit-project \
  --format requirements-txt -o "$requirements"
pip install --require-hashes -r "$requirements"
pip install --no-deps -e .
