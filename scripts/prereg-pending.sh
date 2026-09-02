#!/usr/bin/env bash
# Print the pre-registrations that are genuinely unresolved, one path per line.
#
# ⛔ NOT "contains 'predicted, not yet measured'", which is what `session-close.sh` used to grep
# for. A result is APPENDED beneath the prediction and the prediction is never edited: that is the
# project's rule, and `scripts/check_citation_anchors.py` freezes these files to enforce it. So the
# original header survives in every completed record, forever. Measured 2026-08-29: that grep
# flagged 37 of 37 records when 7 were pending. A checklist that is 81% noise is worse than no
# checklist, because it teaches the reader to skim past the seven that matter.
#
# The rule that fits every convention actually in use: a record is RESOLVED once it carries a
# `Status:` line that is not the prediction's own. This deliberately does not enumerate the ways a
# result can be worded -- "measured", "partially confirmed", "measured (LoCoMo pending)" and
# "abandoned" all resolve, while "still predicted, not yet measured" stays pending -- because an
# enumeration is a list that goes stale silently the first time somebody phrases one differently.
#
# Exists as its own script so it can be tested. Inlined in `session-close.sh` the only way to test
# it was to restate the logic in the test, and two copies of one rule is the drift this repository
# keeps paying for.
#
# Usage: prereg-pending.sh [directory]   (default: docs/preregistrations relative to the repo root)
set -euo pipefail

dir="${1:-}"
if [ -z "$dir" ]; then
    dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/docs/preregistrations"
fi

[ -d "$dir" ] || exit 0

shopt -s nullglob
for prereg in "$dir"/*.md; do
    grep -q "Status:.*predicted, not yet measured" "$prereg" 2>/dev/null || continue
    # `grep -v` exits 0 when at least one line fails to match, which is exactly "there is a status
    # line that is not the prediction's own".
    if grep -h "Status:" "$prereg" 2>/dev/null | grep -qv "predicted, not yet measured"; then
        continue
    fi
    printf '%s\n' "$prereg"
done
