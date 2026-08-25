#!/usr/bin/env bash
# Ask this project's own corpus what state it is actually in, and print the answer.
#
# Why this exists as a separate check rather than a line in session-open.sh
# -------------------------------------------------------------------------
# Until 2026-08-25 the session report said `.mcp.json present (2 servers)` and stopped there.
# That sentence was TRUE while both servers pointed at a `recall-dogfood` container on
# 127.0.0.1:5433 that was not running, on a host that no longer held the corpus, with
# `RECALL_TRUST_MODE=development` hardcoded to work around a calibration failure that had since
# been fixed. Four things were wrong at once and the report was green for all four, because it
# checked the existence of a FILE and was read as a statement about a CORPUS.
#
# The failure mode is worth naming because it is general: a cheap check standing in for an
# expensive one is not a weaker check, it is a misleading one. Nobody re-checks a green line.
#
# What this asks, and what each answer costs
# ------------------------------------------
# One ssh, one SQL query. It reports, per tenant, whether there is an active generation and
# whether a published CERTIFIED calibration is bound to it with a matching corpus fingerprint --
# which is precisely what `CalibrationRepository.resolve` decides `trusted` on. It does NOT run a
# search: that costs an embedder load and roughly 17 seconds per server, which is not a price a
# session opener may charge.
#
# It never exits non-zero for an unreachable corpus. A session editing code does not need it, and
# a setup script that refuses to open on a condition most sessions do not care about is a setup
# script people stop running.

set -uo pipefail

VPS2_HOST="${RECALL_VPS2_HOST:-vps2}"
VPS2_ENV_FILE="${RECALL_VPS2_ENV:-~/recall-repos/.env}"
# Bounded so an unreachable or slow host costs a session a few seconds, never a hang. BatchMode
# additionally refuses to sit on a passphrase prompt, which is the way an ssh check hangs forever
# without ever printing anything.
SSH_TIMEOUT="${RECALL_CORPUS_SSH_TIMEOUT:-20}"

case "${1:-status}" in
    status) ;;
    *)
        echo "usage: scripts/session-corpus.sh [status]" >&2
        exit 2
        ;;
esac

# The query mirrors `resolve`: an active generation, joined to a calibration that is published,
# certified, and bound to the SAME corpus fingerprint. A certified artifact whose fingerprint has
# moved is stale, and stale is a refusal, so the fingerprint comparison is not decoration.
read -r -d '' SQL <<'EOSQL' || true
SELECT g.tenant_id
     || ' | gen ' || left(g.generation_id, 16)
     || ' | ' || coalesce(
            (SELECT CASE
                      WHEN c.certified AND c.corpus_fingerprint = g.corpus_fingerprint
                        THEN 'CERTIFIED thr=' || round(c.threshold::numeric, 3)
                           || ' sep=' || round(c.separability::numeric, 3)
                           || ' n=' || c.n_answerable || '/' || c.n_unanswerable
                      WHEN c.certified THEN 'STALE (fingerprint moved)'
                      ELSE 'UNCERTIFIED'
                    END
             FROM recall_calibrations c
             WHERE c.tenant_id = g.tenant_id
               AND c.generation_id = g.generation_id
               AND c.lifecycle_state = 'published'
             ORDER BY c.created_at DESC
             LIMIT 1),
            'NO PUBLISHED CALIBRATION -> strict refuses')
FROM recall_generations g
WHERE g.state = 'active'
ORDER BY g.tenant_id;
EOSQL

out="$(timeout "$SSH_TIMEOUT" ssh -o BatchMode=yes -o ConnectTimeout=10 "$VPS2_HOST" \
        "set -a; . $VPS2_ENV_FILE 2>/dev/null; set +a; \
         psql \"\$RECALL_DSN\" -t -A -P pager=off -c $(printf '%q' "$SQL")" 2>&1)"
rc=$?

if [ $rc -ne 0 ]; then
    printf 'UNREACHABLE via ssh %s (exit %s)\n' "$VPS2_HOST" "$rc"
    printf 'the recall MCP servers will have no tools this session.\n'
    # The first line of the error, not all of it: a full ssh/psql trace buries the one useful
    # sentence under stack noise in a report meant to be read at a glance.
    printf 'first line: %s\n' "$(printf '%s' "$out" | head -1)"
    printf 'this is not fatal. It only matters if you meant to query the corpus.\n'
    exit 0
fi

if [ -z "$(printf '%s' "$out" | tr -d '[:space:]')" ]; then
    printf 'reachable, but NO ACTIVE GENERATION on any tenant.\n'
    printf 'strict trust refuses every query with INDEX_NOT_READY until one is promoted.\n'
    exit 0
fi

printf '%s\n' "$out"

# Named rather than left for the reader to spot. "CERTIFIED" on every line is the only state in
# which a strict server answers `trust_state=trusted`, and any other line is the difference
# between a corpus that answers and one that refuses.
if printf '%s' "$out" | grep -qv 'CERTIFIED'; then
    printf 'at least one tenant is not certified: strict refuses it, and the failure code names why.\n'
else
    printf 'all active tenants certified -- a strict server serves these as trusted.\n'
fi
