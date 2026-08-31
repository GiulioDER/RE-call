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
# What it therefore does NOT prove, named here and in the output rather than left implicit
# ------------------------------------------------------------------------------------------
# That the server STARTS. Every answer below is about rows in a database; none of it is about the
# process a client launches. The two come apart in exactly one direction and it is not rare: the
# serving checkout on VPS2 carries code whose migration level the database has not reached, the
# server raises `SchemaTooOld` on stderr at startup, and the client renders that as a server with
# NO tools, while every tenant here still reports CERTIFIED, because the corpus is fine.
#
# That is the same substitution this file was written to remove, one layer up: a cheap check
# standing in for an expensive one. So the summary prints the boundary and the command that
# crosses it instead of leaving a reader to infer that a certified corpus means a working
# session.
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
# The boundary this report cannot cross, printed before EVERY exit.
#
# Defined HERE, above every call site, because the first version of this fix defined it at the
# bottom of the file next to the summary and both early exits then died with
# `not_proved: command not found`. A bash function must be defined before it is called, and the
# fix for "printed always" was therefore printing nothing at all on the two paths it was
# written for. Found by scripts/session_corpus_tests.sh on its first run.
#
# It is a function rather than two lines at the end because the first draft put them at the end,
# commented them "printed always", and they were not: the two early exits returned first, and
# those are precisely the states a session most needs the pointer from. A comment asserting an
# invariant the code does not have is the same defect this file exists to remove, one level down.
not_proved() {
    printf 'NOT PROVED: that the server starts. This asked the corpus, not the process.\n'
    printf '  scripts/session-serving.sh verify   # ~30s, read-only, a real JSON-RPC handshake\n'
}

read -r -d '' SQL <<'EOSQL' || true
SELECT g.tenant_id
     || ' | gen ' || left(g.generation_id, 16)
     || ' | ' || coalesce(
            (SELECT CASE
                      WHEN c.certified AND c.corpus_fingerprint = g.corpus_fingerprint
                        -- Every interpolated value is coalesced. `separability`, `n_answerable`
                        -- and `n_unanswerable` are all NULLABLE, and in SQL one NULL inside a
                        -- `||` chain makes the WHOLE string NULL -- which the outer coalesce
                        -- would then report as "NO PUBLISHED CALIBRATION". A certified artifact
                        -- would be shown as a missing one, which is the wrong answer in the
                        -- alarming direction.
                        THEN 'CERTIFIED thr=' || round(c.threshold::numeric, 3)
                           || ' sep=' || coalesce(round(c.separability::numeric, 3)::text, '?')
                           || ' n=' || coalesce(c.n_answerable::text, '?')
                           || '/'   || coalesce(c.n_unanswerable::text, '?')
                      WHEN c.certified THEN 'STALE (fingerprint moved)'
                      ELSE 'NOT-CERTIFIED'
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
    not_proved
    exit 0
fi

if [ -z "$(printf '%s' "$out" | tr -d '[:space:]')" ]; then
    printf 'reachable, but NO ACTIVE GENERATION on any tenant.\n'
    printf 'strict trust refuses every query with INDEX_NOT_READY until one is promoted.\n'
    not_proved
    exit 0
fi

printf '%s\n' "$out"

# Named rather than left for the reader to spot. A `CERTIFIED` line is the only state in which a
# strict server answers `trust_state=trusted`; any other line is the difference between a corpus
# that answers and one that refuses.
#
# ⛔ Counted POSITIVELY, against `| CERTIFIED `, and not with `grep -v CERTIFIED`. The obvious
# spelling is wrong and wrong in the direction this whole script exists to remove: the failure
# state used to render as "UNCERTIFIED", which CONTAINS the substring "CERTIFIED", so an inverted
# match selected nothing and a genuinely uncertified tenant was summarised as "all certified".
# The status token is now `NOT-CERTIFIED` as well, so neither half of the check depends on the
# other being careful. A summary line that can only ever say "fine" is worse than no summary.
total="$(printf '%s\n' "$out" | grep -c '[^[:space:]]')"
certified="$(printf '%s\n' "$out" | grep -cF '| CERTIFIED ')"
if [ "$certified" -eq "$total" ]; then
    printf 'all %s active tenants certified: a strict server WOULD serve these as trusted.\n' "$total"
else
    printf '%s of %s active tenants certified. Strict refuses the rest, and the line above names why.\n' \
        "$certified" "$total"
fi

# The subjunctive above is doing real work: every line of this report is about rows in a database,
# and a session's actual question is whether its recall tools will answer. A serving checkout at
# the wrong migration level is a client with no tools at all, with all of the above still green.
not_proved
