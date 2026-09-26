#!/usr/bin/env bash
# C9 on VPS3 for the MM-1/MM-3 experiment: three Search arms over ONE ingested table. Never VPS2.
#   aml_mm_scope_vps3.sh setup <commit> | start | stop | status | teardown
#
# Pre-registration: docs/preregistrations/2026-09-25-aml-c9-multimodal-scope-and-dates.md
#
# Arm B (route, the served behaviour) on 18031 takes every Add; arms P (preserve) on 18032 and
# D (dual) on 18033 only Search the same table, so ingest, compile and embeddings are shared by
# construction. MM-3 is applied offline to stored responses, so every process serves undated
# image items. The Add-time compiler uses DeepSeek V4.1 Flash here (amendment 1: no gpt-4o-mini).
# Own clone, database, ports, lock, cache and API key; teardown removes all of them.
set -euo pipefail

readonly variant="C9_routed_specialists_grounded_graph_atomic"
readonly root="$HOME/mm1-mm3/serve"
readonly repo="$HOME/mm1-mm3/repo"
readonly db="mm_scope"
readonly table="recall_aml_mm_scope_chunks"
readonly generation="aml-mm-scope-v1"
readonly env_file="$root/base.env"
readonly compile_model="deepseek/deepseek-v4.1-flash"
declare -A ports=([B]=18031 [P]=18032 [D]=18033)
declare -A scopes=([B]=route [P]=preserve [D]=dual)

# The testbench is recognised by a hash of its hostname, so the public tree does not name the host.
readonly testbench_host_sha256="bf00783408a0e46d1e13f117840355e4e8acfecf3c549277ae44a454d7336c9e"
[[ "$(hostname | tr -d '[:space:]' | sha256sum | cut -c1-64)" == "$testbench_host_sha256" ]] || { echo "refusing: VPS3 only" >&2; exit 2; }

set -a
# shellcheck disable=SC1091
. "$HOME/recall-tb/db.env"
set +a
base_dsn="${RECALL_MIGRATION_DSN:-${RECALL_DSN:-}}"
dsn_for() {
    python3 - "$base_dsn" "$1" <<'EOF'
import sys
from urllib.parse import urlsplit, urlunsplit
parts = urlsplit(sys.argv[1])
print(urlunsplit(parts._replace(path="/" + sys.argv[2])))
EOF
}

pid_file() { echo "$root/serve-$1.pid"; }
alive() { local f; f="$(pid_file "$1")"; [[ -f "$f" ]] && kill -0 "$(cat "$f")" 2>/dev/null; }

case "${1:-}" in
    setup)
        commit="${2:?usage: setup <commit>}"
        mkdir -p "$root/.locks" "$root/.cache"
        git -C "$repo" fetch --quiet origin
        git -C "$repo" checkout --quiet --detach "$commit"
        commit="$(git -C "$repo" rev-parse HEAD)"
        if [[ ! -x "$repo/.venv/bin/python" ]]; then
            python3 -m venv "$repo/.venv"
            "$repo/.venv/bin/pip" install --quiet --upgrade pip
            "$repo/.venv/bin/pip" install --quiet -e "$repo[hosted]"
        fi
        dsn="$(dsn_for "$db")"
        psql "$base_dsn" -tAc "SELECT 1 FROM pg_database WHERE datname='$db'" | grep -q 1 \
            || psql "$base_dsn" -qc "CREATE DATABASE $db"
        psql "$dsn" -qc "CREATE EXTENSION IF NOT EXISTS vector"
        set -a
        # shellcheck disable=SC1091
        . "$HOME/recall-tb/.env"
        set +a
        api_key="$(openssl rand -hex 32)"
        umask 077
        {
            printf 'RECALL_AML_DATABASE_URL=%s\n' "$dsn"
            printf 'RECALL_AML_API_KEY=%s\n' "$api_key"
            printf 'RECALL_AML_AUTHORIZED_USER_ID=*\n'
            printf 'RECALL_AML_GIT_COMMIT=%s\n' "$commit"
            printf 'RECALL_AML_TABLE=%s\n' "$table"
            printf 'RECALL_AML_GENERATION=%s\n' "$generation"
            printf 'RECALL_AML_HOST=127.0.0.1\n'
            printf 'RECALL_AML_VARIANT=%s\n' "$variant"
            printf 'RECALL_AML_GENERATION_MODEL=%s\n' "$compile_model"
            printf 'RECALL_AML_DATED_MULTIMODAL=0\n'
            printf 'RECALL_AML_EMBED_LOCK_PATH=%s\n' "$root/.locks/embed.lock"
            printf 'RECALL_AML_EMBED_CACHE_PATH=%s\n' "$root/.cache/aml-hosted-embeddings.sqlite"
            printf 'RECALL_AML_ADD_CONCURRENCY=2\n'
            printf 'RECALL_AML_SEARCH_CONCURRENCY=2\n'
            printf 'VOYAGE_API_KEY=%s\n' "$(printf '%s' "${VOYAGE_API_KEY:-}" | tr -d '\r')"
            printf 'OPENROUTER_API_KEY=%s\n' "$(printf '%s' "${OPENROUTER_API_KEY:-}" | tr -d '\r')"
        } >"$env_file"
        # Global generation migrations go through the default `chunks` target first; the
        # custom table is refused until they have.
        "$repo/.venv/bin/recall" --serving-dsn "$dsn" --migration-dsn "$dsn" \
            --embedder "voyage:voyage-code-4" schema apply >/dev/null
        "$repo/.venv/bin/recall" --serving-dsn "$dsn" --migration-dsn "$dsn" \
            --embedder "voyage:voyage-code-4" --table "$table" schema apply >/dev/null
        echo "setup done: $commit $variant"
        ;;
    start)
        for arm in B P D; do
            if alive "$arm"; then echo "$arm already running"; continue; fi
            (
                set -a
                # shellcheck disable=SC1090
                . "$env_file"
                set +a
                export RECALL_AML_PORT="${ports[$arm]}" RECALL_AML_MULTIMODAL_SCOPE="${scopes[$arm]}"
                cd "$repo"
                setsid nohup "$repo/.venv/bin/python" -m recall_aml \
                    >>"$root/serve-$arm.log" 2>&1 </dev/null &
                echo $! >"$(pid_file "$arm")"
            )
        done
        for arm in B P D; do
            for _ in $(seq 1 180); do
                curl -fsS "http://127.0.0.1:${ports[$arm]}/health" >/dev/null 2>&1 && break
                sleep 1
            done
            printf '%s ' "$arm"
            curl -fsS "http://127.0.0.1:${ports[$arm]}/version" | python3 -c \
                "import json,sys;d=json.load(sys.stdin);print(d['git_commit'][:8],d['variant'],d['multimodal_scope'],d['search_content_profile'],d['generation_model'])"
        done
        ;;
    stop)
        for arm in B P D; do
            if alive "$arm"; then
                kill "$(cat "$(pid_file "$arm")")"
                for _ in $(seq 1 30); do alive "$arm" || break; sleep 1; done
            fi
            rm -f "$(pid_file "$arm")"
        done
        echo stopped
        ;;
    status)
        for arm in B P D; do
            if alive "$arm"; then echo "$arm running pid $(cat "$(pid_file "$arm")")"
            else echo "$arm not running"; fi
        done
        ;;
    key)
        grep -E '^RECALL_AML_API_KEY=' "$env_file" | cut -d= -f2-
        ;;
    teardown)
        "$0" stop
        psql "$base_dsn" -qc "DROP DATABASE IF EXISTS $db"
        rm -rf "$root"
        echo "torn down"
        ;;
    *) echo "usage: $0 setup <commit>|start|stop|status|key|teardown" >&2; exit 2 ;;
esac
