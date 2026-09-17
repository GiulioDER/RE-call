#!/usr/bin/env bash
set -euo pipefail

# Provision the isolated VPS2 service used by the preregistered AML experience compiler replay.
# This script intentionally reads provider credentials only on VPS2 and never prints them.

readonly app_root="${1:?usage: aml_experience_vps2_setup.sh APP_ROOT EXPECTED_COMMIT [VARIANT]}"
readonly expected_commit="${2:?expected git commit is required}"
readonly selected_variant="${3:-E0_raw}"
readonly recall_env="${RECALL_SOURCE_ENV:-/home/sentiment/recall-repos/.env}"
readonly amb_env="${AMB_SOURCE_ENV:-/home/sentiment/.amb.env}"
readonly runtime_dir="${HOME}/.config/recall-aml"
readonly runtime_env="${runtime_dir}/experience-compiler.env"
readonly unit_dir="${HOME}/.config/systemd/user"
readonly unit_path="${unit_dir}/recall-aml-experiment.service"
readonly table="recall_aml_experience_chunks"
readonly generation="aml-experience-v1"
readonly port="18004"

case "$selected_variant" in
    E0_raw|E1_compiled|E2_compiled_raw) ;;
    *) echo "unsupported experience variant" >&2; exit 2 ;;
esac

resolved_root="$(realpath -- "$app_root")"
case "$resolved_root" in
    /home/sentiment/recall-repos/aml-experience-compiler-*) ;;
    *) echo "app root is outside the dedicated experiment directory" >&2; exit 2 ;;
esac

actual_commit="$(git -C "$resolved_root" rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
    echo "app commit does not match the requested immutable revision" >&2
    exit 2
fi
if [[ ! -x "$resolved_root/.venv/bin/recall" || ! -x "$resolved_root/.venv/bin/recall-hosted" ]]; then
    echo "hosted virtual environment is incomplete" >&2
    exit 2
fi
if [[ ! -r "$recall_env" || ! -r "$amb_env" ]]; then
    echo "one or more server-side source environment files are unavailable" >&2
    exit 2
fi

# Capture each credential immediately after its owning environment file is loaded. This avoids
# an unrelated variable of the same name in the second file silently changing a database role.
set -a
# shellcheck disable=SC1090
. "$recall_env"
set +a
serving_dsn="${RECALL_DSN:-}"
migration_dsn="${RECALL_MIGRATION_DSN:-}"
voyage_key="${VOYAGE_API_KEY:-}"
set -a
# shellcheck disable=SC1090
. "$amb_env"
set +a
openrouter_key="${OPENROUTER_API_KEY:-}"

for value in "$serving_dsn" "$migration_dsn" "$voyage_key" "$openrouter_key"; do
    if [[ -z "$value" || "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
        echo "a required server-side setting is absent or malformed" >&2
        exit 2
    fi
done

mkdir -p -- "$runtime_dir" "$unit_dir"
chmod 700 -- "$runtime_dir"
api_key=""
if [[ -r "$runtime_env" ]]; then
    # shellcheck disable=SC1090
    . "$runtime_env"
    api_key="${RECALL_AML_API_KEY:-}"
fi
if [[ -z "$api_key" ]]; then
    api_key="$(openssl rand -hex 32)"
fi

env_tmp="$(mktemp "${runtime_dir}/experience-compiler.env.XXXXXX")"
unit_tmp="$(mktemp "${unit_dir}/recall-aml-experiment.service.XXXXXX")"
cleanup() {
    rm -f -- "$env_tmp" "$unit_tmp"
}
trap cleanup EXIT
chmod 600 -- "$env_tmp"
{
    printf 'RECALL_AML_DATABASE_URL=%s\n' "$serving_dsn"
    printf 'RECALL_AML_API_KEY=%s\n' "$api_key"
    printf 'RECALL_AML_GIT_COMMIT=%s\n' "$expected_commit"
    printf 'RECALL_AML_TABLE=%s\n' "$table"
    printf 'RECALL_AML_GENERATION=%s\n' "$generation"
    printf 'RECALL_AML_HOST=127.0.0.1\n'
    printf 'RECALL_AML_PORT=%s\n' "$port"
    printf 'RECALL_AML_VARIANT=%s\n' "$selected_variant"
    printf 'VOYAGE_API_KEY=%s\n' "$voyage_key"
    printf 'OPENROUTER_API_KEY=%s\n' "$openrouter_key"
} >"$env_tmp"
mv -f -- "$env_tmp" "$runtime_env"
chmod 600 -- "$runtime_env"

"$resolved_root/.venv/bin/recall" \
    --serving-dsn "$serving_dsn" \
    --migration-dsn "$migration_dsn" \
    --embedder voyage:voyage-4 \
    --table "$table" \
    schema apply

cat >"$unit_tmp" <<EOF
[Unit]
Description=RE-call AML experience compiler experiment
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$resolved_root
EnvironmentFile=$runtime_env
ExecStart=$resolved_root/.venv/bin/recall-hosted
Restart=on-failure
RestartSec=3
UMask=0077
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=default.target
EOF
mv -f -- "$unit_tmp" "$unit_path"

systemctl --user daemon-reload
systemctl --user enable recall-aml-experiment.service >/dev/null
systemctl --user restart recall-aml-experiment.service

version_json=""
for _ in $(seq 1 30); do
    if version_json="$(curl --fail --silent --show-error "http://127.0.0.1:${port}/version" 2>/dev/null)"; then
        break
    fi
    sleep 1
done
if [[ -z "$version_json" ]]; then
    systemctl --user status recall-aml-experiment.service --no-pager >&2 || true
    exit 1
fi
"$resolved_root/.venv/bin/python" -c \
    'import json,sys; d=json.loads(sys.argv[1]); assert d["git_commit"]==sys.argv[2]; assert d["variant"]==sys.argv[3]' \
    "$version_json" "$expected_commit" "$selected_variant"
curl --fail --silent --show-error "http://127.0.0.1:${port}/health" | \
    "$resolved_root/.venv/bin/python" -c \
    'import json,sys; d=json.load(sys.stdin); assert d["status"]=="ready"'

echo "recall-aml-experiment ready: commit=${expected_commit} variant=${selected_variant} port=${port}"
