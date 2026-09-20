#!/usr/bin/env bash
set -euo pipefail

# Run one frozen MemEye arm through a localhost-only hosted service on VPS2.
# Provider credentials are read on VPS2, written only to a mode-0600 EnvironmentFile, and removed
# after the arm. The result directory and dataset cache must already exist.

readonly arm="${1:?usage: aml_multimodal_memeye_vps2.sh ARM APP_ROOT COMMIT RESULT_DIR}"
readonly app_root="${2:?app root is required}"
readonly expected_commit="${3:?commit is required}"
readonly result_dir="${4:?result directory is required}"
readonly recall_env="${RECALL_SOURCE_ENV:-/home/sentiment/recall-repos/.env}"
readonly amb_env="${AMB_SOURCE_ENV:-/home/sentiment/.amb.env}"
readonly runtime_dir="${HOME}/.config/recall-aml"
readonly unit_dir="${HOME}/.config/systemd/user"
readonly unit_name="recall-aml-memeye.service"
readonly unit_path="${unit_dir}/${unit_name}"
readonly runtime_env="${runtime_dir}/memeye.env"
readonly port="18110"
readonly api_key="memeye-local-c898"

case "$arm" in
    MM0_caption) readonly arm_slug="mm0" ;;
    MM1_preserve) readonly arm_slug="mm1" ;;
    MM2_dual) readonly arm_slug="mm2" ;;
    *) echo "unsupported MemEye arm" >&2; exit 2 ;;
esac

resolved_root="$(realpath -- "$app_root")"
resolved_result="$(realpath -- "$result_dir")"
case "$resolved_root" in
    /home/sentiment/recall-repos/aml-memeye-*) ;;
    *) echo "app root is outside the dedicated MemEye checkout" >&2; exit 2 ;;
esac
case "$resolved_result" in
    "$resolved_root"/results/aml-multimodal-memeye-brand-v1/*) ;;
    *) echo "result path is outside the dedicated experiment root" >&2; exit 2 ;;
esac
if [[ "$(git -C "$resolved_root" rev-parse HEAD)" != "$expected_commit" ]]; then
    echo "app commit does not match the frozen revision" >&2
    exit 2
fi
if [[ ! -x "$resolved_root/.venv/bin/python" || ! -r "$resolved_result/cache/identity.json" ]]; then
    echo "runtime or materialized dataset is incomplete" >&2
    exit 2
fi
if [[ ! -r "$recall_env" || ! -r "$amb_env" ]]; then
    echo "server-side credential source is unavailable" >&2
    exit 2
fi

set -a
# shellcheck disable=SC1090
. "$recall_env"
set +a
serving_dsn="${RECALL_DSN:-}"
voyage_key="${VOYAGE_API_KEY:-}"
set -a
# shellcheck disable=SC1090
. "$amb_env"
set +a
openrouter_key="${OPENROUTER_API_KEY:-}"
for value in "$serving_dsn" "$voyage_key" "$openrouter_key"; do
    if [[ -z "$value" || "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
        echo "a required server-side setting is absent or malformed" >&2
        exit 2
    fi
done

commit_slug="${expected_commit:0:8}"
table="recall_aml_memeye_${arm_slug}_${commit_slug}"
generation="aml-memeye-${arm_slug}-${commit_slug}"
mkdir -p -- "$runtime_dir" "$unit_dir"
chmod 700 -- "$runtime_dir"
env_tmp="$(mktemp "${runtime_dir}/memeye.env.XXXXXX")"
unit_tmp="$(mktemp "${unit_dir}/recall-aml-memeye.service.XXXXXX")"
cleanup() {
    systemctl --user stop "$unit_name" >/dev/null 2>&1 || true
    rm -f -- "$runtime_env" "$env_tmp" "$unit_tmp"
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
    printf 'RECALL_AML_VARIANT=%s\n' "$arm"
    printf 'RECALL_AML_ADD_CONCURRENCY=1\n'
    printf 'RECALL_AML_SEARCH_CONCURRENCY=1\n'
    printf 'VOYAGE_API_KEY=%s\n' "$voyage_key"
} >"$env_tmp"
mv -f -- "$env_tmp" "$runtime_env"
chmod 600 -- "$runtime_env"

printf '%s\n' \
    '[Unit]' \
    "Description=RE-call MemEye ${arm} experiment" \
    'After=network-online.target' \
    'Wants=network-online.target' \
    '' \
    '[Service]' \
    'Type=simple' \
    "WorkingDirectory=${resolved_root}" \
    "EnvironmentFile=${runtime_env}" \
    "ExecStart=${resolved_root}/.venv/bin/python -m recall_aml" \
    'Restart=on-failure' \
    'RestartSec=3' \
    'UMask=0077' \
    'NoNewPrivileges=true' \
    'PrivateTmp=true' >"$unit_tmp"
mv -f -- "$unit_tmp" "$unit_path"
systemctl --user daemon-reload
systemctl --user restart "$unit_name"

version_json=""
for _ in $(seq 1 180); do
    if version_json="$(curl --fail --silent --show-error "http://127.0.0.1:${port}/version" 2>/dev/null)"; then
        break
    fi
    sleep 1
done
if [[ -z "$version_json" ]]; then
    systemctl --user status "$unit_name" --no-pager >&2 || true
    exit 1
fi
"$resolved_root/.venv/bin/python" -c \
    'import json,sys; d=json.loads(sys.argv[1]); assert d["git_commit"]==sys.argv[2]; assert d["variant"]==sys.argv[3]' \
    "$version_json" "$expected_commit" "$arm"
curl --fail --silent --show-error "http://127.0.0.1:${port}/health" | \
    "$resolved_root/.venv/bin/python" -c \
    'import json,sys; assert json.load(sys.stdin)["status"]=="ready"'

export RECALL_AML_TOKEN="$api_key"
export OPENROUTER_API_KEY="$openrouter_key"
"$resolved_root/.venv/bin/python" "$resolved_root/scripts/aml_multimodal_memeye.py" run-arm \
    --arm "$arm" \
    --base-url "http://127.0.0.1:${port}" \
    --cache-dir "$resolved_result/cache" \
    --output "$resolved_result/${arm}.json" \
    --run-id "${commit_slug}-live1" \
    --spend-ledger "$resolved_result/spend-ledger.json"
