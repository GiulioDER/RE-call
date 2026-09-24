#!/usr/bin/env bash
# Run the C9 Add speed measurement on the VPS3 testbench. Run as root on VPS3.
#
# Pre-registration: docs/preregistrations/2026-09-24-aml-c9-add-speed.md
#
#   bash scripts/aml_add_speed_vps3.sh <run-id>
#
# Measurement 1: four arms, one server per arm, at RECALL_AML_ADD_CONCURRENCY 8, 16, 12, 8.
# Measurement 2: the compiler through three OpenRouter routes. Writes everything to
# /home/sentiment/c9-speed/<run-id>/. Touches nothing on VPS2 and no official service.
set -euo pipefail

RUN=${1:?run id}
U=sentiment
B=/home/$U/locomo-route
CODE=$B/code
PY=$B/.venv/bin/python
OUT=/home/$U/c9-speed/$RUN
UNIT=recall-aml-c9-speed
PORT=18121
URL=http://127.0.0.1:$PORT
KEY_FILE=$B/c9su.key

install -d -o "$U" -g "$U" "$OUT"
git -C "$CODE" log -1 --format='%H %s' | tee "$OUT/commit.txt"

cleanup() {
  systemctl stop "$UNIT" 2>/dev/null || true
  systemctl reset-failed "$UNIT" 2>/dev/null || true
  rm -f "$OUT"/server-*.env
}
trap cleanup EXIT

start_server() {
  local k=$1 label=$2
  systemctl stop "$UNIT" 2>/dev/null || true
  systemctl reset-failed "$UNIT" 2>/dev/null || true
  local env_file=$OUT/server-$label.env
  (
    set -a; . /home/$U/recall-tb/db.env; . /home/$U/recall-tb/.env; set +a
    cat <<EOF
RECALL_AML_DATABASE_URL=${RECALL_DSN%/recall}/locomo_route_20260923
RECALL_AML_API_KEY=$(cat "$KEY_FILE")
RECALL_AML_GIT_COMMIT=$(git -C "$CODE" rev-parse HEAD)
RECALL_AML_TABLE=recall_aml_locomo_route_chunks
RECALL_AML_GENERATION=locomo-route-20260923
RECALL_AML_VARIANT=C9_routed_specialists_grounded_graph_atomic
RECALL_AML_EMBED_LOCK_PATH=$OUT/embed.lock
RECALL_AML_EMBED_CACHE_PATH=$OUT/embeddings.sqlite
RECALL_AML_AUTHORIZED_USER_ID=*
RECALL_AML_HOST=127.0.0.1
RECALL_AML_PORT=$PORT
RECALL_AML_ADD_CONCURRENCY=$k
RECALL_AML_SEARCH_CONCURRENCY=3
VOYAGE_API_KEY=$VOYAGE_API_KEY
OPENROUTER_API_KEY=$OPENROUTER_API_KEY
EOF
  ) > "$env_file"
  chmod 600 "$env_file"; chown "$U:$U" "$env_file"
  date +%s > "$OUT/start-$label.epoch"
  systemd-run --unit="$UNIT" --uid="$U" --gid="$U" \
    --property=EnvironmentFile="$env_file" \
    --property=MemoryMax=4G --property=MemorySwapMax=0 --property=CPUAccounting=yes \
    --working-directory="$CODE" "$PY" -m recall_aml
  for _ in $(seq 1 90); do curl -fsS "$URL/health" >/dev/null 2>&1 && break; sleep 2; done
  local pid
  pid=$(systemctl show "$UNIT" -p MainPID --value)
  # Apparatus check: the process must really run with the concurrency this arm claims.
  local seen
  seen=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^RECALL_AML_ADD_CONCURRENCY=//p')
  [ "$seen" = "$k" ] || { echo "server runs Add concurrency '$seen', arm wants $k" >&2; exit 3; }
  echo "{\"arm\":\"$label\",\"pid\":$pid,\"add_concurrency_in_process\":$seen}" > "$OUT/server-$label.json"
}

stop_server() {
  local label=$1
  systemctl show "$UNIT" -p MemoryPeak -p NRestarts -p CPUUsageNSec > "$OUT/server-$label.systemd.txt"
  # Per-Add latency_ms and the compile completion times, to split compile from the rest.
  journalctl -u "$UNIT" --no-pager -o json --since "@$(cat "$OUT/start-$label.epoch")" \
    > "$OUT/journal-$label.json"
  systemctl stop "$UNIT"
  systemctl reset-failed "$UNIT" 2>/dev/null || true
}

run_arm() {
  local arm=$1 label=$2 k=$3
  start_server "$k" "$label"
  runuser -u "$U" -- env RECALL_AML_API_KEY="$(cat "$KEY_FILE")" \
    "$PY" "$CODE/scripts/aml_add_throughput.py" --base-url "$URL" --data "$B/locomo10.json" \
    --arm "$arm" --arm-label "$label" --server-add-concurrency "$k" --run-id "$RUN" \
    --out "$OUT/throughput-$label.json" | tee -a "$OUT/throughput.log"
  stop_server "$label"
}

# Measurement 1, in the pre-registered order.
run_arm 0 k8a 8
run_arm 1 k16 16
run_arm 2 k12 12
run_arm 3 k8b 8

# Measurement 2.
(
  set -a; . /home/$U/recall-tb/.env; set +a
  cd "$CODE"
  runuser -u "$U" -- env OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
    "$PY" scripts/aml_compiler_provider_latency.py --data "$B/locomo10.json" \
    --run-id "$RUN" --out "$OUT/providers.json" | tee "$OUT/providers.log"
)
sha256sum "$OUT"/*.json | tee "$OUT/SHA256SUMS"
