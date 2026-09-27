#!/usr/bin/env bash
# Run a chain of measurement boots, one arm at a time (docs/fidelity/PLAN.md §4, amendments 13-14).
# For each entry "OVERLAY_NAME:ARM:STEP,STEP,...": coordinated stop of the currently serving
# overlay, deploy, fabric check, up, both functional gates, boot record, determinism probe,
# then the measurement steps via run_measure_boot.sh. Any failed up or gate stops the stack with
# the same overlay and ends the chain; nothing is repaired automatically.
#
#   CURRENT_TP4_ENV=scripts/node/experiments/fidelity/n-m.env \
#     scripts/fidelity/run_arm_chain.sh cpre-m:Cpre:prompt-a,prompt-rep2,prompt-rep3 ...
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
. ./cluster.env
OVL=scripts/node/experiments/fidelity
LOG=data/fidelity/logs/arm-chain.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
current=${CURRENT_TP4_ENV?set CURRENT_TP4_ENV to the overlay that is serving now (empty for the default)}
for entry in "$@"; do
  IFS=: read -r name arm steps <<<"$entry"
  label="$name-$(date -u +%m%d%H%M)"
  say "=== $name ($arm): stop ${current:-default}"
  TP4_ENV="$current" ./scripts/tp4ctl down >>"$LOG" 2>&1 || { say "down FAILED"; exit 1; }
  current="$OVL/$name.env"
  export TP4_ENV="$current"
  ./scripts/deploy.sh >>"$LOG" 2>&1 || { say "deploy FAILED"; exit 1; }
  ./scripts/tp4ctl fabric-check >>"$LOG" 2>&1 || { say "fabric-check FAILED"; exit 1; }
  if ! ./scripts/tp4ctl up >>"$LOG" 2>&1; then
    say "up FAILED for $name; coordinated stop"; ./scripts/tp4ctl down >>"$LOG" 2>&1; exit 1
  fi
  if ! python3 scripts/fidelity/gates.py --base-url "http://$MASTER_IP:8000" --health-since "$(date +%s)" \
       --out "data/fidelity/boots/$label-gates.json" >>"$LOG" 2>&1; then
    say "gates FAILED for $name; coordinated stop"; ./scripts/tp4ctl down >>"$LOG" 2>&1; exit 1
  fi
  python3 scripts/fidelity/boot_record.py --label "$label" --out "data/fidelity/boots/$label.json" \
    --public "docs/fidelity/boots/$label.json" >>"$LOG" 2>&1
  data/fidelity/.venv/bin/python scripts/fidelity/determinism_probe.py --base-url "http://$MASTER_IP:8000" \
    --x w0001 --y w0002 --out "data/fidelity/prelim/determinism-$name.json" >>"$LOG" 2>&1
  say "$name up, gates PASS, probe done"
  BASE_URL="http://$MASTER_IP:8000" HOSTS="$NODES" scripts/fidelity/run_measure_boot.sh "$arm" "$label" \
    ${steps//,/ } >>"$LOG" 2>&1 || { say "measurement steps for $name ended rc=$?"; exit 1; }
  say "$name done: $steps"
done
say "chain complete; serving overlay: $current"
