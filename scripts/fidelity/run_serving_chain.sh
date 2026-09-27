#!/usr/bin/env bash
# Serving-mode boots for tasks, voxel runs and probes, then the final E29 default
# (docs/fidelity/PLAN.md §4, §9, amendments 13-14). Entries are "OVERLAY_NAME:LABEL:STEP,...";
# steps: qeval3 (qeval greedy x3), qeval1 (greedy x1), hardset (greedy x1), voxel (both prompts,
# greedy + 3 sampled), voxelg (both prompts, greedy only), corruption, replay (cold/replay decode logprobs, same salt), bridge (decode
# no-spec subset under production scheduling). Every boot passes both gates first; a failure
# stops the stack with the same overlay and ends the chain. A failed step or a memory abort
# skips that boot's remaining steps; the chain still ends by restoring the plain default
# recipe (no TP4_ENV), both gates and check-f0.py, and then exits non-zero if any step,
# abort or the final identity check failed.
#
#   CURRENT_TP4_ENV=scripts/node/experiments/fidelity/r0fp8-m.env \
#     scripts/fidelity/run_serving_chain.sh cp-s:Cp:qeval3,hardset,voxel,corruption,replay,bridge ...
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
. ./cluster.env
OVL=scripts/node/experiments/fidelity
PY=data/fidelity/.venv/bin/python
LOG=data/fidelity/logs/serving-chain.log
ABORT=data/fidelity/ABORT
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
current=${CURRENT_TP4_ENV?set CURRENT_TP4_ENV (empty string for the default recipe)}
BASE="http://$MASTER_IP:8000"
failed=0

boot() { # $1 overlay path or "" for the default; $2 label
  say "stop ${current:-default}"
  TP4_ENV="$current" ./scripts/tp4ctl down >>"$LOG" 2>&1 || { say "down FAILED"; exit 1; }
  current=$1
  if [ -n "$current" ]; then export TP4_ENV="$current"; else unset TP4_ENV; fi
  ./scripts/deploy.sh >>"$LOG" 2>&1 || { say "deploy FAILED"; exit 1; }
  ./scripts/tp4ctl fabric-check >>"$LOG" 2>&1 || { say "fabric-check FAILED"; exit 1; }
  if ! ./scripts/tp4ctl up >>"$LOG" 2>&1; then say "up FAILED ($2); coordinated stop"; ./scripts/tp4ctl down >>"$LOG" 2>&1; exit 1; fi
  if ! python3 scripts/fidelity/gates.py --base-url "$BASE" --health-since "$(date +%s)" \
       --out "data/fidelity/boots/$2-gates.json" >>"$LOG" 2>&1; then
    say "gates FAILED ($2); coordinated stop"; ./scripts/tp4ctl down >>"$LOG" 2>&1; exit 1
  fi
  python3 scripts/fidelity/boot_record.py --label "$2" --out "data/fidelity/boots/$2.json" \
    --public "docs/fidelity/boots/$2.json" >>"$LOG" 2>&1
  say "$2 up, gates PASS"
}

KNOWN_STEPS=" qeval3 qeval1 hardset voxel voxelg corruption replay bridge "
for entry in "$@"; do  # validate every step before any boot
  IFS=: read -r name label steps <<<"$entry"
  [ -f "$OVL/$name.env" ] || { say "unknown overlay $name"; exit 2; }
  for step in ${steps//,/ }; do
    case "$KNOWN_STEPS" in *" $step "*) ;; *) say "unknown step $step in $entry"; exit 2 ;; esac
  done
done

for entry in "$@"; do
  IFS=: read -r name label steps <<<"$entry"
  boot_label="$name-$(date -u +%m%d%H%M)"
  boot "$OVL/$name.env" "$boot_label"
  rm -f "$ABORT"
  python3 scripts/fidelity/mem_sampler.py --hosts $NODES --out "data/fidelity/mem/$boot_label-serving.jsonl" \
    --abort-file "$ABORT" --abort-gib 1.0 >>"$LOG" 2>&1 &
  sampler=$!
  for step in ${steps//,/ }; do
    [ ! -e "$ABORT" ] || { say "memory abort before $step"; failed=1; break; }
    say "$label $step"
    case "$step" in
      qeval3)  $PY scripts/fidelity/tasks/run_tasks.py --set qeval --mode greedy --runs 3 --concurrency 4 \
                 --base-url "$BASE" --label "$label" ;;
      hardset) $PY scripts/fidelity/tasks/run_tasks.py --set hardset --mode greedy --concurrency 4 \
                 --base-url "$BASE" --label "$label" ;;
      qeval1)  $PY scripts/fidelity/tasks/run_tasks.py --set qeval --mode greedy --runs 1 --concurrency 4 \
                 --base-url "$BASE" --label "$label" ;;
      voxelg)  $PY scripts/fidelity/voxel/run_voxel.py --prompt pagoda-bench-artificialanalysis --sampled-runs 0 \
                 --base-url "$BASE" --label "$label" &
               v1=$!
               r=0
               $PY scripts/fidelity/voxel/run_voxel.py --prompt voxel-pagoda-miaai-x --sampled-runs 0 \
                 --base-url "$BASE" --label "$label" || r=1
               wait "$v1" || r=1
               [ "$r" -eq 0 ] ;;
      voxel)   $PY scripts/fidelity/voxel/run_voxel.py --prompt pagoda-bench-artificialanalysis --base-url "$BASE" --label "$label" &
               v1=$!
               r=0
               $PY scripts/fidelity/voxel/run_voxel.py --prompt voxel-pagoda-miaai-x --base-url "$BASE" --label "$label" || r=1
               wait "$v1" || r=1
               [ "$r" -eq 0 ] ;;
      corruption) $PY scripts/fidelity/corruption_probe.py --base-url "$BASE" --label "$label" \
                 --out "data/fidelity/probes/corruption-$label.json" ;;
      replay)  salt=data/fidelity/raw/$label/replay-salts.json
               $PY scripts/fidelity/collect_generation.py --base-url "$BASE" --arm "$label" --run cold --K 20 \
                 --max-tokens 256 --subset data/fidelity/corpus/subsets/decode-replay.txt --replay-salt "$salt" \
                 --boot-json "data/fidelity/boots/$boot_label.json" --abort-file "$ABORT" &&
               $PY scripts/fidelity/collect_generation.py --base-url "$BASE" --arm "$label" --run replay --K 20 \
                 --max-tokens 256 --subset data/fidelity/corpus/subsets/decode-replay.txt --replay-salt "$salt" \
                 --boot-json "data/fidelity/boots/$boot_label.json" --abort-file "$ABORT" ;;
      bridge)  $PY scripts/fidelity/collect_generation.py --base-url "$BASE" --arm "$label" --run gen-nospec --K 20 \
                 --max-tokens 1024 --subset data/fidelity/corpus/subsets/decode-nospec.txt \
                 --boot-json "data/fidelity/boots/$boot_label.json" --abort-file "$ABORT" ;;
      *) say "unknown step $step"; false ;;
    esac >>"$LOG" 2>&1 || { rc=$?; say "$label $step ended with rc=$rc; skipping the remaining steps"; failed=1; break; }
  done
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  [ ! -e "$ABORT" ] || { say "memory abort during $label"; failed=1; }  # checked once the sampler has exited
  say "$label serving steps done"
done
boot "" "e29-final-$(date -u +%m%d%H%M)"
if ./scripts/check-f0.py >>"$LOG" 2>&1; then
  say "FINAL: E29 default serving, gates PASS, check-f0 PASS"
else
  say "FINAL: E29 default serving, gates PASS, check-f0 FAILED (see report)"; failed=1
fi
[ "$failed" -eq 0 ] || { say "chain finished with failures"; exit 1; }
