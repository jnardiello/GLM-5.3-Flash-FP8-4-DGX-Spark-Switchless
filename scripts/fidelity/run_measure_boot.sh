#!/usr/bin/env bash
# Run the pre-registered measurement steps of one measurement-mode boot, in order, with the
# rank-memory sampler and its abort file (docs/fidelity/REPORT.md). Stops at the first
# failed or aborted step; completed results stay and every collector resumes on rerun.
#
#   BASE_URL=http://HOST:8000 HOSTS="r0 r1 r2 r3" scripts/fidelity/run_measure_boot.sh ARM BOOT_LABEL STEP...
#
# Steps: prompt-a prompt-b (full corpus, K=20), prompt-k100 (k100 subset, K=100),
# prompt-crossboot (crossboot subset, K=20), prompt-ladder (ladder subset, K=20),
# prompt-rep2..9 (repeated executions on the crossboot subset, amendment 13),
# gen-a (decode set, greedy, 1024 tokens, K=20), gen-floor (decode floor subset),
# gen-nospec (decode no-spec subset).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
ARM=$1; BOOT=$2; shift 2
: "${BASE_URL:?}" "${HOSTS:?}"
PY=data/fidelity/.venv/bin/python
SUB=data/fidelity/corpus/subsets
ABORT=data/fidelity/ABORT
BOOT_JSON=data/fidelity/boots/$BOOT.json
LOG=data/fidelity/logs/measure-$ARM-$BOOT.log
umask 077
rm -f "$ABORT"
python3 scripts/fidelity/mem_sampler.py --hosts $HOSTS --out "data/fidelity/mem/$BOOT-measure.jsonl" \
  --abort-file "$ABORT" --abort-gib 1.0 >>"$LOG" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT
common=(--base-url "$BASE_URL" --arm "$ARM" --boot-json "$BOOT_JSON" --abort-file "$ABORT")
for step in "$@"; do
  echo "=== $(date -u +%FT%TZ) $ARM $step" | tee -a "$LOG"
  case "$step" in
    prompt-a|prompt-b) $PY scripts/fidelity/collect_prompt_logprobs.py "${common[@]}" --run "$step" --K 20 ;;
    prompt-k100)       $PY scripts/fidelity/collect_prompt_logprobs.py "${common[@]}" --run "$step" --K 100 --subset "$SUB/k100.txt" ;;
    prompt-crossboot)  $PY scripts/fidelity/collect_prompt_logprobs.py "${common[@]}" --run "$step" --K 20 --subset "$SUB/crossboot.txt" ;;
    prompt-ladder)     $PY scripts/fidelity/collect_prompt_logprobs.py "${common[@]}" --run "$step" --K 20 --subset "$SUB/ladder.txt" ;;
    prompt-rep[0-9])   $PY scripts/fidelity/collect_prompt_logprobs.py "${common[@]}" --run "$step" --K 20 --subset "$SUB/crossboot.txt" ;;
    gen-a)             $PY scripts/fidelity/collect_generation.py "${common[@]}" --run "$step" --K 20 --max-tokens 1024 ;;
    gen-floor)         $PY scripts/fidelity/collect_generation.py "${common[@]}" --run "$step" --K 20 --max-tokens 1024 --subset "$SUB/decode-floor.txt" ;;
    gen-nospec|gen-nospec2) $PY scripts/fidelity/collect_generation.py "${common[@]}" --run "$step" --K 20 --max-tokens 1024 --subset "$SUB/decode-nospec.txt" ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac >>"$LOG" 2>&1 || { rc=$?; echo "step $step failed rc=$rc" | tee -a "$LOG"; exit "$rc"; }
  [ ! -e "$ABORT" ] || { echo "memory abort during $step" | tee -a "$LOG"; exit 3; }
done
echo "=== $(date -u +%FT%TZ) $ARM done: $*" | tee -a "$LOG"
