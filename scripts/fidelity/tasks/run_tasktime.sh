#!/usr/bin/env bash
# tasktime: how long does an OpenCode-driven agent need to finish each of the
# six seeded repositories under third_party/knapcio-bench/bench/tasktime/tasks/?
#
# Adapted (endpoint layer only) from the vendored bench/tasktime/run_tasktime.sh:
# same six tasks, same 15-minute cap and pass/verify-with-own-tests contract,
# but pointed at this repository's arms (local vLLM or z.ai) through a
# generated, throwaway OpenCode provider config instead of the maintainer's
# global `dscode`/`glmcode4` launcher aliases. Never edits
# ~/.config/opencode/opencode.jsonc: the generated config is passed via the
# OPENCODE_CONFIG env var for the child process only.
#
# Usage:
#   run_tasktime.sh --arm local --base-url http://HOST:PORT --model NAME --label LABEL [--dry-run]
#   run_tasktime.sh --arm zai   --label LABEL [--dry-run]
#
# --dry-run prints the generated config and every command that would run,
# without invoking opencode or touching the network. This script must never
# be pointed at a real endpoint except by explicit, separate invocation.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TASKS_DIR="$REPO_ROOT/third_party/knapcio-bench/bench/tasktime/tasks"
OUT_ROOT="$REPO_ROOT/data/fidelity/tasktime"
CAP_SECONDS=900
RUNS=3
EFFORT="high"

ARM=""
LABEL=""
BASE_URL=""
MODEL=""
DRY_RUN=0

usage() {
  echo "usage: $0 --arm local|zai --label LABEL [--base-url URL] [--model NAME] [--dry-run]" >&2
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --arm) ARM="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
done

[ -n "$ARM" ] || usage
[ -n "$LABEL" ] || usage
case "$ARM" in local|zai) ;; *) echo "--arm must be local or zai" >&2; exit 2 ;; esac
if [ "$ARM" = "local" ] && [ -z "$BASE_URL" ]; then
  echo "--base-url is required for --arm local" >&2; exit 2
fi
if [ "$ARM" = "zai" ] && [ -z "${ZAI_API_KEY:-}" ] && [ "$DRY_RUN" -eq 0 ]; then
  echo "ZAI_API_KEY must be set in the environment for --arm zai" >&2; exit 2
fi

CFG_DIR="$OUT_ROOT/config"
CFG_PATH="$CFG_DIR/${LABEL}.opencode.json"
RESULTS_DIR="$OUT_ROOT/$LABEL"
mkdir -p "$CFG_DIR" "$RESULTS_DIR"

MODEL_ID="${MODEL:-glm-5.3-flash}"

# Generate a throwaway OpenCode provider config with one openai-compatible
# provider named "tasktime". Never written to the user's global config.
if [ "$ARM" = "local" ]; then
  cat > "$CFG_PATH" <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "tasktime": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "tasktime-local",
      "options": { "baseURL": "${BASE_URL%/}/v1" },
      "models": {
        "${MODEL_ID}": {
          "name": "${MODEL_ID} (tasktime local arm)",
          "reasoning": true,
          "interleaved": { "field": "reasoning_content" },
          "options": { "chat_template_kwargs": { "reasoning_effort": "${EFFORT}" } },
          "variants": { "${EFFORT}": { "reasoningEffort": "${EFFORT}" } }
        }
      }
    }
  }
}
EOF
else
  cat > "$CFG_PATH" <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "tasktime": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "tasktime-zai",
      "options": { "baseURL": "https://api.z.ai/api/paas/v4" },
      "models": {
        "glm-5.3-flash": {
          "name": "glm-5.3-flash (tasktime z.ai arm)",
          "reasoning": true,
          "interleaved": { "field": "reasoning_content" },
          "options": { "reasoning_effort": "${EFFORT}" },
          "variants": { "${EFFORT}": { "reasoningEffort": "${EFFORT}" } }
        }
      }
    }
  }
}
EOF
  MODEL_ID="glm-5.3-flash"
fi

echo "wrote OpenCode provider config: $CFG_PATH"
[ "$DRY_RUN" -eq 1 ] && cat "$CFG_PATH"

run_one_task() {
  task_dir="$1"; run_idx="$2"
  name="$(basename "$task_dir")"
  work="$RESULTS_DIR/run${run_idx}/${name}"
  rm -rf "$work"
  mkdir -p "$(dirname "$work")"
  cp -R "$task_dir" "$work"

  prompt="$(cat "$work/TASK.md") Work only inside this directory. Run the tests yourself and stop when they pass."

  cmd=(env OPENCODE_CONFIG="$CFG_PATH" \
       opencode run --dir "$work" --model "tasktime/${MODEL_ID}" --variant "${EFFORT}" \
       --format json "$prompt")

  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY RUN task=%s run=%s cap=%ss cmd=' "$name" "$run_idx" "$CAP_SECONDS"
    printf '%q ' "${cmd[@]}"
    echo
    echo '{"task":"'"$name"'","run":'"$run_idx"',"dry_run":true}' > "$work.result.json"
    return 0
  fi

  t0=$(date +%s)
  agent_rc=0
  perl -e 'alarm shift; exec @ARGV' "$CAP_SECONDS" "${cmd[@]}" > "$work/agent.log" 2>&1 || agent_rc=$?
  t1=$(date +%s)
  secs=$((t1 - t0))

  verify_rc=0
  if [ -f "$work/package.json" ]; then
    (cd "$work" && perl -e 'alarm 120; exec @ARGV' node --test) > "$work/verify.log" 2>&1 || verify_rc=$?
  else
    (cd "$work" && perl -e 'alarm 120; exec @ARGV' python3 -m pytest -q -x -p no:cacheprovider) > "$work/verify.log" 2>&1 || verify_rc=$?
  fi

  status="FAIL"; [ "$verify_rc" -eq 0 ] && status="PASS"
  tool_lines=$(grep -c -i -E "tool|bash|edit|write" "$work/agent.log" || true)

  python3 - "$work.result.json" "$name" "$run_idx" "$secs" "$status" "$agent_rc" "$tool_lines" <<'PYEOF'
import json, sys
out_path, name, run_idx, secs, status, agent_rc, tool_lines = sys.argv[1:8]
json.dump({
    "task": name, "run": int(run_idx), "seconds": int(secs), "verify": status,
    "agent_rc": int(agent_rc), "tool_lines": int(tool_lines),
}, open(out_path, "w"), indent=1)
PYEOF
  echo "task=$name run=$run_idx secs=$secs verify=$status agent_rc=$agent_rc tool_lines=$tool_lines"
}

for run_idx in $(seq 1 "$RUNS"); do
  for task_dir in "$TASKS_DIR"/*/; do
    run_one_task "${task_dir%/}" "$run_idx"
  done
done

echo "results under: $RESULTS_DIR"
