#!/usr/bin/env bash
# Regenerate every fidelity metric, figure and report from data/fidelity/raw/:
#   1. analyze_campaign.py  -> docs/fidelity/metrics-v2/*.json (public), data/fidelity/metrics-v2/ (private)
#   2. plots.py             -> docs/fidelity/plots/*.png, *.svg and plots.json
#   3. build_report.py      -> docs/fidelity/REPORT.md, docs/fidelity/report.html, then the leak check
# Needs numpy and matplotlib==3.11.2; PY defaults to data/fidelity/.venv/bin/python.
# Missing runs are reported as pending, never fatal. Extra arguments go to plots.py
# (for example --stride 4 for a faster, deterministically subsampled per-position pass).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$ROOT/data/fidelity/.venv/bin/python}"
cd "$ROOT"

if [[ ! -x "$PY" ]]; then
  echo "make_all: $PY not found; set PY to a Python with numpy and matplotlib" >&2
  exit 1
fi

echo "== metrics"
"$PY" scripts/fidelity/analyze_campaign.py --config scripts/fidelity/campaign.config.json
echo "== tasks and probes"
# qeval: exact McNemar and paired bootstrap need two arms; E29 production (Cp) against the
# reference (R0) is the pre-registered pair, against NVFP4 (N) the second question.
for pair in "R0 Cp" "N Cp"; do
  read -r arm_a arm_b <<<"$pair"
  if [[ -d "data/fidelity/tasks/$arm_a/qeval" && -d "data/fidelity/tasks/$arm_b/qeval" ]]; then
    lc=$(echo "$arm_b-vs-$arm_a" | tr '[:upper:]' '[:lower:]')
    "$PY" scripts/fidelity/tasks/stats.py --set qeval --arms "$arm_a" "$arm_b" \
      --out "docs/fidelity/metrics/tasks-qeval-$lc.json" >/dev/null
  fi
done
mkdir -p docs/fidelity/metrics
for f in data/fidelity/probes/corruption-*.json; do
  [[ -e "$f" ]] || continue
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); json.dump(d["summary"], open(sys.argv[2],"w"), indent=1)' \
    "$f" "docs/fidelity/metrics/$(basename "$f")"
done
echo "== figures"
"$PY" scripts/fidelity/plots.py "$@"
echo "== reports"
"$PY" scripts/fidelity/build_report.py
