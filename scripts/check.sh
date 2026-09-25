#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1

public_path() {
  case "$1" in
    ./.git/*|./.claude/*|./docs/rigmark_reports/*|./scripts/mirror-snapshot.sh|\
    ./scripts/mirror-allow.txt|./scripts/mirror-private-terms.example)
      return 1 ;;
    *) return 0 ;;
  esac
}

required=(
  docs/install-from-zero.md
  docs/operations.md
  docs/fabric.md
  docs/production-recipe.md
  docs/benchmarks/README.md
  docs/rigmark_reports/README.md
  scripts/tp4ctl
  scripts/check-f0.py
  scripts/prepare-sparkcache.py
  scripts/f0-reference.py
  scripts/launcher/launch-glm53-tp4.sh
  scripts/agent-preflight.sh
  scripts/nccl_gid_check.py
  scripts/bootstrap-node.sh
  scripts/deploy.sh
  scripts/deploy-host.sh
  scripts/fetch-fp8-weights.sh
  scripts/render-netplan.sh
  scripts/verify-node.sh
  scripts/node/flusher-unconditional.sh
  scripts/node/sparse_attn_indexer_kpool_sm121.py
  scripts/node/host/tp4-iommu.sh
  scripts/node/nccl/build.sh
  scripts/node/nccl/install-nccl.sh
  scripts/node/patches/adaptive_k_scheduler.py
)
for path in "${required[@]}"; do
  [ -e "$path" ] || { echo "check: missing required public path: $path" >&2; exit 1; }
done

shell_count=0
while IFS= read -r file; do
  public_path "$file" || continue
  bash -n "$file"
  shell_count=$((shell_count + 1))
done < <(find . -type f -name '*.sh' -print | sort)
echo "bash-syntax: PASS ($shell_count files)"
bash -n scripts/tp4ctl
echo "controller-syntax: PASS"

python_count=0
while IFS= read -r file; do
  public_path "$file" || continue
  python3 -c 'import ast, pathlib, sys; p=pathlib.Path(sys.argv[1]); ast.parse(p.read_text(encoding="utf-8"), str(p))' "$file"
  python_count=$((python_count + 1))
done < <(find . -type f -name '*.py' -print | sort)
echo "python-ast: PASS ($python_count files)"

python3 scripts/check_markdown_links.py

./scripts/tp4ctl --help >/dev/null
python3 scripts/check-f0.py --help >/dev/null
python3 scripts/prepare-sparkcache.py --help >/dev/null
python3 scripts/f0-reference.py --help >/dev/null
./scripts/deploy.sh --help >/dev/null
./scripts/deploy-host.sh --help >/dev/null
./scripts/bootstrap-node.sh --help >/dev/null
./scripts/verify-node.sh --help >/dev/null
./scripts/fetch-fp8-weights.sh --help >/dev/null
./scripts/render-netplan.sh --help >/dev/null
./scripts/agent-preflight.sh --help >/dev/null 2>&1
scripts/node/nccl/build.sh --help >/dev/null
scripts/node/nccl/install-nccl.sh --help >/dev/null
echo "command-help: PASS"

./scripts/tests/test-agent-preflight.sh
python3 scripts/tests/test-nccl-gid-selection.py
python3 scripts/tests/test-sircl-gid-selection.py
python3 scripts/tests/test-check-f0.py
python3 scripts/tests/test-verify-node.py
python3 scripts/tests/test-prepare-sparkcache.py
python3 scripts/tests/test-sparkcache-launcher.py
python3 scripts/tests/test-kda-hybrid.py
python3 scripts/tests/test-e03-config.py
python3 scripts/tests/test-adaptive-draft-budget.py
python3 scripts/tests/test-draft-budget-config.py
python3 scripts/tests/test-bf16-residue-config.py
python3 scripts/tests/test-drafter-w8a16-config.py
python3 scripts/tests/test-long-prefill-cadence-config.py
python3 scripts/tests/test-queued-cadence-config.py
python3 scripts/tests/test-draft-depth-7-config.py
python3 scripts/tests/test-end-drain-config.py
python3 scripts/tests/test-end-drain-policy.py
python3 scripts/tests/test-accepted-recipe.py
python3 scripts/tests/test-f0-reference.py
./scripts/tests/test-host-lifecycle.sh
bash ./scripts/tests/test-controller-lifecycle.sh
python3 scripts/tests/test-model-snapshot.py
python3 scripts/tests/test-chat-template.py
python3 scripts/node/patches/test_adaptive_k_policy.py

echo "check: PASS"
