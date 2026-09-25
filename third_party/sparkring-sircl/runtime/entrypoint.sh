#!/usr/bin/env bash
set -euo pipefail
umask 077

rank=''
expect_rank=0
rank_count=0
for argument in "$@"; do
  if [[ "$expect_rank" == 1 ]]; then
    rank=$argument
    expect_rank=0
    continue
  fi
  case "$argument" in
    --node-rank) rank_count=$((rank_count + 1)); expect_rank=1 ;;
    --node-rank=*) rank_count=$((rank_count + 1)); rank=${argument#*=} ;;
  esac
done
[[ "$expect_rank" == 0 && "$rank_count" == 1 && "$rank" =~ ^[0-3]$ ]] || {
  printf 'SIRCL serving entrypoint requires exactly one TP4 node rank\n' >&2
  exit 78
}

sha256sum --check --strict /opt/sircl-serving/SHA256SUMS >/dev/null
set -a
# shellcheck source=/dev/null
. /opt/sircl-serving/common.env
# shellcheck source=/dev/null
. "/opt/sircl-serving/rank${rank}.env"
set +a

[[ "$SPARKRING_SIRCL_NATIVE_SHA256" == f53c88b4bf885533c4d6de60dae1a9e46cf26db0695b37265955b7cc74934119 ]]
[[ "$SPARKRING_SIRCL_MANIFEST_SHA256" == 85a231e6d2a290f7d6cccbc2cc6b1ccad7a6adbefc7ce4dde05b158f249aadd4 ]]
python3 -S /opt/sircl-serving/sircl_gid_check.py --fabric-ifaces enp1s0f0np0 enp1s0f1np1
exec vllm serve "$@"
