#!/usr/bin/env bash
set -euo pipefail

# Fidelity campaign arm N: Alex Ellis's GLM-5.3-Flash NVFP4 TP4 recipe
# (github.com/alexellis/glm-5.3-flash-4x-dgx-spark-switchless @ e2d0839a, MIT,
# scripts/rank-launcher.sh) run on this repository's fabric. The ENGINE layer is his:
# image, checkpoint, drafter, chat template and serve arguments, verbatim. The FABRIC layer
# is ours: the site's netplan, the pinned patched NCCL, our NCCL environment and the
# validated HCA/GID selection, exactly as this repository's September 11 lane ran the same
# image. His fabric-specific NCCL_SWITCHLESS_RING_ONLY and NCCL_IB_MERGE_NICS=0 are not set
# (the first boot with them failed an RDMA QP connect). Deviations: docs/fidelity/PLAN.md.
# Selected by an overlay with LAUNCHER=experiments/fidelity/launch-nvfp4-tp4.sh; tp4ctl
# runs it as ~/tp4/$LAUNCHER <rank>, so cluster.env lives two directories up.

HERE=$(cd "$(dirname "$0")" && pwd)
if [ -f "$HERE/../../cluster.env" ]; then
  ENV_DIR=$(cd "$HERE/../.." && pwd)                       # node: ~/tp4/experiments/fidelity
elif [ -f "$HERE/../../../../cluster.env" ]; then
  ENV_DIR=$(cd "$HERE/../../../.." && pwd)                 # checkout (dry runs only)
else
  echo "[nvfp4] ERROR: cluster.env not found" >&2; exit 1
fi
. "$ENV_DIR/cluster.env"
BASE_CONTAINER=${CONTAINER-}
if [ -n "${TP4_ENV:-}" ]; then
  case "$TP4_ENV" in /*|*..*) echo "[nvfp4] ERROR: bad TP4_ENV" >&2; exit 1 ;; esac
  [[ "$TP4_ENV" =~ ^[A-Za-z0-9._/-]+$ ]] || { echo "[nvfp4] ERROR: bad TP4_ENV" >&2; exit 1; }
  [ -f "$ENV_DIR/$TP4_ENV" ] || { echo "[nvfp4] ERROR: overlay missing: $TP4_ENV" >&2; exit 1; }
  # shellcheck disable=SC1090
  . "$ENV_DIR/$TP4_ENV"
fi
[ "${CONTAINER-}" = "$BASE_CONTAINER" ] || { echo "[nvfp4] ERROR: TP4_ENV must not change CONTAINER" >&2; exit 1; }
DRY_RUN=${TP4_DRY_RUN:-0}

expand_home() { local v=$1; v=${v/#\$HOME/$HOME}; v=${v/#\~/$HOME}; printf '%s' "$v"; }
resolve_rank_value() {
  local rank=$1 scalar=$2 array=$3 fallback=$4 n=0 value=""
  if declare -p "$array" >/dev/null 2>&1; then
    eval "n=\${#${array}[@]}"
    if [ "$n" -gt 0 ]; then eval "value=\${${array}[$rank]-}"; printf '%s' "$value"; return; fi
  fi
  eval "value=\${${scalar}:-}"; printf '%s' "${value:-$fallback}"
}

read -r -a _MGMT_IPS <<<"$MGMT_IPS"
[ $# -eq 1 ] && [[ "$1" =~ ^[0-3]$ ]] || { echo "usage: $0 <rank 0-3>" >&2; exit 2; }
RANK=$1
MIP=${_MGMT_IPS[$RANK]}
MGMT_IF=$(resolve_rank_value "$RANK" MGMT_IF MGMT_IF_BY_RANK enP7s7)
FABRIC_IFACES=$(resolve_rank_value "$RANK" FABRIC_IFACES FABRIC_IFACES_BY_RANK "enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1")
NCCL_IB_HCA=$(resolve_rank_value "$RANK" NCCL_IB_HCA NCCL_IB_HCA_BY_RANK rocep1s0f0,rocep1s0f1)
NCCL_IB_GID_INDEX=$(resolve_rank_value "$RANK" NCCL_IB_GID_INDEX NCCL_IB_GID_INDEX_BY_RANK 3)

MODEL_DIR=$(expand_home "$MODEL_DIR")
DRAFT_DIR=$(expand_home "$DRAFT_DIR")
NCCL_DIR=$(expand_home "$NCCL_DIR")
N_CACHE_DIR=$(expand_home "${N_CACHE_DIR:?}")
N_CHAT_TEMPLATE=$(expand_home "${N_CHAT_TEMPLATE:?}")
# Published recipe values; measurement overlays change only these four.
N_MAX_MODEL_LEN=${N_MAX_MODEL_LEN:-262144}
N_MAX_NUM_SEQS=${N_MAX_NUM_SEQS:-6}
N_KV_CACHE_MEMORY=${N_KV_CACHE_MEMORY:-12884901888}
N_SPEC_TOKENS=${N_SPEC_TOKENS:-7}
N_EXTRA_ARGS=${N_EXTRA_ARGS:-}
[[ "$N_SPEC_TOKENS" =~ ^[0-9]+$ ]] || { echo "[nvfp4] ERROR: N_SPEC_TOKENS" >&2; exit 1; }
# Upstream zai-org chat template at the pinned revision, as the published recipe checks.
N_CHAT_TEMPLATE_SHA256=0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5

if [ "$DRY_RUN" != 1 ]; then
  LOCAL_IPS=$(ip -4 addr show "$MGMT_IF" | awk '/inet /{print $2}' | cut -d/ -f1)
  echo "$LOCAL_IPS" | grep -qx "$MIP" || { echo "[nvfp4] ERROR: rank $RANK is not this node" >&2; exit 1; }
  GID_SELECTION=$(python3 "$ENV_DIR/scripts/nccl_gid_check.py" --hcas "$NCCL_IB_HCA" \
      --gid-index "$NCCL_IB_GID_INDEX" --fabric-ifaces "$FABRIC_IFACES") \
    || { echo "[nvfp4] ERROR: HCA/GID preflight failed" >&2; exit 1; }
  echo "[nvfp4] NCCL HCA/GID: $GID_SELECTION"
  [ -f "$MODEL_DIR/config.json" ] || { echo "[nvfp4] ERROR: missing $MODEL_DIR/config.json" >&2; exit 1; }
  [ -f "$DRAFT_DIR/model.safetensors" ] || { echo "[nvfp4] ERROR: missing drafter" >&2; exit 1; }
  [ -f "$NCCL_DIR/libnccl.so.2" ] || { echo "[nvfp4] ERROR: missing patched NCCL" >&2; exit 1; }
  printf '%s  %s\n' "$N_CHAT_TEMPLATE_SHA256" "$N_CHAT_TEMPLATE" | sha256sum --check --status - \
    || { echo "[nvfp4] ERROR: chat template does not match the pinned revision" >&2; exit 1; }
  ID=$(sudo docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null) \
    || { echo "[nvfp4] ERROR: image not present: $IMAGE" >&2; exit 1; }
  if [ -n "${IMAGE_ID:-}" ] && [ "$ID" != "$IMAGE_ID" ]; then
    echo "[nvfp4] ERROR: image content ID mismatch: $ID" >&2; exit 1
  fi
  mkdir -p "$N_CACHE_DIR"
  sudo sysctl -qw vm.swappiness=0
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  sudo docker rm -f "$CONTAINER" 2>/dev/null || true
fi

if [ "$RANK" = 0 ]; then TAIL=(--host 0.0.0.0 --port "$API_PORT"); else TAIL=(--headless); fi
SPEC=()
if [ "$N_SPEC_TOKENS" -gt 0 ]; then
  SPEC=(--speculative-config "{\"method\":\"dflash\",\"model\":\"/draft\",\"num_speculative_tokens\":$N_SPEC_TOKENS}")
fi
# shellcheck disable=SC2206
EXTRA=( $N_EXTRA_ARGS )

DOCKER_CMD=(
  sudo docker run -d --name "$CONTAINER" --restart no
  --cap-add IPC_LOCK --ulimit memlock=-1:-1
  --network host --ipc host --shm-size 32g --gpus all
  --device /dev/infiniband:/dev/infiniband
  -v "$MODEL_DIR:/model:ro" -v "$DRAFT_DIR:/draft:ro"
  -v "$NCCL_DIR:/opt/patched-nccl:ro" -v "$N_CACHE_DIR:/cache"
  -v "$N_CHAT_TEMPLATE:/opt/glm53/chat_template.jinja:ro"
  -e LD_PRELOAD=/opt/patched-nccl/libnccl.so.2 -e VLLM_NCCL_SO_PATH=/opt/patched-nccl/libnccl.so.2
  -e NCCL_SKIP_TREE_CONNECT=1
  -e NCCL_SOCKET_IFNAME="$MGMT_IF" -e GLOO_SOCKET_IFNAME="$MGMT_IF" -e VLLM_HOST_IP="$MIP"
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0
  -e NCCL_IB_HCA="$NCCL_IB_HCA" -e NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX"
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET
  -e NCCL_IB_SUBNET_PREFIX_LEN=24 -e NCCL_IB_SUBNET_AWARE_ROUTING=1
  -e NCCL_ALGO=Ring -e NCCL_PROTO=LL,LL128,Simple -e NCCL_P2P_LEVEL=SYS
  -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 -e NCCL_CROSS_NIC=1 -e NCCL_CUMEM_ENABLE=0
  -e NCCL_NVLS_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN
  -e VLLM_ONE_GPU_PER_NODE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1
  -e HF_HOME=/cache/hf -e XDG_CACHE_HOME=/cache -e VLLM_CACHE_ROOT=/cache/vllm
  -e NODE_RANK="$RANK" -e MASTER_ADDR="$MASTER_IP"
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a
  "$IMAGE"
  /model
  --served-model-name "$SERVED_NAME" --trust-remote-code
  --tensor-parallel-size 4 --nnodes 4 --node-rank "$RANK"
  --master-addr "$MASTER_IP" --master-port "$MASTER_PORT"
  --gpu-memory-utilization 0.85 --max-model-len "$N_MAX_MODEL_LEN"
  --max-num-seqs "$N_MAX_NUM_SEQS" --block-size 2304 --moe-backend marlin
  --limit-mm-per-prompt '{"image":16}'
  --kv-cache-dtype fp8_e4m3 --kv-cache-memory "$N_KV_CACHE_MEMORY"
  ${SPEC[@]+"${SPEC[@]}"}
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45
  --chat-template /opt/glm53/chat_template.jinja
  --default-chat-template-kwargs '{"reasoning_effort":"max"}'
  --distributed-executor-backend mp
  ${EXTRA[@]+"${EXTRA[@]}"}
  "${TAIL[@]}"
)

if [ "$DRY_RUN" = 1 ]; then
  echo "[dry-run] rank $RANK ($MIP) — docker command that would be executed:"
  printf '  %s\n' "${DOCKER_CMD[@]}"
  exit 0
fi
"${DOCKER_CMD[@]}"
echo "container started: $CONTAINER (rank $RANK, $MIP, NVFP4 comparison arm)"
