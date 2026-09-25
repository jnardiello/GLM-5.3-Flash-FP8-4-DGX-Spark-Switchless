#!/usr/bin/env bash
set -euo pipefail

# Generate this site's SIRCL runtime files from cluster.env. Nothing here talks to a node.
#
# The portable SIRCL bundle and runtime are vendored under third_party/sparkring-sircl/ and
# pinned by scripts/node/sircl/SHA256SUMS. What remains is site data, written to
# <out>/site/ (ignored by git) and pinned by <out>/SHA256SUMS.site:
#
#   rank<N>.env   SPARK_TP4_PEER0/1, SPARK_TP4_DEVICE0/1 and SPARK_TP4_GID0/1 of rank N
#   SHA256SUMS    every file the entrypoint mounts, checked by it before `vllm serve`
#
# scripts/deploy.sh copies <out>/site/* to ~/tp4/sircl/runtime/ and SHA256SUMS.site to
# ~/tp4/sircl/. Slot 0 is the first fabric port (f0) and slot 1 the second (f1); a peer's
# slot follows the ring plan of scripts/render-netplan.sh: odd links on f0, even on f1.
#
# INPUTS (cluster.env): FABRIC_TARGETS, NCCL_IB_HCA[_BY_RANK], NCCL_IB_GID_INDEX[_BY_RANK],
# FABRIC_IFACES[_BY_RANK]. SIRCL needs an explicit GID index: a rank whose NCCL value is -1
# (automatic) uses --gid-index. The entrypoint checks every selection before serving.

usage() {
  cat <<EOF
usage: $0 [--env <cluster.env>] [--gid-index <n>] [--out <dir>] [--force]

  --env <file>      recipe to read (default: cluster.env of this repository)
  --gid-index <n>   RoCEv2 GID index for ranks whose NCCL_IB_GID_INDEX is -1 (default 3,
                    the verified ASUS Ascent GX10 value; see docs/fabric.md)
  --out <dir>       output directory (default: scripts/node/sircl of this repository)
  --force           replace existing site files
  -h, --help        this text
EOF
}

REPO=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC2034  # read by scripts/lib/common.sh (log/warn/die prefix)
TP4_LOG_TAG='[sircl-site-files]'
# shellcheck source=lib/common.sh
. "$REPO/scripts/lib/common.sh"

ENV_FILE="$REPO/cluster.env"
GID_DEFAULT=$TP4_DEFAULT_NCCL_IB_GID_INDEX
OUT="$REPO/scripts/node/sircl"
FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --env) [ $# -ge 2 ] || die "--env needs a file"; ENV_FILE=$2; shift 2 ;;
    --gid-index) [ $# -ge 2 ] || die "--gid-index needs a value"; GID_DEFAULT=$2; shift 2 ;;
    --out) [ $# -ge 2 ] || die "--out needs a directory"; OUT=$2; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

[ -f "$ENV_FILE" ] || die "recipe missing: $ENV_FILE (copy cluster.env.example and fill it)"
# shellcheck source=/dev/null
. "$ENV_FILE"
tp4_validate_rank_config

# The vendored entrypoint checks these two interfaces (third_party/sparkring-sircl/README.md).
ENTRYPOINT_IFACES="enp1s0f0np0 enp1s0f1np1"
PORTABLE="$REPO/scripts/node/sircl/SHA256SUMS"
[ -f "$PORTABLE" ] || die "portable manifest missing: $PORTABLE"

valid_gid() { [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -le 255 ]; }
valid_gid "$GID_DEFAULT" || die "--gid-index must be an integer in [0,255] (got: $GID_DEFAULT)"
valid_ipv4() {
  local octet
  local -a octets
  [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || return 1
  IFS=. read -r -a octets <<<"$1"
  for octet in "${octets[@]}"; do
    case "$octet" in 0[0-9]*) return 1 ;; esac
    [ "$octet" -le 255 ] || return 1
  done
}

declare -p FABRIC_TARGETS >/dev/null 2>&1 && [ "${#FABRIC_TARGETS[@]}" -eq 4 ] \
  || die "FABRIC_TARGETS must hold exactly four entries, one per rank"

if [ "$FORCE" = 0 ] && { [ -e "$OUT/SHA256SUMS.site" ] || [ -e "$OUT/site" ]; }; then
  die "site files already exist under $OUT; keep a copy, then rerun with --force to replace them"
fi

umask 077
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

for rank in 0 1 2 3; do
  ifaces=$(tp4_resolve_rank_value "$rank" FABRIC_IFACES FABRIC_IFACES_BY_RANK "$TP4_DEFAULT_FABRIC_IFACES")
  read -r if0 if1 _ <<<"$ifaces"
  [ "$if0 $if1" = "$ENTRYPOINT_IFACES" ] \
    || die "rank $rank: first fabric ports are '$if0 $if1', but the vendored SIRCL entrypoint checks '$ENTRYPOINT_IFACES'; change runtime/entrypoint.sh and its pin first"

  hca=$(tp4_resolve_rank_value "$rank" NCCL_IB_HCA NCCL_IB_HCA_BY_RANK "$TP4_DEFAULT_NCCL_IB_HCA")
  IFS=, read -r dev0 dev1 extra <<<"$hca"
  [ -n "$dev0" ] && [ -n "$dev1" ] && [ -z "${extra:-}" ] && [ "$dev0" != "$dev1" ] \
    || die "rank $rank: NCCL_IB_HCA must name exactly two distinct devices (got: $hca)"

  gid=$(tp4_resolve_rank_value "$rank" NCCL_IB_GID_INDEX NCCL_IB_GID_INDEX_BY_RANK "$TP4_DEFAULT_NCCL_IB_GID_INDEX")
  [ "$gid" = -1 ] && gid=$GID_DEFAULT
  valid_gid "$gid" || die "rank $rank: GID index must be -1 or an integer in [0,255] (got: $gid)"

  # Each entry holds the rank's two ring peers. The peer on the link towards the next rank
  # has node number next+1 and sits on link L(rank+1); the other sits on L(rank), or L4 for
  # rank 0. Odd links use slot 0 (f0), even links slot 1 (f1).
  read -r -a peers <<<"${FABRIC_TARGETS[$rank]}"
  [ "${#peers[@]}" -eq 2 ] || die "FABRIC_TARGETS[$rank] must hold exactly two peer addresses"
  next=$(( (rank + 1) % 4 )); prev=$(( (rank + 3) % 4 ))
  peer0=""; peer1=""
  for peer in "${peers[@]}"; do
    valid_ipv4 "$peer" || die "FABRIC_TARGETS[$rank]: invalid IPv4 address '$peer'"
    host=${peer##*.}
    if [ "$host" = "$((next + 1))" ]; then link=$((rank + 1))
    elif [ "$host" = "$((prev + 1))" ]; then link=$(( rank == 0 ? 4 : rank ))
    else die "FABRIC_TARGETS[$rank]: peer '$peer' has node number $host, expected $((next + 1)) or $((prev + 1))"
    fi
    if [ $((link % 2)) -eq 1 ]; then
      [ -z "$peer0" ] || die "FABRIC_TARGETS[$rank]: both peers map to the first fabric port"
      peer0=$peer
    else
      [ -z "$peer1" ] || die "FABRIC_TARGETS[$rank]: both peers map to the second fabric port"
      peer1=$peer
    fi
  done

  printf 'SPARK_TP4_PEER0=%s\nSPARK_TP4_PEER1=%s\nSPARK_TP4_DEVICE0=%s\nSPARK_TP4_DEVICE1=%s\nSPARK_TP4_GID0=%s\nSPARK_TP4_GID1=%s\n' \
    "$peer0" "$peer1" "$dev0" "$dev1" "$gid" "$gid" > "$TMP/rank$rank.env"
done

sha_of() { shasum -a 256 "$1" | awk '{print $1}'; }

# The entrypoint runs `sha256sum --check --strict` on container paths: the bundle is mounted
# at /opt/spark-sircl and the runtime at /opt/sircl-serving.
{
  awk '$2 ~ /^bundle\// { sub(/^bundle\//, "/opt/spark-sircl/", $2); print $1 "  " $2 }' "$PORTABLE" | LC_ALL=C sort -k2
  {
    awk '$2 ~ /^runtime\// { sub(/^runtime\//, "/opt/sircl-serving/", $2); print $1 "  " $2 }' "$PORTABLE"
    for rank in 0 1 2 3; do
      printf '%s  /opt/sircl-serving/rank%s.env\n' "$(sha_of "$TMP/rank$rank.env")" "$rank"
    done
  } | LC_ALL=C sort -k2
} > "$TMP/SHA256SUMS"

{
  for rank in 0 1 2 3; do
    printf '%s  runtime/rank%s.env\n' "$(sha_of "$TMP/rank$rank.env")" "$rank"
  done
  printf '%s  runtime/SHA256SUMS\n' "$(sha_of "$TMP/SHA256SUMS")"
} > "$TMP/SHA256SUMS.site"

mkdir -p "$OUT/site"
for f in rank0.env rank1.env rank2.env rank3.env SHA256SUMS; do
  cp "$TMP/$f" "$OUT/site/$f"
done
cp "$TMP/SHA256SUMS.site" "$OUT/SHA256SUMS.site"
log "wrote $OUT/site/{rank0..3.env,SHA256SUMS} and $OUT/SHA256SUMS.site; keep them private"
