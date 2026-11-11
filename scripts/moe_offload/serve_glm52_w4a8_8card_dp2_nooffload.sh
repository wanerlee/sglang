#!/usr/bin/env bash
# GLM-5.2 ModelSlim W4A8 mixed serving baseline, without MoE offload or MTP.
# Match serve_glm52_w4a8_8card_dp2.sh: eight physical cards / 16 dies by default.
# Source the matching Ascend environment before running. No benchmark or cleanup.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to a local GLM-5.2 ModelSlim W4A8 checkpoint}
WORLD_SIZE=${WORLD_SIZE:-16}
case "$WORLD_SIZE" in
    8|16) ;;
    *) echo "WORLD_SIZE must be 8 or 16 logical NPU devices" >&2; exit 1 ;;
esac
ATTN_DP_SIZE=2
ATTN_TP_SIZE=$((WORLD_SIZE / ATTN_DP_SIZE))
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.80}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-32}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-131072}
SERVER_HOST=${SERVER_HOST:-127.0.0.1}
SERVER_PORT=${SERVER_PORT:-8818}
DIST_PORT=${DIST_PORT:-29500}
LOG_DIR=${LOG_DIR:-/tmp/sglang-glm52-w4a8-dp2-nooffload/$(date +%Y%m%d_%H%M%S)-$$}

DRY_RUN=0
case "${1:-}" in
    --dry-run) DRY_RUN=1; shift ;;
    "") ;;
    *) echo "Usage: MODEL_PATH=... bash $0 [--dry-run]" >&2; exit 1 ;;
esac
[[ $# -eq 0 ]]
[[ "$MAX_RUNNING_REQUESTS" =~ ^[1-9][0-9]*$ ]] && ((MAX_RUNNING_REQUESTS <= 128))

if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    ASCEND_RT_VISIBLE_DEVICES=0
    for ((i=1; i<WORLD_SIZE; i++)); do ASCEND_RT_VISIBLE_DEVICES+=",$i"; done
fi
IFS=, read -r -a devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
if [[ ${#devices[@]} -ne "$WORLD_SIZE" ]]; then
    echo "ASCEND_RT_VISIBLE_DEVICES must list WORLD_SIZE logical NPU devices" >&2
    exit 1
fi
export ASCEND_RT_VISIBLE_DEVICES
export PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-lo}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-auto}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-300}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-300}
export ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT:-300}
export STREAMS_PER_DEVICE=${STREAMS_PER_DEVICE:-32}
export INF_NAN_MODE_FORCE_DISABLE=1 TASK_QUEUE_ENABLE=1
export SGLANG_SET_CPU_AFFINITY=${SGLANG_SET_CPU_AFFINITY:-1}
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0 SGLANG_ENABLE_SPEC_V2=0
export DEEP_USE_MODE=default MOE_ENABLE_TOPK_NEG_ONE=1 MOE_EXPERT_TOKEN_NUMS_TYPE=1
export SGLANG_ZBAL_LOCAL_MEM_SIZE=0 SGLANG_NPU_USE_MULTI_STREAM=1
export SGLANG_ENABLE_WAR_BARRIER=1 SGLANG_FORCE_COARSE_WAR_BARRIER=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128

# Keep the same graph buckets as the offload comparison.
GRAPH_MAX_BS=$(((MAX_RUNNING_REQUESTS + ATTN_TP_SIZE - 1) / ATTN_TP_SIZE * ATTN_TP_SIZE))
GRAPH_BS=()
for ((bs=ATTN_TP_SIZE; bs<GRAPH_MAX_BS; bs*=2)); do GRAPH_BS+=("$bs"); done
GRAPH_BS+=("$GRAPH_MAX_BS")

cmd=("$PYTHON_BIN" -u -m sglang.launch_server
    --model-path "$MODEL_PATH" --served-model-name glm52-w4a8-nooffload
    --host "$SERVER_HOST" --port "$SERVER_PORT" --device npu --base-gpu-id 0
    --dist-init-addr "127.0.0.1:$DIST_PORT" --trust-remote-code
    --attention-backend ascend --page-size 128
    --tp-size "$WORLD_SIZE" --ep-size "$WORLD_SIZE" --dp-size 1 --moe-dp-size 1
    --attn-dp-size "$ATTN_DP_SIZE" --enable-dp-lm-head
    --load-format safetensors --quantization modelslim --kv-cache-dtype bfloat16
    --moe-a2a-backend ascend_ops --deepep-mode auto
    --moe-ascend-ops-config '{"prefill":"alltoallv","decode":"low_latency"}'
    --disable-overlap-schedule --disable-shared-experts-fusion
    --mem-fraction-static "$MEM_FRACTION_STATIC" --context-length "$CONTEXT_LENGTH"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --max-prefill-tokens "$CHUNKED_PREFILL_SIZE" --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled
    --cuda-graph-bs-decode "${GRAPH_BS[@]}" --cuda-graph-max-bs-decode "$GRAPH_MAX_BS"
    --skip-server-warmup --watchdog-timeout 9000 --decode-log-interval 10 --log-level info)

printf 'MoE offload disabled; TP=EP=%s Attention DP=2 Attention TP=%s\n' \
    "$WORLD_SIZE" "$ATTN_TP_SIZE"
if ((DRY_RUN)); then
    printf '%q ' "${cmd[@]}"; printf '\n'
    exit 0
fi

mkdir -p -- "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/server.log") 2>&1
exec "${cmd[@]}"
