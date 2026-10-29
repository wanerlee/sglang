#!/usr/bin/env bash
# Eight dual-die Ascend A3 cards (16 ranks), W8A8, no offload or speculation.
# Starts the server in the foreground; send benchmark requests separately.
# Graph capture is disabled by default for the prefill baseline.
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_ROOT=${SGLANG_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_PATH=${MODEL_PATH:-/home/weights/DeepSeek-V4-Flash-0731-w8a8}
WORLD_SIZE=16
ATTN_DP_SIZE=${ATTN_DP_SIZE:-2}
DECODE_GRAPH_BACKEND=${DECODE_GRAPH_BACKEND:-disabled}
SERVER_PORT=${SERVER_PORT:-30000}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.6}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-256000}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-16384}
# This fork divides the configured chunk by ATTN_DP_SIZE: 16384 / 2 = 8192.
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-16384}
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

SERVER_CMD=(
    "$PYTHON_BIN" -m sglang.launch_server
    --model-path "$MODEL_PATH" --served-model-name deepseek-v4-flash-a3
    --host 127.0.0.1 --port "$SERVER_PORT"
    --device npu --attention-backend dsv4 --page-size 128 --trust-remote-code
    --load-format safetensors --quantization modelslim --kv-cache-dtype bfloat16
    --tp-size "$WORLD_SIZE" --ep-size "$WORLD_SIZE"
    --dp-size 1 --attn-dp-size "$ATTN_DP_SIZE"
    --moe-a2a-backend deepep --deepep-mode auto --enable-dp-lm-head
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --prefill-max-requests 1 --max-running-requests 1
    --max-prefill-tokens "$MAX_PREFILL_TOKENS"
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --context-length "$CONTEXT_LENGTH"
    --cuda-graph-backend-decode "$DECODE_GRAPH_BACKEND" --cuda-graph-backend-prefill disabled
    --watchdog-timeout 9000 --log-level info
)
if [[ "$DECODE_GRAPH_BACKEND" != disabled ]]; then
    SERVER_CMD+=(--cuda-graph-bs-decode 1 --cuda-graph-max-bs-decode 1)
fi

printf 'Server: '; printf '%q ' "${SERVER_CMD[@]}"; printf '\n'
if [[ "${1:-}" == --dry-run ]]; then exit 0; fi
if [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/launch_deepseek_v4_flash_a3_128k.sh [--dry-run]" >&2
    exit 1
fi

# Reuse installed Ascend environment scripts; an already configured image is OK.
for env_script in \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/nnal/atb/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash \
    /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash; do
    if [[ -f "$env_script" ]]; then
        set +u
        # shellcheck disable=SC1090
        source "$env_script"
        set -u
    fi
done
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset ASCEND_LAUNCH_BLOCKING HCCL_IF_IP HCCL_SOCKET_FAMILY RANK_TABLE_FILE
unset SGLANG_HICACHE_HOST_MEM_BACKEND FORCE_DRAFT_MODEL_NON_QUANT
unset DEEP_USE_MODE MOE_ENABLE_TOPK_NEG_ONE MOE_EXPERT_TOKEN_NUMS_TYPE
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 ASCEND_RT_VISIBLE_DEVICES
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export STREAMS_PER_DEVICE=${STREAMS_PER_DEVICE:-32}
export INF_NAN_MODE_FORCE_DISABLE=1 SGLANG_SET_CPU_AFFINITY=1
export HCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo HCCL_OP_EXPANSION_MODE=AIV
export HCCL_CONNECT_TIMEOUT=300 HCCL_EXEC_TIMEOUT=300 HCCL_BUFFSIZE=2048
export ACL_DEVICE_SYNC_TIMEOUT=300 TASK_QUEUE_ENABLE=1
export ASCEND_USE_FIA=1 SGLANG_NPU_USE_MLAPO=0
export SGLANG_OPT_FP8_WO_A_GEMM=0 SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export SGLANG_OPT_FUSE_WQA_WKV=0 SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_DSV4_FP4_EXPERTS=False SGLANG_DSV4_FP4_DEQUANT=0 IS_DEEPSEEK_V4=1
export SGLANG_ENABLE_SPEC_V2=0 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1 DEEPEP_HCCL_BUFFSIZE=2048
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=60
export SGLANG_ENABLE_WAR_BARRIER=1 SGLANG_FORCE_COARSE_WAR_BARRIER=1


exec "${SERVER_CMD[@]}"
