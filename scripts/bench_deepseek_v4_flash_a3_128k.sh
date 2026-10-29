#!/usr/bin/env bash
# Based on launch_deepseek_v4_flash_a3.sh reference mode, without offload or MTP.
# Defaults to four dual-die A3 cards: eight logical devices / ranks.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/bench_deepseek_v4_flash_a3_128k.sh --dry-run
  bash scripts/bench_deepseek_v4_flash_a3_128k.sh

Starts DeepSeek V4 Flash W8A8 on four Ascend A3 cards (8 DIE), benchmarks it,
then stops its server.
No expert offload; no speculative decoding.
Exactly 4 benchmark requests, concurrency 1, input 131072 / output 1024 tokens.
No extra benchmark warmup requests. Model startup still performs its own warmup.

Defaults:
  MODEL_PATH=/home/weights/DeepSeek-V4-Flash-0731-w8a8
  WORLD_SIZE=8 (logical DIE count; TP=EP=DP attention=8), SERVER_PORT=30000
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  MEM_FRACTION_STATIC=0.6 CONTEXT_LENGTH=256000
  MAX_PREFILL_TOKENS=131072 CHUNKED_PREFILL_SIZE=131072
  STARTUP_TIMEOUT=9000

Overrides:
  MODEL_PATH, WORLD_SIZE, ASCEND_RT_VISIBLE_DEVICES, PYTHON_BIN, SGLANG_ROOT
  SERVER_PORT, MEM_FRACTION_STATIC, CONTEXT_LENGTH
  MAX_PREFILL_TOKENS, CHUNKED_PREFILL_SIZE, STARTUP_TIMEOUT, LOG_DIR
  DATASET_PATH: optional local ShareGPT JSON array with at least four conversations.
    If omitted, creates four synthetic text seeds locally; no dataset download.

To use the other eight logical devices, set:
  ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
Use the logical device IDs exposed by your container, not physical card numbers.
WORLD_SIZE=16 retains the original eight-card / 16-DIE configuration.

Results: LOG_DIR/server.log, bench.log, benchmark.jsonl, validation.json.
The benchmark repeats/truncates tokenized text to 128K; it does not test accuracy.
--dry-run only prints commands; does not source CANN, start processes or write files.
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
DRY_RUN=0
case "${1:-}" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) usage >&2; die "Unknown argument: $1" ;;
esac
[[ $# -le 1 ]] || die "Unexpected arguments"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_ROOT=${SGLANG_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_PATH=${MODEL_PATH:-/home/weights/DeepSeek-V4-Flash-0731-w8a8}
WORLD_SIZE=${WORLD_SIZE:-8}
SERVER_PORT=${SERVER_PORT:-30000}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.6}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-256000}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-131072}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-131072}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-9000}
LOG_DIR=${LOG_DIR:-${TMPDIR:-/tmp}/sglang-a3-128k-$(date +%Y%m%d-%H%M%S)-$$}
[[ "$LOG_DIR" == /* ]] || LOG_DIR="$PWD/$LOG_DIR"
DATASET_PATH=${DATASET_PATH:-}
BENCH_DATASET=${DATASET_PATH:-$LOG_DIR/dataset.json}

for name in WORLD_SIZE SERVER_PORT CONTEXT_LENGTH MAX_PREFILL_TOKENS CHUNKED_PREFILL_SIZE STARTUP_TIMEOUT; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer"
done
(( SERVER_PORT <= 65535 )) || die "Invalid SERVER_PORT"
(( CONTEXT_LENGTH >= 132096 )) || die "CONTEXT_LENGTH must fit 131072 input + 1024 output tokens"
if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    ASCEND_RT_VISIBLE_DEVICES=0
    for ((i=1; i<WORLD_SIZE; i++)); do ASCEND_RT_VISIBLE_DEVICES+=",$i"; done
fi
IFS=, read -r -a devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
[[ ${#devices[@]} -eq "$WORLD_SIZE" ]] || die "Visible-device count must equal WORLD_SIZE"
for ((i=0; i<WORLD_SIZE; i++)); do
    [[ "${devices[i]}" =~ ^[0-9]+$ ]] || die "Device IDs must be numeric"
    for ((j=0; j<i; j++)); do
        [[ "${devices[i]}" != "${devices[j]}" ]] || die "Duplicate device ID"
    done
done

SERVER_CMD=(
    "$PYTHON_BIN" -m sglang.launch_server
    --model-path "$MODEL_PATH" --served-model-name deepseek-v4-flash-a3
    --host 127.0.0.1 --port "$SERVER_PORT"
    --device npu --attention-backend dsv4 --page-size 128 --trust-remote-code
    --load-format safetensors --quantization modelslim --kv-cache-dtype bfloat16
    --tp-size "$WORLD_SIZE" --ep-size "$WORLD_SIZE"
    --dp-size "$WORLD_SIZE" --enable-dp-attention
    --moe-a2a-backend deepep --deepep-mode auto --enable-dp-lm-head
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --prefill-max-requests 1 --max-running-requests 1
    --max-prefill-tokens "$MAX_PREFILL_TOKENS"
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --context-length "$CONTEXT_LENGTH"
    --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled
    --cuda-graph-bs-decode 1 --cuda-graph-max-bs-decode 1
    --watchdog-timeout 9000 --log-level info
)
BENCH_CMD=(
    "$PYTHON_BIN" -m sglang.benchmark.serving
    --backend sglang --base-url "http://127.0.0.1:$SERVER_PORT"
    --model "$MODEL_PATH" --tokenizer "$MODEL_PATH"
    --dataset-name random --dataset-path "$BENCH_DATASET" --tokenize-prompt
    --random-input-len 131072 --random-output-len 1024 --random-range-ratio 1
    --num-prompts 4 --max-concurrency 1 --request-rate inf
    --warmup-requests 0 --seed 42 --temperature 0
    --flush-cache --cache-report --output-details
    --output-file "$LOG_DIR/benchmark.jsonl"
)
printf 'No offload / no speculation; WORLD_SIZE=%s DEVICES=%s\n' "$WORLD_SIZE" "$ASCEND_RT_VISIBLE_DEVICES"
printf 'Benchmark: 131072 input / 1024 output, concurrency=1, requests=4\n'
printf 'Server: '; printf '%q ' "${SERVER_CMD[@]}"; printf '\n'
printf 'Bench: '; printf '%q ' "${BENCH_CMD[@]}"; printf '\n'
[[ "$DRY_RUN" == 0 ]] || exit 0

for command in "$PYTHON_BIN" setsid curl; do command -v "$command" >/dev/null || die "Missing command: $command"; done
[[ -d "$MODEL_PATH" ]] || die "Missing model: $MODEL_PATH"
[[ -d "$SGLANG_ROOT/python/sglang" ]] || die "Invalid SGLANG_ROOT: $SGLANG_ROOT"
[[ ! -e "$LOG_DIR" ]] || die "Use a new LOG_DIR to avoid appending to earlier benchmark results"
mkdir -p -- "$LOG_DIR"
printf '%q ' "${SERVER_CMD[@]}" > "$LOG_DIR/server-command.txt"
printf '%q ' "${BENCH_CMD[@]}" > "$LOG_DIR/bench-command.txt"

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

"$PYTHON_BIN" - "$MODEL_PATH" "$BENCH_DATASET" "$DATASET_PATH" "$SERVER_PORT" <<'PY'
import json
import socket
import sys
from pathlib import Path

model, dataset, supplied, port = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
config = json.loads((model / "config.json").read_text())
quant = json.loads((model / "quant_model_description.json").read_text())
if config.get("model_type") != "deepseek_v4" or quant.get("model_quant_type") != "W8A8_DYNAMIC":
    raise SystemExit("Expected a DeepSeek V4 ModelSlim W8A8_DYNAMIC checkpoint")
with socket.socket() as sock:
    sock.bind(("127.0.0.1", port))  # Refuse to benchmark an unrelated running server.
if not supplied:
    texts = [
        "Database indexes organize records for efficient lookup. Explain indexing with a concrete example.",
        "Solar panels convert sunlight into electricity. Describe the conversion and its practical limitations.",
        "Concurrent programs coordinate tasks through queues. Explain a producer and consumer example.",
        "Matrix multiplication combines rows and columns. Demonstrate the computation and its applications.",
    ]
    dataset.write_text(json.dumps([
        {"conversations": [{"from": "human", "value": text}, {"from": "gpt", "value": "Benchmark seed."}]}
        for text in texts
    ]))
data = json.loads(dataset.read_text())
if not isinstance(data, list) or len(data) < 4:
    raise SystemExit("DATASET_PATH must contain at least four ShareGPT conversations")
PY

# A private session keeps cleanup restricted to this invocation's server/workers.
SERVER_PID=
cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n "$SERVER_PID" ]]; then
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
        for ((i=0; i<30; i++)); do
            kill -0 -- "-$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
setsid "${SERVER_CMD[@]}" > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl --noproxy '*' -fsS --max-time 5 "http://127.0.0.1:$SERVER_PORT/ready" >/dev/null 2>&1; do
    kill -0 "$SERVER_PID" 2>/dev/null || die "Server exited; see $LOG_DIR/server.log"
    (( SECONDS < deadline )) || die "Startup timed out; see $LOG_DIR/server.log"
    sleep 2
done
curl --noproxy '*' -fsS "http://127.0.0.1:$SERVER_PORT/server_info" > "$LOG_DIR/server-info.json"
"${BENCH_CMD[@]}" 2>&1 | tee "$LOG_DIR/bench.log"

"$PYTHON_BIN" - "$LOG_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = (root / "benchmark.jsonl").read_text().splitlines()
if len(rows) != 1:
    raise SystemExit("Expected exactly one benchmark result")
result = json.loads(rows[0])
checks = {
    "completed_4": result.get("completed") == 4,
    "inputs_128k": result.get("input_lens") == [131072] * 4,
    "outputs_1k": result.get("output_lens") == [1024] * 4,
    "total_input": result.get("total_input_tokens") == 524288,
    "total_output": result.get("total_output_tokens") == 4096,
    "no_errors": len(result.get("errors", [])) == 4 and not any(result["errors"]),
}
report = {"status": "passed" if all(checks.values()) else "failed", "checks": checks}
(root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
if report["status"] != "passed":
    raise SystemExit(1)
PY
printf 'Benchmark complete. Results: %s\n' "$LOG_DIR"
