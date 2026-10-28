#!/usr/bin/env bash
# Single-node Ascend A5, DeepSeek V4 FP4 or ModelSlim W8A8, TP=EP=WORLD_SIZE.
# Source the matching CANN/torch_npu environment before running this script.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  MODEL_PATH=/path/to/DeepSeek-V4-Flash bash scripts/launch_moe_offload_npu.sh
  MODEL_PATH=/path/to/DeepSeek-V4-Flash bash scripts/launch_moe_offload_npu.sh --dry-run

Defaults: WORLD_SIZE=4, ASCEND_RT_VISIBLE_DEVICES=0,1,2,3, TP=4, EP=4.
ATTN_DP_SIZE=1 by default; model-replica DP=1 and MoE DP=1 stay fixed.
MODEL_PATH must point to local DeepSeek V4 FP4 or ModelSlim W8A8 safetensors.
GLM, converted FP8 experts and speculative decoding are not supported.

Overrides:
  SGLANG_ROOT, PYTHON_BIN, WORLD_SIZE, ASCEND_RT_VISIBLE_DEVICES
  ATTN_DP_SIZE=1 (must divide WORLD_SIZE; e.g. 8/8 or 16/16)
  MOE_EXPERT_FORMAT=auto (fp4 or w8a8; auto reads quant_model_description.json)
  MOE_OFFLOAD_CONFIG (defaults to the repository's npu/examples/v4.json)
  SERVER_HOST=127.0.0.1 SERVER_PORT=8818 SERVED_MODEL_NAME=deepseek-v4-moe-offload
  MAX_RUNNING_REQUESTS=8 CHUNKED_PREFILL_SIZE=1024 CONTEXT_LENGTH=8192
  MEM_FRACTION_STATIC=0.8 BUILD_NATIVE=1
  LOG_DIR (defaults to a new directory under /tmp/sglang-moe-offload)
  LOG_LEVEL=info MOE_LOG_LEVEL=info LOG_LEVEL_HTTP=warning DECODE_LOG_INTERVAL=40
  LOG_REQUESTS=1 LOG_REQUESTS_LEVEL=0 (metadata only) LOG_REQUESTS_FORMAT=json
  ENABLE_TRACE=0 OTLP_TRACES_ENDPOINT=localhost:4317 OTLP_SERVICE_NAME
  PROFILE_CAPTURE=0 (NPU graph capture profiling; explicit opt-in)

Set MOE_LOG_LEVEL=debug for MoE layer/wave/copy submission logs. Each batch's
assignment hit rate is already enabled by the offload JSON (batch_log_interval=1).
Logs, the config snapshot and the generated logging config are saved in LOG_DIR.
An existing SGLANG_LOGGING_CONFIG_PATH takes precedence over the generated config.
Trace requires an OTLP collector and installed OpenTelemetry exporters. It covers
request spans; PROFILE_CAPTURE records the NPU warmup/capture timeline instead.

--dry-run prints the environment, native build and server commands without
building, loading weights or initializing NPU devices. Budgets in the JSON are
per rank: the default host reservation is 128 GiB/rank, 512 GiB for four ranks.
No log directory or config snapshots are created by --dry-run. Offline statistics
are optional: without offline_stats_path, initial slots are spread across layers.
With a profile, offline_model_id optionally checks its model label. Online cache
replacement remains global within each EP rank.
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

DRY_RUN=0
case "${1:-}" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) usage >&2; die "Unknown argument: $1" ;;
esac
[[ $# -eq 0 ]] || die "Unexpected arguments: $*"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_ROOT=${SGLANG_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_PATH=${MODEL_PATH:-}
WORLD_SIZE=${WORLD_SIZE:-4}
ATTN_DP_SIZE=${ATTN_DP_SIZE:-1}
SERVER_HOST=${SERVER_HOST:-127.0.0.1}
SERVER_PORT=${SERVER_PORT:-8818}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-deepseek-v4-moe-offload}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-8}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-1024}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-8192}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.8}
BUILD_NATIVE=${BUILD_NATIVE:-1}
MOE_OFFLOAD_CONFIG=${MOE_OFFLOAD_CONFIG:-$SGLANG_ROOT/python/sglang/srt/offloading/moe/npu/examples/v4.json}
LOG_DIR=${LOG_DIR:-${TMPDIR:-/tmp}/sglang-moe-offload/$(date +%Y%m%d-%H%M%S)-$$}
[[ "$LOG_DIR" == /* ]] || LOG_DIR="$PWD/$LOG_DIR"
LOG_LEVEL=${LOG_LEVEL:-info}
MOE_LOG_LEVEL=${MOE_LOG_LEVEL:-$LOG_LEVEL}
LOG_LEVEL_HTTP=${LOG_LEVEL_HTTP:-warning}
DECODE_LOG_INTERVAL=${DECODE_LOG_INTERVAL:-40}
LOG_REQUESTS=${LOG_REQUESTS:-1}
LOG_REQUESTS_LEVEL=${LOG_REQUESTS_LEVEL:-0}
LOG_REQUESTS_FORMAT=${LOG_REQUESTS_FORMAT:-json}
ENABLE_TRACE=${ENABLE_TRACE:-0}
OTLP_TRACES_ENDPOINT=${OTLP_TRACES_ENDPOINT:-localhost:4317}
OTLP_SERVICE_NAME=${OTLP_SERVICE_NAME:-$SERVED_MODEL_NAME}
PROFILE_CAPTURE=${PROFILE_CAPTURE:-0}

[[ -n "$MODEL_PATH" ]] || die "Set MODEL_PATH to a DeepSeek V4 FP4 or W8A8 checkpoint; see --help."
for name in WORLD_SIZE ATTN_DP_SIZE SERVER_PORT MAX_RUNNING_REQUESTS CHUNKED_PREFILL_SIZE CONTEXT_LENGTH DECODE_LOG_INTERVAL; do
    value=${!name}
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer."
done
(( WORLD_SIZE % ATTN_DP_SIZE == 0 )) || die "ATTN_DP_SIZE must divide WORLD_SIZE."
(( SERVER_PORT <= 65535 )) || die "SERVER_PORT exceeds 65535."
(( MAX_RUNNING_REQUESTS <= 128 )) || die "MAX_RUNNING_REQUESTS must not exceed the DeepEP dispatch limit (128)."
[[ "$BUILD_NATIVE" =~ ^[01]$ ]] || die "BUILD_NATIVE must be 0 or 1."
for name in LOG_REQUESTS ENABLE_TRACE PROFILE_CAPTURE; do
    [[ "${!name}" =~ ^[01]$ ]] || die "$name must be 0 or 1."
done
for name in LOG_LEVEL MOE_LOG_LEVEL LOG_LEVEL_HTTP; do
    case "${!name}" in debug|info|warning|error|critical) ;; *) die "$name must be debug, info, warning, error or critical." ;; esac
done
[[ "$LOG_REQUESTS_LEVEL" =~ ^[0-3]$ ]] || die "LOG_REQUESTS_LEVEL must be 0, 1, 2 or 3."
case "$LOG_REQUESTS_FORMAT" in text|json) ;; *) die "LOG_REQUESTS_FORMAT must be text or json." ;; esac
[[ -d "$SGLANG_ROOT/python/sglang" ]] || die "Invalid SGLANG_ROOT: $SGLANG_ROOT"
[[ -f "$MOE_OFFLOAD_CONFIG" ]] || die "Missing offload JSON: $MOE_OFFLOAD_CONFIG"
command -v "$PYTHON_BIN" >/dev/null || die "Python executable not found: $PYTHON_BIN"

if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    ASCEND_RT_VISIBLE_DEVICES=0
    for ((i=1; i<WORLD_SIZE; i++)); do ASCEND_RT_VISIBLE_DEVICES+=",$i"; done
fi
IFS=, read -r -a visible_devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
[[ ${#visible_devices[@]} -eq "$WORLD_SIZE" ]] || die "Visible-device count must equal WORLD_SIZE."
[[ "$ASCEND_RT_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "Use comma-separated numeric device IDs."
for ((i=0; i<WORLD_SIZE; i++)); do
    for ((j=0; j<i; j++)); do
        [[ "${visible_devices[i]}" != "${visible_devices[j]}" ]] || die "Duplicate visible device ID."
    done
done
export ASCEND_RT_VISIBLE_DEVICES
export PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export SGLANG_LOG_MS=${SGLANG_LOG_MS:-1}

GENERATE_LOGGING_CONFIG=0
if [[ -n "${SGLANG_LOGGING_CONFIG_PATH:-}" ]]; then
    [[ -f "$SGLANG_LOGGING_CONFIG_PATH" ]] || die "Missing logging config: $SGLANG_LOGGING_CONFIG_PATH"
else
    export SGLANG_LOGGING_CONFIG_PATH="$LOG_DIR/logging.json"
    GENERATE_LOGGING_CONFIG=1
fi
if [[ "$ENABLE_TRACE" == 1 ]]; then
    export SGLANG_TRACE_ASYNC=${SGLANG_TRACE_ASYNC:-1}
fi
if [[ "$PROFILE_CAPTURE" == 1 ]]; then
    export SGLANG_TORCH_PROFILER_DIR=${SGLANG_TORCH_PROFILER_DIR:-$LOG_DIR/profiler}
fi

if [[ "$DRY_RUN" == 0 ]]; then
    command -v tee >/dev/null || die "tee is required for terminal and file logging."
    mkdir -p -- "$LOG_DIR"
    # Redirect the shell's descriptors, retaining exec, signals and server exit
    # status. The inference process is not wrapped in a shell pipeline.
    exec > >(tee -a "$LOG_DIR/server.log") 2>&1
fi

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
unset HCCL_IF_IP HCCL_SOCKET_FAMILY RANK_TABLE_FILE
unset SGLANG_HICACHE_HOST_MEM_BACKEND
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-300}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-300}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-2048}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export HCCL_INTRA_PCIE_ENABLE=${HCCL_INTRA_PCIE_ENABLE:-1}
export HCCL_INTRA_ROCE_ENABLE=${HCCL_INTRA_ROCE_ENABLE:-0}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-auto}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-lo}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT:-300}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export TASK_QUEUE_ENABLE=1
export STREAMS_PER_DEVICE=${STREAMS_PER_DEVICE:-32}
export INF_NAN_MODE_FORCE_DISABLE=1
export ENABLE_PROFILING=0
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
export ASCEND_USE_FIA=1
export SGLANG_NPU_USE_MLAPO=0

# These are correctness requirements of the current offload backend.
export DEEP_USE_MODE=default
export MOE_ENABLE_TOPK_NEG_ONE=1
export MOE_EXPERT_TOKEN_NUMS_TYPE=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=0
export SGLANG_ZBAL_LOCAL_MEM_SIZE=0
export SGLANG_ENABLE_SPEC_V2=0
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
export SGLANG_NPU_USE_MULTI_STREAM=0
export IS_DEEPSEEK_V4=1
MOE_EXPERT_FORMAT=$("$PYTHON_BIN" - "$MODEL_PATH" "${MOE_EXPERT_FORMAT:-auto}" <<'PYFORMAT'
import json
import sys
from pathlib import Path

requested = sys.argv[2]
if requested not in ("auto", "fp4", "w8a8"):
    raise SystemExit("MOE_EXPERT_FORMAT must be auto, fp4 or w8a8")
description = Path(sys.argv[1], "quant_model_description.json")
quant = json.loads(description.read_text()) if description.is_file() else None
if quant is not None and quant.get("model_quant_type") != "W8A8_DYNAMIC":
    raise SystemExit("This launcher supports ModelSlim W8A8_DYNAMIC only")
detected = "w8a8" if quant is not None else "fp4"
if requested != "auto" and requested != detected:
    raise SystemExit("MOE_EXPERT_FORMAT disagrees with the checkpoint quantization metadata")
print(detected)
PYFORMAT
)
if [[ "$MOE_EXPERT_FORMAT" == w8a8 ]]; then
    MODEL_QUANTIZATION=modelslim
    export SGLANG_DSV4_FP4_EXPERTS=False
else
    MODEL_QUANTIZATION=fp8
    export SGLANG_DSV4_FP4_EXPERTS=True
fi
export SGLANG_DSV4_FP4_DEQUANT=0
export USE_FUSED_HC_PRE_ASCENDC=1
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_FP8_WO_A_GEMM=False

NATIVE_BUILD_DIR=$("$PYTHON_BIN" - "$MOE_OFFLOAD_CONFIG" "$DRY_RUN" "$LOG_DIR" "$LOG_LEVEL" "$MOE_LOG_LEVEL" "$GENERATE_LOGGING_CONFIG" <<'PY'
import json
import math
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
dry_run = sys.argv[2] == "1"
log_dir = Path(sys.argv[3])


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit("Duplicate offload configuration key: " + key)
        result[key] = value
    return result


config = json.loads(source.read_text(), object_pairs_hook=unique)
if not isinstance(config, dict):
    raise SystemExit("Offload configuration must be a JSON object")
directory = config.get("native_extension_directory")
if not isinstance(directory, str) or not Path(directory).is_absolute():
    raise SystemExit("Offload JSON requires an absolute native_extension_directory")
interval = config.get("batch_log_interval", 1)
capacity = config.get("batch_log_capacity", 64)
if type(interval) is not int or interval < 0:
    raise SystemExit("batch_log_interval must be a nonnegative integer")
if type(capacity) is not int or capacity <= 0:
    raise SystemExit("batch_log_capacity must be a positive integer")
ratio = config.get("cache_ratio")
if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
    raise SystemExit("cache_ratio must be a finite number in [0, 1]")
profile = config.get("offline_stats_path")
model_id = config.get("offline_model_id")
if profile is not None and (not isinstance(profile, str) or not profile):
    raise SystemExit("offline_stats_path must be a nonempty path")
if model_id is not None and (not isinstance(model_id, str) or not model_id):
    raise SystemExit("offline_model_id must be a nonempty model identifier")
if isinstance(profile, str) and profile:
    profile = Path(profile)
    profile = profile if profile.is_absolute() else source.parent / profile
    config["offline_stats_path"] = str(profile.resolve())
    if ratio > 0 and not dry_run and not profile.is_file():
        raise SystemExit(
            "Missing offline expert profile: "
            + str(profile)
            + "; update MOE_OFFLOAD_CONFIG before building"
        )
    if ratio > 0 and not dry_run:
        statistics = json.loads(profile.read_text())
        if not isinstance(statistics, dict) or not isinstance(statistics.get("layers"), dict):
            raise SystemExit("Offline profile must contain a layers object")
        if model_id is not None and statistics.get("model_id") != model_id:
            raise SystemExit("Offline profile model_id must match offline_model_id")
config["batch_log_interval"] = interval
config["batch_log_capacity"] = capacity
print(
    f"MoE batch logging: interval={interval} capacity={capacity}; offline profile={config.get('offline_stats_path', 'N/A')}",
    file=sys.stderr,
)
if not dry_run:
    (log_dir / "offload-config.snapshot.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    if sys.argv[6] == "1":
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "runtime": {
                    "format": "[%(asctime)s.%(msecs)03d %(levelname)s pid=%(process)d %(name)s] %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                    "formatter": "runtime",
                }
            },
            "root": {"level": sys.argv[4].upper(), "handlers": ["console"]},
            "loggers": {
                "sglang.srt.offloading.moe": {
                    "level": sys.argv[5].upper(),
                    "propagate": True,
                },
                "httpx": {"level": "WARNING"},
                "httpcore": {"level": "WARNING"},
            },
        }
        (log_dir / "logging.json").write_text(
            json.dumps(logging_config, indent=2) + "\n"
        )
print(directory)
PY
)

# Include the attention-TP alignment bucket even when concurrency is lower.
# The graph runner applies its final backend-specific bucket filtering.
ATTN_TP_SIZE=$((WORLD_SIZE / ATTN_DP_SIZE))
GRAPH_MAX_BS=$(((MAX_RUNNING_REQUESTS + ATTN_TP_SIZE - 1) / ATTN_TP_SIZE * ATTN_TP_SIZE))
GRAPH_BS=()
for ((bs=1; bs<GRAPH_MAX_BS; bs*=2)); do GRAPH_BS+=("$bs"); done
GRAPH_BS+=("$GRAPH_MAX_BS")
BUILD_CMD=("$PYTHON_BIN" -m sglang.srt.offloading.moe.npu.native_acl --build-directory "$NATIVE_BUILD_DIR")
SERVER_CMD=(
    "$PYTHON_BIN" -m sglang.launch_server
    --model-path "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME"
    --host "$SERVER_HOST" --port "$SERVER_PORT"
    --tp-size "$WORLD_SIZE" --ep-size "$WORLD_SIZE" --dp-size 1
    --attn-dp-size "$ATTN_DP_SIZE" --moe-dp-size 1
    --device npu --attention-backend dsv4 --page-size 128 --trust-remote-code
    --load-format safetensors --quantization "$MODEL_QUANTIZATION" --kv-cache-dtype bfloat16
    --moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend ascend
    --disable-overlap-schedule --disable-shared-experts-fusion
    --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled
    --cuda-graph-bs-decode "${GRAPH_BS[@]}"
    --cuda-graph-max-bs-decode "$GRAPH_MAX_BS"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --max-prefill-tokens "$CHUNKED_PREFILL_SIZE" --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --context-length "$CONTEXT_LENGTH" --mem-fraction-static "$MEM_FRACTION_STATIC"
    --watchdog-timeout 9000 --enable-metrics --load-balance-method round_robin
    --moe-offload-config "$MOE_OFFLOAD_CONFIG"
    --log-level "$LOG_LEVEL" --log-level-http "$LOG_LEVEL_HTTP"
    --decode-log-interval "$DECODE_LOG_INTERVAL"
)
if [[ "$LOG_REQUESTS" == 1 ]]; then
    SERVER_CMD+=(--log-requests --log-requests-level "$LOG_REQUESTS_LEVEL" --log-requests-format "$LOG_REQUESTS_FORMAT" --log-requests-target stdout)
fi
if [[ "$ENABLE_TRACE" == 1 ]]; then
    SERVER_CMD+=(--enable-trace --trace-modules request --otlp-traces-endpoint "$OTLP_TRACES_ENDPOINT" --otlp-service-name "$OTLP_SERVICE_NAME")
fi
if [[ "$PROFILE_CAPTURE" == 1 ]]; then
    SERVER_CMD+=(--enable-profile-cuda-graph)
fi

printf 'WORLD_SIZE=%s TP=%s EP=%s ATTN_DP=%s ATTN_TP=%s DP=1 MOE_DP=1 DEVICES=%s\n' "$WORLD_SIZE" "$WORLD_SIZE" "$WORLD_SIZE" "$ATTN_DP_SIZE" "$((WORLD_SIZE / ATTN_DP_SIZE))" "$ASCEND_RT_VISIBLE_DEVICES"
printf 'Expert format: %s; quantization: %s\n' "$MOE_EXPERT_FORMAT" "$MODEL_QUANTIZATION"
printf 'Offload JSON: %s\nNative build: %s\n' "$MOE_OFFLOAD_CONFIG" "$NATIVE_BUILD_DIR"
printf 'Log directory: %s\nLogging config: %s\n' "$LOG_DIR" "$SGLANG_LOGGING_CONFIG_PATH"
printf 'Logging: root=%s MoE=%s requests=%s request_level=%s decode_interval=%s\n' "$LOG_LEVEL" "$MOE_LOG_LEVEL" "$LOG_REQUESTS" "$LOG_REQUESTS_LEVEL" "$DECODE_LOG_INTERVAL"
printf 'Request trace=%s endpoint=%s; graph capture profile=%s\n' "$ENABLE_TRACE" "$OTLP_TRACES_ENDPOINT" "$PROFILE_CAPTURE"
if [[ "$PROFILE_CAPTURE" == 1 ]]; then
    printf 'Capture profile directory: %s\n' "$SGLANG_TORCH_PROFILER_DIR"
fi
if [[ "$GENERATE_LOGGING_CONFIG" == 0 ]]; then
    printf 'Custom SGLANG_LOGGING_CONFIG_PATH controls logger levels.\n'
fi
if [[ "$BUILD_NATIVE" == 1 ]]; then printf '%q ' "${BUILD_CMD[@]}"; printf '\n'; fi
printf '%q ' "${SERVER_CMD[@]}"; printf '\n'
[[ "$DRY_RUN" == 0 ]] || exit 0

"$PYTHON_BIN" - "$MODEL_PATH" "$WORLD_SIZE" <<'PY'
import json
import sys
from pathlib import Path

model = Path(sys.argv[1])
config = json.loads((model / "config.json").read_text())
if config.get("model_type") != "deepseek_v4":
    raise SystemExit(
        "MODEL_PATH must be DeepSeek V4; GLM is unsupported"
    )
experts = config.get("n_routed_experts", 0)
if not isinstance(experts, int) or experts <= 0 or experts % int(sys.argv[2]):
    raise SystemExit("n_routed_experts must divide evenly across WORLD_SIZE")
if not any(model.glob("*.safetensors")):
    raise SystemExit("MODEL_PATH contains no local safetensors checkpoint")
PY

if [[ "$ENABLE_TRACE" == 1 ]]; then
    "$PYTHON_BIN" -c 'import opentelemetry.sdk.trace; import opentelemetry.exporter.otlp.proto.grpc.trace_exporter; import opentelemetry.exporter.otlp.proto.http.trace_exporter' \
        || die "ENABLE_TRACE=1 requires the OpenTelemetry SDK and OTLP gRPC/HTTP exporters in PYTHON_BIN."
fi
if [[ "$BUILD_NATIVE" == 1 ]]; then "${BUILD_CMD[@]}"; fi
exec "${SERVER_CMD[@]}"
