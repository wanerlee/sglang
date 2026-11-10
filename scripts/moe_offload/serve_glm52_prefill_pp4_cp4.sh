#!/usr/bin/env bash
# A3 单机 16 die：PP4 × TP4，DP1 / CP4，stage 内 EP4，仅 Prefill 权重卸载。
# 启动前激活已安装 torch_npu / SGLang 依赖的 Python 环境。
# Usage: MODEL_PATH=/models/GLM-5.2-w8a8 bash serve_glm52_prefill_pp4_cp4.sh [--dry-run]
set -eo pipefail

# ==================== 必填 ====================
MODEL_PATH=${MODEL_PATH:-/home/weights/GLM-5.2-w8a8} # 本机 GLM-5.2 ModelSlim W8A8 权重目录

# ==================== 按容量调整 ====================
HOST_POOL_GIB=${HOST_POOL_GIB:-56} # 每 rank；16 rank 预留 1024 GiB，另留系统/加载内存
PREFILL_RESIDENT_RATIO=${PREFILL_RESIDENT_RATIO:-0} # P 每层本地 routed experts 的常驻比例
EXECUTION_SLOTS=${EXECUTION_SLOTS:-""} # P 每 rank 权重槽位上限，留空按 PP stage 计算
export SGLANG_PP_LAYER_PARTITION=${SGLANG_PP_LAYER_PARTITION:-18,20,24,16}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-180000}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-16384}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-$CHUNKED_PREFILL_SIZE}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-128}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.6}
P_MOE_A2A_BACKEND=${P_MOE_A2A_BACKEND:-ascend_ops}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

# ==================== 服务与单机通信 ====================
SERVER_HOST=${SERVER_HOST:-0.0.0.0}
SERVER_PORT=${SERVER_PORT:-30000}
P_IP=${P_IP:-127.0.0.1} # 单机内部通信默认 loopback；不需要自动识别 IP
COMM_IFACE=${COMM_IFACE:-lo}
DIST_PORT=${DIST_PORT:-5000}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
MF_STORE_PORT=${MF_STORE_PORT:-24707}

# ==================== 非标准安装位置才需要改 ====================
PYTHON_BIN=${PYTHON_BIN:-python3} # 默认使用当前激活环境；可填写 Python 可执行文件的绝对路径
CANN_ENV=${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
ATB_ENV=${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}
CUSTOMIZE_ENV=${CUSTOMIZE_ENV:-/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash}
CUSTOM_TRANSFORMER_ENV=${CUSTOM_TRANSFORMER_ENV:-/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash}
RUNTIME_DIR=${RUNTIME_DIR:-/tmp/sglang-glm52-prefill-pp4-cp4-runtime}
LOG_DIR=${LOG_DIR:-/tmp/sglang-glm52-prefill-pp4-cp4-$(date +%Y%m%d_%H%M%S)-$$}

fail() { echo "ERROR: $*" >&2; exit 1; }
usage() {
  cat <<EOF
Usage: MODEL_PATH=/models/GLM-5.2-w8a8 $0 [--dry-run]
  --dry-run : 只读模型 config.json、检查配置并打印命令，不加载 NPU 或创建运行文件。
固定拓扑：单机 PP4 × TP4 / DP1 / CP4 zigzag / stage 内 EP4；关闭 MTP。
P_MOE_A2A_BACKEND=deepep 可使用 DeepEP normal；默认 ascend_ops。
请求直发 P 的 /generate，设置 max_new_tokens=1、bootstrap_host=2.2.2.2、bootstrap_room=0。
不启动 D / Router；真实 PD KV 传输不在这次独立 P 测试范围内。
EOF
}
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; fail "Unknown argument: $arg" ;;
  esac
done
[[ ${ENABLE_MTP:-0} == 0 ]] || fail "This prefill-only test requires ENABLE_MTP=0"
[[ -f $MODEL_PATH/config.json ]] || fail "Set MODEL_PATH to a local GLM-5.2 ModelSlim W8A8 checkpoint"
[[ $RUNTIME_DIR == /* && $LOG_DIR == /* ]] || fail "RUNTIME_DIR and LOG_DIR must be absolute paths"
[[ -n $COMM_IFACE ]] || fail "COMM_IFACE must name a local communication interface"
case "$P_MOE_A2A_BACKEND" in
  ascend_ops)
    moe_args=(--moe-a2a-backend ascend_ops --deepep-mode auto
      --moe-ascend-ops-config '{"prefill":"alltoallv","decode":"low_latency"}')
    ;;
  deepep)
    moe_args=(--moe-a2a-backend deepep --deepep-mode normal
      --deepep-dispatcher-output-dtype int8)
    ;;
  *) fail "P_MOE_A2A_BACKEND must be ascend_ops or deepep" ;;
esac
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1

# Preflight uses only the Python standard library. Runtime storage checks use
# the actual quantization layout; the optional weight estimate below is W8 only.
"$PYTHON_BIN" - "$MODEL_PATH" "$HOST_POOL_GIB" "$EXECUTION_SLOTS" "$PREFILL_RESIDENT_RATIO" "$SGLANG_ROOT" \
  "$RUNTIME_DIR" "$LOG_DIR" "$DRY_RUN" "$SGLANG_PP_LAYER_PARTITION" \
  "$ASCEND_RT_VISIBLE_DEVICES" "$CONTEXT_LENGTH" "$CHUNKED_PREFILL_SIZE" \
  "$MAX_PREFILL_TOKENS" "$MAX_RUNNING_REQUESTS" "$MEM_FRACTION_STATIC" \
  "$SERVER_PORT" "$DIST_PORT" "$BOOTSTRAP_PORT" "$MF_STORE_PORT" "$P_IP" <<'PY_CONFIG'
import ipaddress
import json
import math
import sys
from itertools import accumulate
from pathlib import Path
from runpy import run_path

(model, host, slots, resident_ratio, root, runtime, logs, dry, partition, devices,
 context, chunk, prefill, requests, fraction, *ports_and_ip) = sys.argv[1:]
for port in ports_and_ip[:-1]:
    if not port.isdecimal() or not 1 <= int(port) <= 65535:
        raise ValueError(f"Invalid TCP port: {port}")
address = ipaddress.IPv4Address(ports_and_ip[-1])
if address.is_unspecified or address.is_multicast or str(address) == "255.255.255.255":
    raise ValueError("P_IP must be a local unicast IPv4 address (loopback is allowed)")
visible = devices.split(",")
if len(visible) != 16 or len(set(visible)) != 16 or not all(d.isdecimal() for d in visible):
    raise ValueError("Expose 16 distinct A3 NPU devices for PP4 x TP4")
for name, value in (("HOST_POOL_GIB", host), ("CONTEXT_LENGTH", context),
                    ("CHUNKED_PREFILL_SIZE", chunk), ("MAX_PREFILL_TOKENS", prefill),
                    ("MAX_RUNNING_REQUESTS", requests)):
    if not value.isdecimal() or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
if int(chunk) % 128:
    raise ValueError("CHUNKED_PREFILL_SIZE must be divisible by page size 128")
if not math.isfinite(float(fraction)) or not 0 < float(fraction) < 1:
    raise ValueError("MEM_FRACTION_STATIC must be in (0, 1)")
cfg = json.loads((Path(model) / "config.json").read_text())
if cfg.get("model_type") != "glm_moe_dsa" or "GlmMoeDsaForCausalLM" not in cfg.get("architectures", []):
    raise ValueError("MODEL_PATH must contain GLM-5.2 / GlmMoeDsaForCausalLM")
experts = cfg.get("n_routed_experts")
if type(experts) is not int or experts <= 0 or experts % 4:
    raise ValueError("n_routed_experts must be positive and divisible by stage EP4")
layers, first, frequency = (cfg.get(k) for k in ("num_hidden_layers", "first_k_dense_replace", "moe_layer_freq"))
if type(layers) is not int or layers <= 0 or type(first) is not int or first < 0 or type(frequency) is not int or frequency <= 0:
    raise ValueError("Invalid num_hidden_layers / first_k_dense_replace / moe_layer_freq")
parts = [int(x) for x in partition.split(",")]
if len(parts) != 4 or any(x <= 0 for x in parts) or sum(parts) != layers:
    raise ValueError(f"SGLANG_PP_LAYER_PARTITION must contain four positive lengths summing to {layers}")
bounds = [0, *accumulate(parts)]
counts = [sum(i >= first and i % frequency == 0 for i in range(start, end))
          for start, end in zip(bounds, bounds[1:])]
if not all(counts):
    raise ValueError("Each PP stage must contain at least one routed MoE layer")
local_experts = experts // 4
options = run_path(str(Path(root) / "python/sglang/srt/offloading/moe/npu/options.py"))
ratio = options["parse_prefill_resident_ratio"](float(resident_ratio))
if slots and (not slots.isdecimal() or int(slots) <= 0):
    raise ValueError("EXECUTION_SLOTS must be a positive integer")
slot_budget = int(slots) if slots else None
stage_layouts = [
    options["prefill_weight_slots"](local_experts, layers, ratio, slot_budget)
    for layers in counts
]
resident = stage_layouts[0][0]
output = {
    "mode": "prefill_only", "cache_ratio": 0,
    "host_pool_gib": int(host),
    "native_extension_directory": str(Path(runtime) / "native_acl"),
}
if ratio:
    output["prefill_resident_ratio"] = ratio
if slot_budget is not None:
    output["execution_slots"] = slot_budget
elif not resident:
    output["execution_slots"] = stage_layouts[0][1]
print(json.dumps(output, indent=2))
print(f"Stage routed layers: {counts}; experts/rank/layer: {experts // 4}")
if ratio:
    print(f"Prefill resident experts/rank/layer: {resident}/{local_experts}; "
          f"HBM slots/rank by stage: {[allocated for _, allocated in stage_layouts]}")
hidden, intermediate = cfg.get("hidden_size"), cfg.get("moe_intermediate_size")
if type(hidden) is int and hidden > 0 and type(intermediate) is int and intermediate > 0:
    rank_layer_bytes = 3 * hidden * intermediate * (experts // 4)
    estimate = [rank_layer_bytes * count / 1024**3 for count in counts]
    print(f"W8 weight-only Host GiB/rank by stage: {estimate}; metadata/alignment/loading need extra RAM")
    if max(estimate) >= int(host):
        raise ValueError("HOST_POOL_GIB leaves no room for the largest stage's W8 weights and metadata")
print(f"Host pool reservation: {host} GiB/rank, {16 * int(host)} GiB/node; allow extra RAM for loading and the OS")
if dry == "0":
    Path(logs).mkdir(parents=True, exist_ok=True)
    (Path(logs) / "offload.json").write_text(json.dumps(output, indent=2) + "\n")
PY_CONFIG

export SGLANG_TEST_FAKE_PREFILL_RADIX_CACHE=1

command=("$PYTHON_BIN" -u -m sglang.launch_server
  --model-path "$MODEL_PATH" --served-model-name glm52-prefill-offload
  --device npu --base-gpu-id 0 --attention-backend ascend --quantization modelslim
  --load-format safetensors --dtype bfloat16 --trust-remote-code
  --host "$SERVER_HOST" --port "$SERVER_PORT" --dist-init-addr "$P_IP:$DIST_PORT"
  --nnodes 1 --node-rank 0 --pp-size 4 --tp-size 4 --dp-size 1 --attn-dp-size 1
  --enable-prefill-cp --attn-cp-size 4 --cp-strategy zigzag
  --ep-size 4 --moe-dp-size 1 --moe-dense-tp-size 1
  "${moe_args[@]}"
  --disaggregation-mode prefill --disaggregation-transfer-backend ascend
  --disaggregation-bootstrap-port "$BOOTSTRAP_PORT"
  --disable-overlap-schedule --disable-shared-experts-fusion
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled
  --page-size 128 --kv-cache-dtype bfloat16
  --context-length "$CONTEXT_LENGTH" --mem-fraction-static "$MEM_FRACTION_STATIC"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --moe-offload-config "$LOG_DIR/offload.json" --watchdog-timeout 600)

export PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_HOST_IP="$P_IP"
export ASCEND_MF_STORE_URL=${ASCEND_MF_STORE_URL:-tcp://$P_IP:$MF_STORE_PORT}
export HCCL_SOCKET_IFNAME="$COMM_IFACE" GLOO_SOCKET_IFNAME="$COMM_IFACE"
printf 'PP partition: %s\nKV store: %s\nRuntime: %s\nLogs: %s\nCommand:' \
  "$SGLANG_PP_LAYER_PARTITION" "$ASCEND_MF_STORE_URL" "$RUNTIME_DIR" "$LOG_DIR"
printf ' %q' "${command[@]}"
printf '\n'
if [[ $DRY_RUN == 1 ]]; then exit 0; fi

mkdir -p "$RUNTIME_DIR/python" "$RUNTIME_DIR/native_acl" "$LOG_DIR"
exec > >(tee "$LOG_DIR/server.log") 2>&1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ASCEND_LAUNCH_BLOCKING
for env_file in "$CANN_ENV" "$ATB_ENV" "$CUSTOMIZE_ENV" "$CUSTOM_TRANSFORMER_ENV"; do
  [[ -f $env_file ]] || fail "Missing Ascend environment script: $env_file"
  source "$env_file"
done

# Reuse the existing isolated A3 offload extension build. After CANN / torch_npu
# or MemFabric upgrades, choose a fresh RUNTIME_DIR.
"$PYTHON_BIN" - "$RUNTIME_DIR/python" <<'PY_BUILD'
import importlib.util
import shutil
import sys
from pathlib import Path

source = Path(importlib.util.find_spec("memfabric_hybrid").origin).parent
target = Path(sys.argv[1]) / "memfabric_hybrid"
if not target.exists():
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
PY_BUILD
export PYTHONPATH="$SGLANG_ROOT/python:$RUNTIME_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$RUNTIME_DIR/python/memfabric_hybrid/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_EXTENSIONS_DIR="$RUNTIME_DIR/torch_extensions"
export MAX_JOBS=${MAX_JOBS:-4}
"$PYTHON_BIN" - "$RUNTIME_DIR/python/memfabric_hybrid/lib" <<'PY_EXT'
import sys
from memfabric_hybrid._acc_offload_build import ensure_acc_offload_ext

assert ensure_acc_offload_ext("A3", install_dir=sys.argv[1])
PY_EXT
"$PYTHON_BIN" -m sglang.srt.offloading.moe.npu.native_acl --build-directory "$RUNTIME_DIR/native_acl"

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True SGLANG_SET_CPU_AFFINITY=1
export HCCL_BUFFSIZE=1024 HCCL_OP_EXPANSION_MODE=AIV
if [[ $P_MOE_A2A_BACKEND == deepep ]]; then
  export DEEPEP_HCCL_BUFFSIZE=${DEEPEP_HCCL_BUFFSIZE:-$HCCL_BUFFSIZE}
  export DEEPEP_NORMAL_LONG_SEQ_ROUND=${DEEPEP_NORMAL_LONG_SEQ_ROUND:-72}
  export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=${DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS:-1024}
  export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=${DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ:-1}
  export MOE_EXPERT_TOKEN_NUMS_TYPE=1 DEEP_NORMAL_MODE_USE_INT8_QUANT=1
fi
export HCCL_CONNECT_TIMEOUT=300 HCCL_EXEC_TIMEOUT=300 ACL_DEVICE_SYNC_TIMEOUT=300
export STREAMS_PER_DEVICE=32 INF_NAN_MODE_FORCE_DISABLE=1 TASK_QUEUE_ENABLE=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
exec "${command[@]}"
