#!/usr/bin/env bash
# GLM-5.2 ModelSlim W8A8_DYNAMIC / W4A8_DYNAMIC，A3 四 die。
# 代码已接入 offload；此启动配置仍需目标 NPU 数值、Graph 和性能验收。
set -eo pipefail

MODEL_PATH=${MODEL_PATH:-/home/weights/GLM-5.2-w8a8}
# export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
SERVER_PORT=${SERVER_PORT:-30000}
DIST_PORT=${DIST_PORT:-29500}
CACHE_RATIO=${CACHE_RATIO:-0.20}
ENABLE_MTP=${ENABLE_MTP:-0}
# GLM 的仓库部署配置使用 BF16 draft；主模型 W4/W8 不决定 draft 的量化。
MTP_DRAFT_QUANTIZATION=${MTP_DRAFT_QUANTIZATION:-unquant}
# 每 rank 的 Host 池预算；W4 per-channel 可显式设 96，per-group 另核算。
# 不根据目录名猜量化位数，loader 会检查完整 schema 的实际字节需求。
HOST_POOL_GIB=${HOST_POOL_GIB:-64}
RUNTIME_DIR=${RUNTIME_DIR:-/home/y00951466/moe-offload/tmp/sglang-glm52-offload-runtime}
LOG_DIR=${LOG_DIR:-/home/y00951466/moe-offload/tmp/sglang-glm52-offload-$(date +%Y%m%d_%H%M%S)-$$}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
OFFLINE_STATS_PATH=${OFFLINE_STATS_PATH:-$SCRIPT_DIR/initial_profile_glm52.json}

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "请设置 MODEL_PATH 为本地 GLM-5.2 ModelSlim checkpoint 目录：$MODEL_PATH" >&2
  exit 1
fi
if [[ "$ENABLE_MTP" != 0 && "$ENABLE_MTP" != 1 ]]; then
  echo "ENABLE_MTP 只接受 0 或 1" >&2
  exit 1
fi
mkdir -p "$RUNTIME_DIR/python" "$RUNTIME_DIR/native_acl" "$LOG_DIR"
exec > >(tee "$LOG_DIR/server.log") 2>&1

extra=()
if [[ "$ENABLE_MTP" == 1 ]]; then
  # 在分配主模型及 Host 池前检查额外 MTP 层，避免缺权重或量化误解释。
  python3 "$SCRIPT_DIR/glm52_mtp_config.py" "$MODEL_PATH" \
    --draft-quantization "$MTP_DRAFT_QUANTIZATION"
  extra+=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 2 --speculative-eagle-topk 1
    --speculative-num-draft-tokens 3
    --speculative-draft-model-path "$MODEL_PATH"
    --speculative-draft-model-quantization "$MTP_DRAFT_QUANTIZATION"
    --speculative-draft-attention-backend ascend
  )
fi

# 独立配置与 profile，不借用 DeepSeek V4 的专家访问统计。
python3 - "$SCRIPT_DIR" "$RUNTIME_DIR" "$LOG_DIR" "$CACHE_RATIO" "$HOST_POOL_GIB" "$OFFLINE_STATS_PATH" <<'PY_CONFIG'
import json
import math
import sys
from pathlib import Path

scripts, runtime, logs = (Path(p).resolve() for p in sys.argv[1:4])
ratio, host_gib = float(sys.argv[4]), int(sys.argv[5])
if not math.isfinite(ratio) or not 0 < ratio <= 1:
    raise ValueError('CACHE_RATIO must be finite and in (0, 1]')
if host_gib <= 0:
    raise ValueError('HOST_POOL_GIB must be a positive integer per rank')
profile = Path(sys.argv[6]).resolve(strict=True)
if not profile.is_file():
    raise ValueError('OFFLINE_STATS_PATH must name a profile JSON file')
config = json.loads((scripts / 'config.json').read_text())
config.update(
    cache_ratio=ratio,
    host_pool_gib=host_gib,
    offline_stats_path=str(profile),
    offline_model_id='GLM-5.2',
    native_extension_directory=str(runtime / 'native_acl'),
)
target = logs / 'offload.json'
target.write_text(json.dumps(config, indent=2) + '\n')
print(f'GLM-5.2 offload config: {target}')
print(f'Expert profile: {profile}')
print(f'Host pool: {host_gib} GiB/rank, {4 * host_gib} GiB across four ranks')
print(f'Server log: {logs / "server.log"}')
PY_CONFIG

# 昇腾环境；安装位置不同时修改这四行。
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

# 构建本地搬运扩展；更换 CANN / torch_npu / MemFabric 后更换 RUNTIME_DIR。
python3 - "$RUNTIME_DIR/python" <<'PY_BUILD'
import importlib.util
import shutil
import sys
from pathlib import Path

source = Path(importlib.util.find_spec('memfabric_hybrid').origin).parent
target = Path(sys.argv[1]) / 'memfabric_hybrid'
if not target.exists():
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__'))
PY_BUILD
export PYTHONPATH="$SGLANG_ROOT/python:$RUNTIME_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$RUNTIME_DIR/python/memfabric_hybrid/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAX_JOBS=${MAX_JOBS:-4}
python3 - "$RUNTIME_DIR/python/memfabric_hybrid/lib" <<'PY_EXT'
import sys
from memfabric_hybrid._acc_offload_build import ensure_acc_offload_ext

assert ensure_acc_offload_ext('A3', install_dir=sys.argv[1])
PY_EXT
python3 -m sglang.srt.offloading.moe.npu.native_acl --build-directory "$RUNTIME_DIR/native_acl"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ASCEND_LAUNCH_BLOCKING
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_SET_CPU_AFFINITY=1
export HCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo HCCL_OP_EXPANSION_MODE=AIV
export HCCL_CONNECT_TIMEOUT=300 HCCL_EXEC_TIMEOUT=300 ACL_DEVICE_SYNC_TIMEOUT=300
export STREAMS_PER_DEVICE=32 INF_NAN_MODE_FORCE_DISABLE=1 TASK_QUEUE_ENABLE=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
export HCCL_BUFFSIZE=1024

# ModelSlim 从 checkpoint 自识别 W8A8_DYNAMIC / W4A8_DYNAMIC 及 group_size。
# Ascend attention 的仓库默认 page size 为 128；MTP 使用同 checkpoint 的 NextN。
exec python3 -u -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name glm52-ascend-offload \
  --device npu --base-gpu-id 0 --host 127.0.0.1 --port "$SERVER_PORT" \
  --dist-init-addr "127.0.0.1:$DIST_PORT" --trust-remote-code \
  --attention-backend ascend --page-size 128 \
  --tp-size 16 --attn-dp-size 1 --ep-size 16 \
  --moe-a2a-backend ascend_ops --deepep-mode auto \
  --moe-ascend-ops-config '{"prefill":"alltoallv","decode":"low_latency"}' \
  --load-format safetensors --quantization modelslim --kv-cache-dtype bfloat16 \
  --enable-dp-lm-head --disable-shared-experts-fusion --disable-overlap-schedule \
  --mem-fraction-static 0.60 --context-length 256000 --max-total-tokens 256000 \
  --chunked-prefill-size 65536 --max-prefill-tokens 524288 \
  --max-running-requests 128 --prefill-max-requests 128 \
  --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled \
  --cuda-graph-bs-decode 16 --cuda-graph-max-bs-decode 16 \
  --moe-offload-config "$LOG_DIR/offload.json" "${extra[@]}" \
  --skip-server-warmup --watchdog-timeout 600 --log-level info
