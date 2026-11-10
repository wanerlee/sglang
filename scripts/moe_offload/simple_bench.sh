#!/usr/bin/env bash
set -euo pipefail

# ==================== 主要配置 ====================
HOST="127.0.0.1"
PORT=30000
# PORT=6688
KV_CACHE_HIT_RATE=0.99                 # 0~1；0.90=90%，1.0=原参考的全共享前缀
# MODEL="/home/weights/DeepSeek-V4-Flash-0731-w8a8"
MODEL="/home/weights/GLM-5.2-w8a8"
CONCURRENCY=1                      # 正式压测的客户端并发上限

# ==================== 测试规模 ====================
INPUT_LEN=128000                       # 目标总输入长度：共享前缀 + 独立问题
OUTPUT_LEN=1
NUM_REQUESTS=1                       # 正式压测总请求数，与并发上限不同
CACHE_REQUESTS=1                       # 创建缓存阶段的请求数和并发上限

# 每次运行使用新种子，减少复用上轮问题缓存的干扰；两阶段共用它。
SEED=$(python -c 'import secrets; print(secrets.randbits(32))')

# 计算目标长度，并检查参数。
LENGTHS="$(python - "$INPUT_LEN" "$KV_CACHE_HIT_RATE" \
    "$OUTPUT_LEN" "$NUM_REQUESTS" "$CONCURRENCY" \
    "$CACHE_REQUESTS" "$PORT" <<'PY'
import sys
from decimal import Decimal, InvalidOperation

try:
    input_len = int(sys.argv[1])
    hit_rate = Decimal(sys.argv[2])
    values = [int(x) for x in sys.argv[3:]]

    if input_len <= 0 or any(x <= 0 for x in values):
        raise ValueError("输入/输出长度、请求数、并发量和端口必须是正整数")

    if not hit_rate.is_finite() or not Decimal(0) <= hit_rate <= Decimal(1):
        raise ValueError("KV_CACHE_HIT_RATE 必须在 0~1 之间，例如 0.90")

    if values[-1] > 65535:
        raise ValueError("PORT 必须在 1~65535 之间")

except (ValueError, InvalidOperation) as exc:
    sys.exit(f"配置错误：{exc}")

prefix_len = int(Decimal(input_len) * hit_rate)
print(prefix_len, input_len - prefix_len)
PY
)"
read -r PREFIX_LEN QUESTION_LEN <<< "$LENGTHS"

COMMON_ARGS=(
    --backend sglang
    --host "$HOST"
    --port "$PORT"
    --model "$MODEL"
    --dataset-name generated-shared-prefix
    --gsp-num-groups 1
    --gsp-system-prompt-len "$PREFIX_LEN"
    --random-range-ratio 1.0
    --warmup-requests 0
    --request-rate inf
    --seed "$SEED"
    --extra-request-body '{"bootstrap_host":"2.2.2.2","bootstrap_room":0}'
    --cache-report
)

cat <<EOF
==================== 测试配置 ====================
服务地址：        http://${HOST}:${PORT}
模型：            ${MODEL}
目标 KV 命中比例：${KV_CACHE_HIT_RATE}
目标输入/输出：   ${INPUT_LEN}/${OUTPUT_LEN}
共享前缀长度：    ${PREFIX_LEN}
独立问题长度：    ${QUESTION_LEN}
随机种子：        ${SEED}
压测请求总数：    ${NUM_REQUESTS}
客户端并发上限：  ${CONCURRENCY}
==================================================
EOF

if (( CONCURRENCY > NUM_REQUESTS )); then
    echo "注意：只有 ${NUM_REQUESTS} 个压测请求，无法形成 ${CONCURRENCY} 个同时在途请求。"
fi

# ==================== 1. 创建缓存 ====================
# 只预热共享前缀，不预热正式压测中的独立问题。
if (( PREFIX_LEN > 0 )); then
    echo "[1/2] 创建共享前缀缓存"

    python -m sglang.bench_serving \
        "${COMMON_ARGS[@]}" \
        --gsp-prompts-per-group "$CACHE_REQUESTS" \
        --gsp-question-len 0 \
        --gsp-output-len 1 \
        --max-concurrency "$CACHE_REQUESTS"
else
    echo "[1/2] 目标共享前缀长度为 0，跳过创建缓存"
fi

# ==================== 2. 正式压测 ====================
# 不使用 --flush-cache，否则会清除刚刚创建的服务端缓存。
echo "[2/2] 开始正式压测"

# curl -X POST http://127.0.0.1:30000/start_profile -H "Content-Type: application/json"   -d '{
#     "num_steps": 4,
#     "activities": ["CPU","GPU"],
#     "profile_by_stage":true,
#     "profile_stages": ["prefill"],
#     "with_stack":false,
#     "record_shapes":false
#   }'

# curl --noproxy '*' -X POST \
#   http://61.47.19.68:8000/start_profile \
#   -H "Content-Type: application/json" \
#   -d '{
#     "output_dir": "/home/y00951466/pd_test/sglang/scripts/moe_offload/tmp/sglang_profile/prefill-1402",
#     "num_steps": 10,
#     "profile_by_stage": true,
#     "activities": ["CPU", "GPU"],
#     "with_stack": false,
#     "record_shapes": false
#   }'

python -m sglang.bench_serving \
    "${COMMON_ARGS[@]}" \
    --gsp-prompts-per-group "$NUM_REQUESTS" \
    --gsp-question-len "$QUESTION_LEN" \
    --gsp-output-len "$OUTPUT_LEN" \
    --max-concurrency "$CONCURRENCY"