# cpu高性能
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# 绑核
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/cann-9.1.0/opp/vendors/customize/bin/set_env.bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
# pd传输, IP设置为p节点首节点
export ASCEND_MF_LOG_LEVEL=2
export ASCEND_MF_STORE_URL="tcp://61.47.19.68:24707"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200


# p节点IP
P_IP=('61.47.19.68' '61.47.19.69')
# D节点IP D节点首节点IP
D_IP=('61.47.19.70' '61.47.19.71')


MODEL_PATH=/home/weights/GLM-5.2-w8a8

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

# prefill
for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
        echo "${P_IP[$i]}"
        export HCCL_BUFFSIZE=1200
        export DEEPEP_NORMAL_LONG_SEQ_ROUND=72
        export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
        export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
        export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
        export TASK_QUEUE_ENABLE=2
        export ENABLE_PROFILING=0
        export HCCL_SOCKET_IFNAME=enp196s0f0
        export GLOO_SOCKET_IFNAME=enp196s0f0

        export SGLANG_PP_LAYER_PARTITION=18,20,24,16

        # P节点
        python -m sglang.launch_server --model-path ${MODEL_PATH}  --disaggregation-mode prefill --host ${P_IP[$i]} \
        --port 8000 --disaggregation-bootstrap-port 8998 --dist-init-addr 61.47.19.68:5000 --trust-remote-code --nnodes 2 --node-rank $i \
        --mem-fraction-static 0.72 --attention-backend ascend --device npu --quantization modelslim \
        --tp-size 8 \
        --pp-size 4 \
        --dp-size 1 \
        --attn-cp-size 8 --enable-prefill-cp --cp-strategy interleave \
        --enable-dp-attention \
        --enable-dp-lm-head \
        --moe-dense-tp 1 \
        --load-balance-method round_robin \
        --disaggregation-transfer-backend ascend --max-running-requests 64 \
        --served-model-name glm-5 --chunked-prefill-size 4096 --max-prefill-tokens 180000 --moe-a2a-backend deepep --deepep-mode normal \
        --disable-shared-experts-fusion --disable-cuda-graph --dtype bfloat16

        NODE_RANK=$i
        break
    fi
done

# decode
for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "${D_IP[$i]}"
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
        export DEEPEP_HCCL_BUFFSIZE=200
        # export ENABLE_PROFILING=1
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16
        export TASK_QUEUE_ENABLE=0
        # export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
        # export ENABLE_FUSED_MOE=1
        export HCCL_SOCKET_IFNAME=enp196s0f0
        export GLOO_SOCKET_IFNAME=enp196s0f0

        ###p不开双流 / p回退到native算子
        export SGLANG_NPU_USE_MULTI_STREAM=1

        python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode decode --host ${D_IP[$i]} \
        --port 8003 --trust-remote-code --dist-init-addr 61.47.19.70:5000 --nnodes 2 --node-rank $i --tp-size 32 --dp-size 32 --enable-dp-attention --ep-size 32 \
        --mem-fraction-static 0.97 --max-running-requests 32 --attention-backend ascend --device npu --quantization modelslim \
        --served-model-name glm-5 --moe-a2a-backend deepep --deepep-mode low_latency \
        --cuda-graph-bs-decode 1 2 3 --disaggregation-transfer-backend ascend --watchdog-timeout 9000 --context-length 180000 \
        --tokenizer-worker-num 4 --disable-shared-experts-fusion --dtype bfloat16  --load-balance-method round_robin

        NODE_RANK=$i
        break
    fi
done

exit 1
