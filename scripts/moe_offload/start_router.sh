SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://61.47.19.68:8000 8998 \
    --decode http://61.47.19.70:8003 \
    --host 127.0.0.1 \
    --port 6688