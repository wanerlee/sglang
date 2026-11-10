# 单机 A3，仅 Prefill 卸载测试

`serve_glm52_prefill_pp4_cp4.sh` 使用 16 die，固定为 PP4 × TP4、Attention DP1 / CP4 zigzag、stage 内 EP4。只启动 P 服务，关闭 MTP，测试每请求输出 1 个 token。脚本沿用 Ascend KV manager，请求使用已有 fake sender 跳过真实 D 对端；不需要 D / Router。

先激活安装了 SGLang 依赖、torch_npu 和 MemFabric 的 Python 环境，设置本机 GLM-5.2 ModelSlim **W8A8** 权重目录：

```bash
export MODEL_PATH=/models/GLM-5.2-w8a8
bash scripts/moe_offload/serve_glm52_prefill_pp4_cp4.sh --dry-run
bash scripts/moe_offload/serve_glm52_prefill_pp4_cp4.sh
```

`--dry-run` 只读取模型 `config.json` 并检查、打印配置；不需要 NPU 依赖，不创建运行文件。正常启动会在 `LOG_DIR` 生成 `offload.json`，使用 `mode=prefill_only`、`cache_ratio=0`、两个完整本地专家层的 bank；256 专家 / EP4 时，默认 `execution_slots=128`。实际格式和容量继续由运行时代码检查。

在安装了与服务器 CANN / torch_npu 匹配的 DeepEP-Ascend 环境中，可以切换 P 的 MoE 通信后端。

```bash
P_MOE_A2A_BACKEND=deepep bash scripts/moe_offload/serve_glm52_prefill_pp4_cp4.sh --dry-run
P_MOE_A2A_BACKEND=deepep bash scripts/moe_offload/serve_glm52_prefill_pp4_cp4.sh
```

默认使用 `ascend_ops`。选择 `deepep` 时，启动参数为 `--moe-a2a-backend deepep --deepep-mode normal --deepep-dispatcher-output-dtype int8`，通信使用 stage 内 EP group。适配范围为 GLM-5.2 ModelSlim W8A8 的 eager `prefill_only`，要求 `SGLANG_ZBAL_LOCAL_MEM_SIZE=0` 和逐专家计数 `MOE_EXPERT_TOKEN_NUMS_TYPE=1`。权重依旧通过 ACL 双 bank 预取，计算按 W13 / W2 就绪事件读权重，combine 提交并接回计算流后才复用 bank。

P 支持用 `PREFILL_RESIDENT_RATIO` 配置层内常驻比例，默认 0，取值范围为 [0, 1)。每层常驻前 floor(本地专家数 × 比例) 个专家，启动时用 ACL 搬入固定 HBM 槽。每个 chunk 仅预取其余专家，仍按 W13、W2 分开提交。此选项用于 GLM-5.2 ModelSlim W8A8、EP4 的 `prefill_only`，支持 Ascend Ops alltoallv 和 DeepEP normal。

~~~bash
PREFILL_RESIDENT_RATIO=0.25 EXECUTION_SLOTS=480 P_MOE_A2A_BACKEND=deepep \
  bash scripts/moe_offload/serve_glm52_prefill_pp4_cp4.sh --dry-run
~~~

去掉 `--dry-run` 后启动。默认 PP 分区下，比例 0.25 会在每张 NPU 每层常驻 16 个专家，并为 48 个冷专家保留双 bank。PP0 至 PP3 分别需要 336、416、480、352 个权重槽。启用常驻时，`EXECUTION_SLOTS` 是每张 NPU 的总权重槽位上限。留空由各 stage 计算最小容量。显存占用会在 KV pool 定容前计入，Host 权重池仍保留完整权重。

P 的 DeepEP 分轮参数初值沿用基线的 `72` 轮、每 rank 每轮 `1024` token 和 long-seq combine。可用 `DEEPEP_NORMAL_LONG_SEQ_ROUND`、`DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS`、`DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ` 和 `DEEPEP_HCCL_BUFFSIZE` 覆盖，默认通信 buffer 配置为 1024 MB。目标设备上的 16K chunk 容量、数值和性能需实测。[DeepEP-Ascend 参数说明](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md#environment-variables)

按需设置：

| 环境变量 | 默认值与用途 |
| --- | --- |
| `HOST_POOL_GIB` | 每 rank 64 GiB；16 rank 的物理 Host pool 合计 **1024 GiB**，系统和 checkpoint 加载还需额外内存 |
| `SGLANG_PP_LAYER_PARTITION` | `18,20,24,16`；须为四个正数，和等于模型层数，各 stage 至少有一个 routed MoE 层 |
| `CHUNKED_PREFILL_SIZE` / `MAX_PREFILL_TOKENS` | `8192` / 与 chunk 相同；按目标输入长度及 HBM 容量调整 |
| `CONTEXT_LENGTH` / `MAX_RUNNING_REQUESTS` | `16384` / `8` |
| `MEM_FRACTION_STATIC` | `0.8` |
| `SERVER_PORT` | P HTTP 默认 `8000`，默认监听 `0.0.0.0` |
| `P_IP` / `COMM_IFACE` | 单机内部通信默认 `127.0.0.1` / `lo`，不用 `ip`、`ifconfig` 或 `hostname` 工具 |
| `PYTHON_BIN` | 当前环境的 `python3`；仅需要固定解释器时填写绝对路径，已有 `PYTHONPATH` 会保留 |
| `RUNTIME_DIR` / `LOG_DIR` | `/tmp` 下的扩展缓存 / 日志目录；升级 CANN、torch_npu、MemFabric 后换新的 `RUNTIME_DIR` |

CANN 等安装位置非标准时，设置脚本顶部的四个 `*_ENV` 路径。更小的 `HOST_POOL_GIB` 必须能容纳本 stage 的所有本地专家及量化元数据；脚本会先做 W8 权重本体估算，不能把这项估算当成完整内存预算。

服务 ready 后，请求直发 P 的 `/generate`：

```bash
curl -sS http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "请解释流水线并行、上下文并行和专家并行如何协同完成长文本的预填充计算。",
    "sampling_params": {"temperature": 0, "max_new_tokens": 1, "ignore_eos": true},
    "bootstrap_host": "2.2.2.2",
    "bootstrap_room": 0,
    "stream": false
  }'
```

其他机器发请求时，将 URL 中 `127.0.0.1` 换成 P 机器可达的实际 IP。`bootstrap_host=2.2.2.2` 是源码中的 **fake 测试标记**，保持这个值，不换成真实 IP。仅设置输出长度为 1 不能自动绕过 PD 握手。

独立 P 启动脚本会显式设置 `SGLANG_TEST_FAKE_PREFILL_RADIX_CACHE=1`，以便测试跨请求 prefix KV cache；脚本内的设置会覆盖启动前同名环境变量。运行时的开关默认值仍是关闭，其他启动方式需要在启动服务前自行设置为 `1`。

开关仅影响 Prefill 模式下使用 `bootstrap_host=2.2.2.2` 的请求。打开后，此类请求计算出的 P 侧 KV 可以写入本地 radix cache；fake sender 仍然不会发送 KV，也不需要 D。请求仍须携带 `bootstrap_room`。服务启动时的 fake 预热请求也受此开关影响，因此它适用于隔离的基准测试进程。

测试命中时，等待预热请求**成功完成**后，再发相同 token 前缀的请求；通过响应 `meta_info.cached_tokens` 或服务日志 `#cached-token` 观察实际命中。`--page-size 128` 会按页对齐缓存，实际命中数不一定恰好等于目标前缀长度。普通 Prefill / fake Decode 默认行为，以及 embedding override 请求的缓存保护保持不变。

`scripts/moe_offload/simple_bench.sh` 的 `COMMON_ARGS` 已包含以下两项，可用于直连独立 P；使用前将地址、端口和输入长度改成当前 P 服务配置：

```bash
--extra-request-body '{"bootstrap_host":"2.2.2.2","bootstrap_room":0}'
--cache-report
```

脚本中的 `KV_CACHE_HIT_RATE` 是生成共享前缀的目标比例。真正的命中率由服务返回的 cached token 数决定；请求失败、KV 容量不足或生成的前缀不同都会影响结果。使用真实 PD Router 测试时须从脚本移除 fake bootstrap 字段。`simple_bench.sh` 内的 `start_profile` 使用独立的固定地址，执行该步骤前也需改成实际 P 服务地址或跳过它。

可先发送两个共用 512 个 token 前缀的请求，检查第二个请求的缓存计数（将 URL 换成实际 P 服务地址）：

```bash
python3 - <<'PY_SMOKE'
import json
import random
import urllib.request

url = "http://127.0.0.1:8000/generate"
rng = random.SystemRandom()
prefix = [rng.randrange(100, 3000) for _ in range(512)]
for index in range(2):
    body = {
        "input_ids": prefix + [10 + index] * 128,
        "sampling_params": {"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
        "bootstrap_host": "2.2.2.2", "bootstrap_room": 0, "stream": False,
    }
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    meta = result.get("meta_info", {})
    if result.get("error") or (meta.get("finish_reason") or {}).get("type") == "abort":
        raise RuntimeError(result)
    print(f"request={index + 1} prompt_tokens={meta.get('prompt_tokens')} cached_tokens={meta.get('cached_tokens')}")
PY_SMOKE
```

在缓存开启、没有驱逐且 KV 充足的情况下，第二个请求应能看到非零命中；这个测试使用 page 对齐的共享前缀。若依旧为零，请核对实际服务是否使用当前代码、是否通过该脚本启动，以及请求是否成功完成。

第一个 token 在 prefill 结束后由 logits 采样产生，不需要 decode forward。长输入仍可包含多个 prefill chunk。极短输入可能回退到不切 CP 的路径；验收 CP4 时用目标长度输入，并检查 forward / CP 元数据。与同拓扑常驻结果比较 token ID / logits，确认只输出 1 token、没有本请求的 decode forward，且连续请求结束后 KV / metadata / bank 正常回收。

本配置需要目标 NPU 上的数值、PP/CP 通信和性能验收。Fake sender 不传输真实 KV，独立 P 测试通过不代表真实 P→D 传输、D 多 token 生成或 Router 联调已经通过。

## 真实 PD：单机 P 卸载 + 双机 D

新脚本 [serve_glm52_pd_pp4_cp4.sh](serve_glm52_pd_pp4_cp4.sh) 沿用本页的 P 拓扑和权重卸载方式，并接入双机 Decode：

| 节点 | 部署参数 | 专家权重 |
| --- | --- | --- |
| P，一台 A3 / 16 die | PP4 × TP4、DP1、attention DP1 / CP4 zigzag、stage 内 EP4；默认 Ascend Ops alltoallv，可选 DeepEP normal | routed 专家卸载到 Host，双 bank 逐层预取 |
| D0 + D1，两台 A3 / 32 die | TP32、attention DP32、EP32；DeepEP low_latency；Decode graph 档位为 1、2、3 | 常驻设备，不启用 MoE offload |

本组合固定关闭 MTP：不能设置 `ENABLE_MTP=1`。请先用 `max_new_tokens=1` 验证完整 PD 链路；D 的多 token 生成、P→D 的真实 KV 传输以及并发仍需实机验收。脚本不会启动 fake sender 测试模式。

D 的启动参数和进程环境沿用 `best_pd_4rank_baseline.sh`：`DEEPEP_HCCL_BUFFSIZE=200`、每 rank 的 DeepEP dispatch 容量参数为 16，bootstrap 与传输等待超时均为 1200 秒。D 不主动设置通用的 `HCCL_BUFFSIZE`，继承启动环境中的值。默认使用 CANN 9.1.0 的 customize 环境脚本，并将 customize 的 op API 库目录加入 `LD_LIBRARY_PATH`；非默认安装可分别设置 `D_CUSTOMIZE_ENV` 和 `D_CUSTOMIZE_OP_API_LIB`。

D 正常启动时还会按该基线将本机 CPU governor 设为 `performance`，并设置 `vm.swappiness=0`、`kernel.numa_balancing=0`、`kernel.sched_migration_cost_ns=50000`。这些是**主机级**设置，可能需要管理员权限；不希望调整时设置 `D_APPLY_HOST_TUNING=0`。`--dry-run` 不执行这些操作。

启动前，在三台机器上的同一份脚本顶部填写或通过环境变量指定：

- `MODEL_PATH`：每台本地的 GLM-5.2 ModelSlim **W8A8** 权重目录；三台 checkpoint 内容一致。启动前激活安装了 SGLang 和 NPU 依赖的 Python 环境；非默认环境可设置 `PYTHON_BIN`，脚本自动将当前仓库的 `python` 目录加入 `PYTHONPATH`。
- `P_IP`、`D0_IP`、`D1_IP`：三台机器实际绑定、互相可达的通信 IPv4，三份配置保持一致。脚本通过 `pd_node.py` 自动匹配本机 IP 和网卡，不依赖 `ip`、`ifconfig`。容器内须能看到这些 IP。
- P 的 `HOST_POOL_GIB` 默认为每 rank 56 GiB，16 rank 合计 896 GiB；系统、模型加载和其他进程还需要额外内存。启动前会依据 checkpoint 的尺寸做 W8 权重本体估算，运行时还会按实际格式检查。`SGLANG_PP_LAYER_PARTITION` 默认 `18,20,24,16`，需与模型层数相符；`EXECUTION_SLOTS` 留空时按 P 的常驻比例和各 stage 层数分配权重槽。
- 默认 P 的 `CHUNKED_PREFILL_SIZE=8192`、`MAX_PREFILL_TOKENS=8192`、`P_MEM_FRACTION_STATIC=0.6`、`P_MAX_RUNNING_REQUESTS=128`；D 的 `D_MEM_FRACTION_STATIC=0.97`、`D_MAX_RUNNING_REQUESTS=32`。可按容量调整，P/D 的 `CONTEXT_LENGTH` 和模型内容须一致。D 显存比例较高，须在目标机器验证 graph 捕获、KV 传输和实际请求的显存余量。实际 P chunk 会影响搬运次数、性能和峰值内存。
- 若当前环境已设置通用的 `MEM_FRACTION_STATIC` 或 `MAX_RUNNING_REQUESTS`，它们会优先覆盖当前机器对应角色的默认值；需要基线默认值时请先检查这些环境变量。
- D 默认的 graph 档位可通过 `D_CUDA_GRAPH_BS="1 2 3"` 调整；原有的 `D_CUDA_GRAPH_MAX_BS` 覆盖方式仍可使用，设置时优先于档位列表。

三台机器分别执行相同的命令；D0 和 D1 应同时启动，组成一个 D 实例：

~~~bash
bash scripts/moe_offload/serve_glm52_pd_pp4_cp4.sh --dry-run
bash scripts/moe_offload/serve_glm52_pd_pp4_cp4.sh
~~~

也可以在 P 的启动命令前加 `P_MOE_A2A_BACKEND=deepep`，选项只作用于 P。

`--dry-run` 只检查本机配置并打印启动命令；不加载模型、构建扩展或创建运行文件。正常启动时，P 会在 `LOG_DIR` 下生成 `offload.json` 并构建卸载扩展，D 不会生成该文件；日志均为各自 `LOG_DIR/server.log`。`RUNTIME_DIR` 和 `LOG_DIR` 可用绝对路径覆盖。Ascend 软件安装在非默认路径时设置脚本顶部的环境文件路径。

P 和 D ready 后，在可连接 `P_IP` 和 `D0_IP` 的机器上另开终端，单独运行一次 Router：

~~~bash
bash scripts/moe_offload/serve_glm52_pd_pp4_cp4.sh router
~~~

Router 只登记 P 和 D0，D1 不单独登记；客户端向运行 Router 的机器发送请求，例如 `http://<Router-IP>:6688/generate`。默认端口为 P HTTP 8000、P bootstrap 8998、D HTTP 8003、P/D 各自的 rendezvous 5000、KV store 24707、Router 6688，均可在脚本顶部修改。`ASCEND_MF_STORE_URL` 默认在三台均指向 `tcp://P_IP:24707`；按实际 fabric 配置确认跨机传输可用。

给 Router 的真实 PD 请求不要添加 `bootstrap_host=2.2.2.2` 或手写 `bootstrap_room`，Router 会注入正确的对端信息。当前 `simple_bench.sh` 默认包含 fake bootstrap 字段，经过 Router 测试前需要移除这些字段。

该脚本已做 CPU 启动参数契约检查；当前尚无这组 P PP4/CP4 → 双机 D 的 A3 联机数值、KV 传输和性能验证。建议先对比常驻基线的首 token 和多 token 输出，再测试多 chunk、缓存命中及并发。
