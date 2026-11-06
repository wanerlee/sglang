# GLM-5.2 / A3：单机 Prefill 全路由专家卸载 + 两机 Decode 设计稿

状态：已实现 Prefill-only 适配和三机启动入口，80 项隔离 CPU 契约检查与部署预检通过；两个 Triton interpreter 新用例未运行，尚未进行 A3 联机验证。日期：2026-11-04。

审查目录：`/Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/sglang-experiment-ascend-ops-moe-offload-review`。
代码基线：`3c72390a0239c4c79cbe2cc6a046db772b111cdf`；审查开始时工作区干净。下文代码位置均相对此目录，行号对应该基线。

## 实现落点

本次以组合式策略收敛差异，保留 `AscendOpsOffloader` 作为统一入口：

- `npu/options.py`：只负责配置解析和角色检查，不引入设备依赖。
- `npu/execution_mode.py`：`CachedExecution` 调用原混部初始化/forward；`PrefillOnlyExecution` 只创建两层执行池并管理纯 Prefill 生命周期。
- `npu/prefill.py`：继续使用同一个 `LayerPrefetchPipeline`；专家计数内核以编译期 `TRACK_USAGE` 关闭可选统计。
- `npu/model_adapter.py`：增加内部 `offload_all_routed` 参数，原默认驻留策略不变。

只新增一个用户模式开关 `"mode": "prefill_only"`。首个 MoE 层自动纳入卸载，双 bank 容量自动按本地专家数推导，不增加 `resident_routed_layers` 配置项。没有修改 ModelRunner、模型类、dispatcher、scheduler 或 PD KV 协议。

实际启动优先使用 [serve_glm52_pd.sh](serve_glm52_pd.sh) 和 [README 的三机部署说明](README.md#glm-52单机-p-卸载--两机-d)。脚本默认关闭 MTP，可显式开启；支持 `--dry-run` 仅预检并输出配置。

## 1. 推荐部署与范围

已新增独立的 `prefill_only` 专家卸载模式：P 在一台 A3 上运行，全部主模型 routed experts 的长期权重存储放在本机 Host；NPU 仅用两个整层 bank 交替执行。D 在另外两台 A3 上组成一个常驻权重实例。请求沿用现有 PD router 和 Ascend KV 传输。

并行配置保持参考手册的结构，D 按两机规模扩展。手册的 PD 示例本身是 P 一机、D 一机；下面的 D 两机配置是基于代码和手册并行规则的设计推导，不能当成手册已验证配置。[官方部署手册](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/glm_5_2#prefill-decode-disaggregation-deployment)

| 项目 | P：A3-P | D：A3-D0 + A3-D1 |
| --- | --- | --- |
| 物理机器 / NPU die | 1 / 16 | 2 / 32，每机 16 |
| 服务实例 | 1 个 Prefill 实例 | 跨两机的 1 个 Decode 实例 |
| `tp-size` | 16 | 32 |
| Attention DP | 4 | 32 |
| 实际 Attention TP | 4 | 1 |
| MoE EP / MoE TP | 16 / 1 | 32 / 1 |
| CP / PP | 1 / 1 | 1 / 1 |
| MoE 通信 | 保留分支现有 `ascend_ops`，Prefill AllToAllV | 沿用手册 `deepep` / `low_latency` |
| 主模型 routed experts | 所有 MoE 层 Host 常驻，无 Hot 缓存 | 全部 NPU 常驻，按 EP 分布 |
| PD 数据传输 | `ascend` | `ascend` |
| 对外服务 | P:8000，bootstrap:8998 | 仅 D0:8003；D1 是同实例的从节点 |

这里的 TP32 是启动参数所定义的整体并行规模。由于继续使用 Attention DP32 / EP32，并不意味着每个 attention 或单个专家矩阵都做 32 路张量切分。若改为 Attention TP32 / EP1，将改变手册配方，不属于本方案。

主线只替换 P 的 MoE 通信后端，目的是直接复用当前分支已接入的逐层预取；P 的 TP/DP/EP 划分保持手册一致。D 继续用 DeepEP。P 与 D 的 MoE 通信后端不要求相同，它们各自完成本地模型执行，通过已有 KV/请求元数据协议连接。若还要求 P 后端也完全保持 DeepEP normal，见第 8 节。

“全部专家卸载”在本稿中明确指**主模型全部 routed experts，包括首个 MoE 层**。Dense 层、shared experts、attention、embedding、lm_head 和 MTP draft 继续常驻 NPU；执行当前/下一 MoE 层时的临时权重 bank 仍占 HBM。共享专家与 BF16 draft 不在当前 MoE offload 格式覆盖范围内。

P 和 D 使用同一模型版本、tokenizer、主模型量化描述、KV dtype 和 page size；两侧分别创建自己的分布式通信组。三台机器不组成 TP48。D 两机也不注册为两个独立 TP16 worker。

## 2. 原实现的约束与本次适配位置

| 位置 | 当前行为 | 必须做的调整 |
| --- | --- | --- |
| `offloading/moe/npu/ascend_ops.py:73` | 要求单机、TP=EP、MoE TP=1、PP=1 | P 满足，保留这些限制；D 不创建 offloader，无须给 offloader 增加跨机能力 |
| 同文件 `:81` | 只接受 `ascend_ops` | 主线 P 继续使用此后端；D 不携带 offload 配置 |
| 同文件 `:118`、`:135` | 拒绝 `cache_ratio=0` 和空 Hot | 新 `prefill_only` 分支允许零缓存；原混部路径保留现有约束 |
| `offloading/moe/npu/model_adapter.py:35` | GLM 首个 routed MoE 层固定常驻 | P-only 目录把所有 routed MoE 层标记为可卸载；dense/shared 不变 |
| `offloading/moe/npu/prefill.py:45` | Bank B 优先借 Hot；不足时使用 Execution 后半段 | P-only 明确使用两份 Execution bank，容量 `2 * local_experts` |
| `offloading/moe/npu/ascend_ops.py:319` | 无条件初始化 Residency、LFU、缓存替换与地址发布 | P-only 跳过这些对象与初始化 |
| 同文件 `:367` | Decode row mask 要求 Attention DP=EP | 仅混部路径保留；P-only 不初始化 Decode mask，允许 DP4 / EP16 |
| 同文件 `:627`，`npu/cache_update.py:27` | 地址发布 / 缓存维护读取 Hot | 不能只放宽 ratio 校验，否则空 Hot 会进入无效索引路径 |
| 同文件 `:670`、`:722` | 捕获维护图，每轮清统计并重放缓存维护 | P-only 不捕获维护图、不更新 LFU、不恢复 Hot |
| `offloading/moe/npu/prefill.py:245` | 专家计数转换与 usage 统计绑在一起 | 拆出无统计的计数校验/规范化路径，仍保留错误报告 |
| `scripts/moe_offload/serve_glm52_tp4.sh:57` | 拒绝零缓存，固定四 die、DP4、混部参数 | 新增独立 P/D/Router 脚本，保留现有混部脚本 |

以上 Python 路径的公共前缀为 `python/sglang/srt/`。

已有能力可以复用：

- **装载与量化格式**：`layers/quantization/modelslim/modelslim.py:555,579` 已在创建参数时接管 Host 专家；`model_loader/loader.py:158` 已避免把卸载参数整体搬到 NPU；继续使用现有 W8A8/W4A8 格式转换和 seal 机制。
- **逐层流水**：`npu/prefill.py:61` 的 `LayerPrefetchPipeline` 已实现双 bank、首两层预取、combine 后回收、W13/W2 分阶段事件，无须重新设计。
- **Attention TP4 → EP16 的行布局**：`layers/moe/utils.py:810`、`layers/layer_boundary/factories.py:113,124` 和 `boundary.py:503` 已有稀疏 FFN 的 DP×Attention-TP 行分片；Ascend Ops 使用各 rank 的实际行数。不能把“去掉 Decode mask 限制”表述为“完整 DP4 已验证”，但无需预设要重写 communicator。
- **GLM 模型身份**：GLM-5.2 的 `GlmMoeDsaForCausalLM` 复用 DeepseekV2 主模型路径，NextN 复用相应 draft 路径，见 `models/glm4_moe.py:1284`；避免把 hook 加到普通 GLM4 类中。

## 3. P-only 模式设计

### 3.1 配置和角色检查

保留现有 `--moe-offload-config` 入口。JSON 新增 `mode`；下面是实现支持的配置：

```json
{
  "mode": "prefill_only",
  "cache_ratio": 0,
  "execution_slots": 32,
  "host_pool_gib": 48,
  "native_extension_directory": "/tmp/sglang-glm52-p-offload/native_acl"
}
```

`execution_slots=32` 对应 256 个 routed experts / EP16 = 16 个本地专家，两 bank 各 16。通用脚本须从 checkpoint 配置计算，不能把 32 固化在 runtime。`host_pool_gib=48` 是每 rank 的起始预算，实际可行性由格式 probe 和整机内存预检决定。

配置语义：旧配置未写 `mode` 时保持现有混部行为；`prefill_only` 必须同时满足 `disaggregation_mode=prefill`、非 draft worker、NPU、单机、TP=EP、MoE TP=CP=PP=1、无 scheduler/batch overlap。P-only 要求零 Hot，允许 Attention DP 整除 TP；D 配置误带本模式时应在分配前报错。P-only 自动卸载所有 routed 层，原混部默认仍保留首 MoE 层。

纯 P 不需要 offline 热度 profile、cache update interval、replacement/min_usage、decode_mask_capacity；遇到这些混部专用字段应提示配置不适用，避免看似生效。

### 3.2 对象与内存生命周期

从现有实现抽取共同的 catalog、Host 存储、格式转换、资源保留、错误状态与 copy stream 初始化。用独立 P-only runtime 或清晰分离的初始化分支管理以下对象：

1. 本 EP rank 全部 routed experts 的 Host master，初始化后只读。
2. Execution 权重池，两个不重叠的整层 bank；允许 `hot=None`，不要为了兼容创建伪 Hot 缓存。
3. `AscendPrefillWeights`、按层 W13/W2 搬运描述符、copy stream、完成事件及 per-forward 资源引用。
4. 逐专家 counts 与错误标志；不创建 LFU/Hot slot 映射/packed decode 计划/Hot direct key/维护图。

注意 `_init_copy_plan()` 当前既创建公共 copy stream，又创建 Decode 专用计划，需拆开公共部分。`AscendPrefillWeights` 的 bank 来源不能无条件解引用 `owner.hot`；counts 内核不能继续要求 `owner.state.step_access_count` 存在。空收包时仍完成层生命周期和必要 collective，只跳过空 GMM。

继续在参数创建时直接装入 Host，不能改为“整个模型上 NPU 后再卸载”。保留按专家/组件转换、Host 注册与 owner 引用的现有装载流程，避免启动峰值大于稳态容量。

### 3.3 每个 Prefill forward 的执行顺序

1. 在 `ModelRunner` 的主模型 forward 入口检查本轮**全局** `is_extend_in_batch`，重置 invocation，预取卸载层序列中的第 0、1 层到 A/B。
2. Attention、路由、dispatch 正常运行；dispatch 只检查层顺序，不等待未来层权重。
3. 当前层 GMM1 前等待当前 bank 的 W13；GMM2 前等待 W2。W2 搬运可与 GMM1/激活重叠。
4. combine 回传和最终合并提交后记录完成事件；copy stream 等此事件后才能用该 bank 预取序列中的第 i+2 层。
5. 末层完成后保留已有错误检查与流排空/资源回收；P-only 不执行 Hot 恢复、LFU 和缓存维护图。
6. 每个 chunk 都重复上述流程。使用卸载层序列索引选择 bank，不按模型绝对层号奇偶推断。

首个 MoE 层也卸载后，第一份权重预取仍在任何模型层计算之前提交，可与前面的 dense/attention 工作重叠；能隐藏多少传输需实测。

不能只用本 rank 的 `forward_mode.is_extend()`：Attention DP 中，本地 IDLE rank 仍可能承担其他 DP rank 发来的专家工作，必须随全局 prefill 进入同样的 EP collective 和 bank 生命周期。全局 IDLE 由现有 scheduler 跳过，不提交专家 H2D；若它意外进入 offloader，同 decode/target-verify 一样在执行前报错，避免进入未初始化的 Decode 路径。P-only 同时拒绝 DP>1 的 `SGLANG_SCHEDULER_SKIP_ALL_GATHER`，确保全局 phase 可靠。

保留 `model_executor/model_runner.py:508` 的 draft 排除规则。MTP draft 拥有独立 runner；`layer._ascend_ops_offloader` 归属主模型实例，draft prefill 不得推进主模型 bank 状态。

## 4. PD 与两机 D 的集成边界

复用 `disaggregation/ascend/conn.py:95` 继承的现有连接管理。`disaggregation/common/conn.py:1432` 已处理异构 attention TP；对默认 MLA 路径，P Attention TP4 到 D Attention TP1 选择实际 sender，其余必要连接使用 dummy。P 的权重卸载不改变 KV 表示，无须仅为本方案新增 KV 重分片协议。

本方案不启用 `--enable-kv-cache-sharding`：该选项对应的 MLA 路径在 `common/conn.py:1438` 要求两侧 attention TP 相同。保持固定 P/D 角色，不添加动态 role switch。

两个 D 进程使用共同的 D0 rendezvous 地址，分别 `node-rank=0/1`。`entrypoints/http_server.py:3153` 的非零节点不启动 Python HTTP 服务，router 只登记 D0 地址。D 端不传 `--moe-offload-config`，也不初始化 Host 专家池。

MoE Host pool 使用 `memfabric_hybrid.offload` 的 LOCAL 分配器，PD 使用独立的 `TransferEngine`。现代码明确禁止的是 MoE pool 与 MemFabric HiCache 同时占用，不是 PD。LOCAL 池的 `world_size=1/rank_id=0` 是进程内池语义，**不能改成 16**。但 Python 层无冲突不等于底层库已验证共存，必须实测创建顺序、内存注册、错误清理、流排空及进程退出。

三台机器的 `ASCEND_MF_STORE_URL` 指向同一个 P store 地址，D 的 `--dist-init-addr` 指向 D0 的另一个端口；它们用途不同。当前 PD 默认传输协议是 SDMA，`ASCEND_MF_TRANSFER_PROTOCOL` 可选择安装版本支持的协议。需要核对实际 A3 fabric；普通 IP 连通不能证明 SDMA 跨机可达，也不能直接保证 LOCAL H2D 与 PD KV 传输无带宽争用。

MTP 保留手册两侧不同的设置：P NEXTN steps1 / topk1 / draft2；D steps3 / topk1 / draft4。此差异本来就在手册中。draft 权重常驻，并保留 PD 的 draft KV/hidden/topk 元数据链路。checkpoint 必须实际包含与声明量化一致的 NextN 权重；手册列出的某些 W8 下载包不含 MTP，不能只复制参数后假设可用。可复用 `scripts/moe_offload/glm52_mtp_config.py` 做头部预检。若先做无 MTP 基线，两侧同时去掉所有 speculative 参数，作为独立验收配置。

## 5. 内存与吞吐预算

以下使用官方配置的 H=6144、MoE intermediate=2048、256 routed experts、78 层、前 3 层 dense，即 75 个 routed MoE 层；实际 ModelSlim checkpoint 仍须逐字段核对。[官方模型配置](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)

W8 权重本体公式：单专家 `3 × H × I × 1 byte`，每层每 rank `单专家字节 × 256/16`。推算如下，均不含量化元数据、对齐及其他模型权重：

| 项目 | W8A8 | W4A8 权重本体参考 |
| --- | --- | --- |
| 每专家 | 36 MiB | 18 MiB |
| 每层每 P rank | 576 MiB | 288 MiB |
| 双 bank / P rank | 1.125 GiB | 0.5625 GiB |
| 75 层 Host / P rank | 42.1875 GiB | 21.09375 GiB |
| 16 rank Host 合计 | 675 GiB | 337.5 GiB |

主线按手册采用 W8A8。W4 数字只说明权重本体；per-group second scale/offset 等开销不同，不能把 W8 总预算简单减半后当作完整内存需求。

P 若使用 48 GiB/rank，Host 池整机预留是 768 GiB。`npu/storage.py:98` 把该值用于实际 reserve/alloc，并非纯软件上限。还要计入 checkpoint 读取、常驻 CPU 对象、转换暂存、页缓存、系统及服务内存。不能沿用 TP4 脚本的 192 GiB/rank，否则 16 rank 就预留 3 TiB。物理 DRAM 不满足时，W8 全专家 Host 常驻方案不成立，需要调整量化或硬件容量。

HBM 还必须容纳 attention/dense/shared/embedding/lm_head、draft、KV、激活、AllToAllV/HCCL 缓冲与转换峰值。两个权重 bank 仅约 1.125 GiB，不代表单 rank 总显存只有这一规模；容量检查必须发生在 KV 自动预算之前。

更关键的是带宽：每次完整 prefill forward、包括每个 chunk，当前“整层全专家预取”都会搬约 **42.1875 GiB/rank、675 GiB/整机** 的 W8 权重本体。双 bank 减少等待暴露，**不减少搬运字节数**。若有效单 rank H2D 带宽为 B GiB/s，DMA 服务时间约为 `42.1875/B` 秒；例如 B=25 时约 1.69 秒，这是假设带宽下的计算，不是测量结果，真实 TTFT 还受 overlap、激活计算和共享链路影响。

因此先用适中的 chunk 验证，再以生产 prompt/并发测 P 吞吐和 D 等待比例。不要机械沿用当前 TP4 offload 脚本的 chunk512，也不要直接使用手册的超大 chunk。手册 chunk524288 在当前参数处理后会按 Attention DP4 除成每 DP131072；HBM 预算须结合这个实际值。选大 chunk 可摊薄重复权重搬运，但会增加激活/通信峰值和排队时延。

## 6. 实现后的启动模板

以下为显式参数参考，已由新增启动脚本实现。实际使用优先运行脚本的 `--dry-run`；脚本还提供依赖构建、MTP 头部检查与参数预检。三台都要安装兼容的 CANN、torch_npu、sgl-kernel-npu 和 MemFabric；P 额外沿用现有脚本的自定义算子环境及 native ACL 扩展构建步骤。不要只复制启动命令而省略这些依赖。

先在各节点设置模型路径、节点 IP、本机真实通信网卡，并 source 对应环境：

```bash
# 三台一致；IP/网卡/模型路径由实际环境赋值
export ASCEND_MF_STORE_URL="tcp://${P_IP}:24707"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export STREAMS_PER_DEVICE=32
export HCCL_SOCKET_IFNAME="$COMM_IFACE"
export GLOO_SOCKET_IFNAME="$COMM_IFACE"
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

下面容量参数是保守的初次验收起点：context16384、P chunk8192、P max-running16。只保持手册的并行结构和 PD/MTP 机制；通过实机预算后再逐步恢复生产长度/并发。chunk8192 经 DP4 归一后为每 DP2048。

P：

```bash
export TASK_QUEUE_ENABLE=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
export HCCL_OP_EXPANSION_MODE=AIV

python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name glm-5 \
  --device npu --attention-backend ascend --quantization modelslim \
  --load-format safetensors --dtype bfloat16 --trust-remote-code \
  --disaggregation-mode prefill --disaggregation-transfer-backend ascend \
  --host 0.0.0.0 --port 8000 --disaggregation-bootstrap-port 8998 \
  --nnodes 1 --node-rank 0 --tp-size 16 \
  --dp-size 4 --enable-dp-attention --ep-size 16 \
  --enable-dp-lm-head --moe-dense-tp 1 \
  --moe-a2a-backend ascend_ops \
  --moe-ascend-ops-config '{"prefill":"alltoallv","decode":"low_latency"}' \
  --disable-shared-experts-fusion --disable-overlap-schedule \
  --disable-cuda-graph --mem-fraction-static 0.8 \
  --page-size 128 --kv-cache-dtype bfloat16 --context-length 16384 \
  --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
  --max-running-requests 16 --load-balance-method round_robin \
  --speculative-draft-model-quantization unquant \
  --speculative-algorithm NEXTN --speculative-num-steps 1 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 \
  --moe-offload-config "$P_OFFLOAD_CONFIG"
```

Ascend Ops 配置中的 decode 字段是现有 schema 的兼容项；P-only 主模型不执行该分支。P 的 TASK_QUEUE/plan-stream 设置先沿用当前 offload 路径，不能把手册 DeepEP normal 的优化环境变量直接当成 Ascend Ops 的加速开关。

D0、D1 各运行一次，分别设置 `NODE_RANK=0`、`NODE_RANK=1`；其他参数相同。`max-running-requests=128` 在 Attention DP32 下对应每 DP 约 4 个请求的容量；它是初次验收起点，并未维持手册 DP16 下每 DP 8 个请求的负载：

```bash
export TASK_QUEUE_ENABLE=0
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_NPU_USE_MULTI_STREAM=1
export HCCL_BUFFSIZE=650
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32

python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name glm-5 \
  --device npu --attention-backend ascend --quantization modelslim \
  --dtype bfloat16 --trust-remote-code \
  --disaggregation-mode decode --disaggregation-transfer-backend ascend \
  --host 0.0.0.0 --port 8003 \
  --nnodes 2 --node-rank "$NODE_RANK" --dist-init-addr "${D0_IP}:5000" \
  --tp-size 32 --dp-size 32 --enable-dp-attention --ep-size 32 \
  --moe-a2a-backend deepep --deepep-mode low_latency \
  --disable-shared-experts-fusion --mem-fraction-static 0.8 \
  --page-size 128 --kv-cache-dtype bfloat16 --context-length 16384 \
  --max-running-requests 128 --cuda-graph-max-bs-decode 4 \
  --load-balance-method round_robin --tokenizer-worker-num 4 \
  --watchdog-timeout 9000 \
  --speculative-draft-model-quantization unquant \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

Router 在 P/D ready 后启动，可放在 P 主机或独立管理节点：

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation --policy round_robin \
  --prefill "http://${P_IP}:8000" 8998 \
  --decode "http://${D0_IP}:8003" \
  --host 0.0.0.0 --port 6688
```

当前分支也接受更直接的 `--attn-dp-size 4/32`；上述保留手册 `--dp-size N --enable-dp-attention` 写法，内部会归一到 Attention DP，replica DP=1。不要同时堆叠两套 DP 参数。

## 7. 分阶段实现与验收

建议拆成四个可独立检查的改动：

1. **角色与目录**：配置解析、P-only 角色检查、首个 MoE 层可卸载、Host/双 bank 预算。原混部行为默认不变。
2. **P-only 生命周期**：无 Hot 初始化、无 LFU counts、独立错误/清理逻辑；复用原 W13/W2 预取与 dispatch/combine 接口。
3. **部署脚本**：新增 P、D、router 启动模板及 checkpoint/内存预检；不在 D 传播 offload JSON，不沿用四 rank 预算。
4. **A3 三机验收**：先常驻权重 PD 基线，再开启 P-only，在相同输入与模型下比较。

代码验收至少覆盖：

- P-only ratio0 + 两 bank 成功，容量不足提前失败；GLM 75 个 MoE 层均进 Host 目录，首层无特殊常驻，dense/shared/draft 不进入目录。
- P-only 不构造 Hot/LFU/cache-tail graph，不读取空 Hot；原混部的 Prefill→Decode、Hot 恢复和缓存行为无回归。
- DP4 / EP16 下本地 IDLE、非均匀 token、零接收、padding、多个 chunk；global flag 一致，collective 次数/顺序一致，expert ID/counts 不重复也不漏算。
- 沿用现有 `test_ascend_prefill.py` 的异步依赖契约：dispatch 不等未来权重、W13/W2 各自等待、combine 后覆盖；新增零 Hot、第一 MoE 全卸载和 P-only 拒绝 decode 的用例。
- 模拟 W13/W2 部分提交失败、forward 异常、取消/超时后排空流，Host 注册与 bank 在 DMA 结束前不释放。

设备验收至少覆盖：

- P offload 关闭/开启的逐层输出、最终 logits/首 token 与端到端生成对照；量化数值采用合理容差，不能只看服务 ready。
- P16/D32 的主 KV、MTP draft KV 和首轮 seed 传输；长 prompt、chunk 边界、多请求、多 DP 目的 rank。MTP on/off 各有基线。
- D0/D1 组成同一 32 rank 实例，router 只有一个 D 地址，两台均参与计算；D 无专家 Host 池和逐 token H2D。
- 同一 P rank 中 LOCAL offload 与 PD TransferEngine 共存，反复启动/退出、注册失败与传输失败清理正常。
- 记录每 rank/整机 Host RSS、预留内存、HBM 装载峰值与稳态、首/末层 DMA、W13/W2 等待、HCCL/KV 传输重叠。
- 统计 TTFT、P tokens/s、D TPOT、端到端吞吐和 D 等待率。逐步增加 chunk/并发；若 P 成为瓶颈，要据测量调度请求，不能仅凭 D 加到两机就承诺系统吞吐提升。

本次已适配代码并提供 CPU 契约测试和脚本预检，未启动真实模型、未跑 NPU 测试。当前分支 README 也明确 GLM offload 尚未完成整模型 NPU 数值/Graph/性能验收；这里新增的 PD 组合需要单独验收。

## 8. 如果 P 也必须保持手册 DeepEP normal

这可以作为第二阶段，不需要重写预取算法：`layers/moe/moe_runner/ascend.py:566` 已有 DeepEP normal → Ascend runner 的 counts/scale 适配，核心 GMM 已通过 owner 获取权重并等待 W2。

需要新增的是完整生命周期接入：把目前只在 `token_dispatcher/ascend_ops.py:180,210` 里的 start/finish hook 抽成后端可复用接口，或在 DeepEP normal dispatch/combine 中加入同等 hook。start 放在本层 dispatch 前；retire 必须在 DeepEP combine 完成事件已与计算流建立依赖后，才能允许覆盖 bank。核对 DeepEP 的 expert counts、token reorder、INT8 scale 和空 token 契约，禁止重复执行自定义 AllToAllV 与 DeepEP dispatch。

完成适配并单独验收后，P 才可改用 `--moe-a2a-backend deepep --deepep-mode normal`，并移除 Ascend Ops 专用配置。主推荐先落地 P Ascend Ops / D DeepEP 的方案，它已经满足手册的并行划分，也最贴近当前逐层预取实现。
