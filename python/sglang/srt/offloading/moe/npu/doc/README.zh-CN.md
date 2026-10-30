# Ascend V4 offload 运行时装配

> 历史设计记录：本文中的 wave 调度、分组配置及相关接口已在分层分支删除。当前执行方式和配置迁移见[分层实现说明](../../../../../../../scripts/moe_layer/README.md)。缓存策略设计仍供参考。

2026-10-25：当前装配使用唯一运行栈；[重构记录](../../doc/REFACTOR_20261025.zh-CN.md)
列出迁移与验证边界。公共生命周期已提取到 `moe/runtime.py`，模型与量化各自扩展。

上级 `npu/` 目录集中放置 NPU 实现；公共框架见 [公共说明](../../doc/README.zh-CN.md)。

| 文件 / 目录 | 职责 |
| --- | --- |
| `../config.py` / `../backend.py` / `../runtime.py` | 启动配置、资源装配、NPU stream/event/allocator 操作 |
| `../../runtime.py` / `../../owner.py` / `../../graph.py` | 公共 invocation、phase/forward 所有权、graph lease 与显式退休 |
| `../../models.py` / `../../formats.py` | 模型入口适配、完整 bundle schema 与格式接口 |
| `../transfer.py` / `../experts.py` | MemFabric 搬运、DeepEP 与原生 FFN |
| `../storage.py` / `../format.py` / `../format_resolver.py` | host master、共用 ExpertLayerStorage/ExpertBank、格式解析 |
| `../v4_format.py` / `../w8a8_format.py` | FP4/W8A8 recipe、checkpoint 校验和原生转换行为 |
| `../native_acl.py` / `../native/` | ACL host 计划验证/提交、扩展构建与 C++ 源码 |
| `../examples/v4.json` | 每 rank 的 offload 容量配置 |

实现说明：[存储](V4_STORAGE.md)、[EP/FFN](EXPERTS_IMPLEMENTATION.zh-CN.md)、
[ACL](../native/doc/README.zh-CN.md)；本地验证见 [验证记录](../../doc/LOCAL_VALIDATION_1021.zh-CN.md)。

每 batch 命中率、生命周期日志及无额外设备同步的约束见 [维测日志](OBSERVABILITY.zh-CN.md)。

Decode 中七次 ScatterElements 的用途与耗时分析见
[ScatterElements 开销分析](DECODE_SCATTER_ANALYSIS.zh-CN.md)。

`NpuMoEBackend` 在模型构造前安装，由 FusedMoE 的统一权重创建入口接管
host 参数分配；post-load 后装配共享 host master、hot/execution bank、固定 Tensor
residency、离线 Top-K / 周期跨层 LFU、ACL eager prefill、MemFabric decode 和现有 Ascend EP/FFN。
本项依赖 native ACL、真实存储和 EP/FFN 三项适配一起使用。

当前只装配一个兼容 schema arena，分配前拒绝混合格式/不兼容拓扑。地址目录共享
不可变 host/hot/destination 表，binding 只描述目标，wave/tail 可变 scratch 独立；一套 prefill pipeline
拥有所有层的 A/B 读者历史。公共策略只输出 logical target_keys 与显式设备 due，
周期策略捕获普通和维护两种 full graph，非维护 batch 的图尾不包含选择/更新规划。
host 只按有效 decode batch 计数选图，不回读设备热度或路由；详见周期缓存说明。

2026-10-22 缓存接入说明见 [周期缓存配置与验证](PERIODIC_CACHE_1022.zh-CN.md)。
非零热池必须配置 `offline_stats_path` 与 `offline_model_id`；示例 JSON 中的 profile
路径需要替换为真实离线统计文件。在线各层缓存配额可在同一个 EP rank 内重新分配。

公共 `BoundMoERuntime` 在 decoder 层首向常驻 prefill worker 提交任务，在 FusedMoE.forward_impl
进入 routed program，外层原有 shared experts 与缩放保持原路径。层间、forward
间复用 A/B 前会把前一个 compute reader frontier 接到 copy stream。激活打包
workspace 在串行层之间共用。Dispatch 后真实本地计数进入 residency；设备
padding 上界由 ForwardBatch 传入 dispatcher，不使用本 rank Router 频次代替。

Prefill 的初始 microbatch 和后续 wave 共用一个 worker，跨层、跨 chunk 复用线程。
主线程取得当前 wave 后即可提交 FFN，下一 wave 的描述符准备、ACL 提交在 worker 执行；
覆盖 A/B 前仍在 copy stream 等待该 bank 的最后一个 reader。初始组先记录覆盖全部
microbatch 的 copy fence，再允许提交 B，因此首个 FFN 只等待初始组，不被 B 的搬运阻塞。
固定 bank 视图、HostCopyPlan 和 native ACL Batch 使用有界缓存（分别最多 256、1024、1024 项），
命中后复用地址检查结果和 native 描述符；lease、reservation、完成事件仍逐次创建。
缓存淘汰不释放在途 ticket 的资源，runtime 停止提交并 drain 后才清空缓存及释放注册。
`torch_npu` host queue flush 继续保留，确保 stream 上先前的 event wait 排在 ACL 拷贝之前。

`scripts/moe_offload_a3.json` 针对长输入把 `prefill_microbatch_size` 调为 4、
`prefill_max_inflight` 调为 4，`prefill_group_size=8` 不变；初始组若提交满 16 个专家，
ACL microbatch 提交从 16 次降为 4 次，不增加权重池容量。短输入可继续使用更小 microbatch。
CPU 回归验证线程复用、资源保活、覆盖顺序和描述符复用；16K～128K 输入的 TTFT 收益
仍需在 A3 上对比测量，未把 CPU 模拟测试时间作为设备性能结果。

Capture head/tail 覆盖 warmup 与完整模型；update_enabled 在图外关闭/恢复。
真实 replay 不重复执行 Python 统计或 LFU tail，而在提交后记录 fresh NPU
completion events。Eager decode 会明确拒绝。模式转换先排空，保存元数据，
prefill 借池，decode 按原 slot 从 host 重载后检查错误标记再恢复映射和 LFU。
销毁要求先清理 graph lease，停止 worker、drain 所有 NPU 流、移除 hooks，最后
释放注册。失败上下文在 drain 前保留；硬件 fault 不会清零后继续服务。

## 构建与启动

仅在匹配的 Ascend A5 / CANN / torch_npu 环境执行。需要已安装当前仓库和用户
提供版本的 MemFabric、sgl-kernel-npu DeepEP。此 macOS 开发机不能完成硬件验收。

### 单机多卡启动脚本

从仓库根目录执行，先加载目标机匹配的 CANN 环境，再设置实际的 **DeepSeek V4
FP4 或 ModelSlim W8A8 safetensors** 权重目录：

```bash
MODEL_PATH=/path/to/DeepSeek-V4-Flash \
  bash scripts/launch_moe_offload_npu.sh --dry-run

MODEL_PATH=/path/to/DeepSeek-V4-Flash \
  bash scripts/launch_moe_offload_npu.sh
```

默认 `WORLD_SIZE=4`，可见设备 `0,1,2,3`，TP=4、EP=4、DP=1，MoE TP=1；
服务地址 `127.0.0.1:8818`。换用卡 4–7：

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
MODEL_PATH=/path/to/DeepSeek-V4-Flash \
  bash scripts/launch_moe_offload_npu.sh
```

`ATTN_DP_SIZE` 控制同一个 TP/EP 组内的 Attention DP，默认 1，必须整除
`WORLD_SIZE`；模型副本 `--dp-size` 和 `--moe-dp-size` 固定为 1。
例如单机 16 卡的 TP16 / Attention DP16 / EP16：

```bash
WORLD_SIZE=16 ATTN_DP_SIZE=16 \
MODEL_PATH=/models/DeepSeek-V4-Flash \
MOE_OFFLOAD_CONFIG=/configs/moe-offload.json \
MOE_LOG_LEVEL=debug \
  bash scripts/launch_moe_offload_npu.sh
```

可见设备未指定时自动生成 `0..WORLD_SIZE-1`；也支持 `WORLD_SIZE=8 ATTN_DP_SIZE=8`
和能整除 TP 的部分 Attention DP。脚本仍为单机启动，需有对应数量的可见 NPU。
所有 offload 预算仍按每 rank 计，默认 128 GiB host 预留在 16 rank 时合计 2 TiB，
应先调整实际 JSON 预算。普通 decode 在 Attention TP=1 时可使用 BS=1 的图 bucket；
EP 接收容量由 DeepEP 和 workspace 上界决定，不能把 graph BS 当作接收行数。

全局有 prefill 时，全部 rank 的 offload 阶段为 PREFILL，本地 IDLE/DECODE 模式
保持不变；空闲 rank 仍执行专家 dispatch/compute/combine，以服务其他 rank 的路由。
全局全部空闲时保留当前阶段，不恢复/借用热池，也不推进 decode LFU 周期。
纯 decode 必须使用 full graph；不提供 graph miss 后的 eager decode 回退。

脚本从自身位置定位当前源码并设置 `PYTHONPATH`，不依赖固定的 `/home/.../sglang`
目录。默认先构建 ACL 扩展；`BUILD_NATIVE=0` 可复用与当前环境匹配的已构建扩展，
扩展目录直接读取 offload JSON 的 `native_extension_directory`，避免构建与运行使用不同目录。

`MOE_OFFLOAD_CONFIG` 可指定自己的 JSON，默认使用
`python/sglang/srt/offloading/moe/npu/examples/v4.json`。默认 host 保留量为
**每 rank 128 GiB，四 rank 合计 512 GiB**，workspace/metadata 限额也是每 rank 的；
启动前应按机器和模型调整 JSON 中的预算。

首启默认最大并发 8、decode buckets 为 `1 2 4 8`、chunked prefill 和最大 prefill
token 数均为 1024、context length 为 8192，`mem_fraction_static=0.8`。
`MAX_RUNNING_REQUESTS`、`CHUNKED_PREFILL_SIZE`、`CONTEXT_LENGTH`、
`MEM_FRACTION_STATIC` 可通过环境变量修改；并发变化时脚本同步生成捕图 buckets，
捕图上限按 Attention TP 对齐，避免低并发时所有候选 bucket 被过滤。
增大 prefill/concurrency 时还需检查 JSON 的 `max_dispatch_rows` 和 workspace 上界，
容量不足会拒绝运行。`--dry-run` 仅打印命令，不验证真实权重和设备。

此脚本使用 V4 的 `dsv4` attention、DeepEP default、full decode graph 和 eager prefill，
启用本后端要求的 `-1` 路由与逐专家计数。脚本默认根据量化描述选择 `fp8`
（原生 FP4 experts）或 `modelslim`（W8A8_DYNAMIC），也可用 `MOE_EXPERT_FORMAT`
显式指定并校验；两条路径均关闭 FP4→FP8 解量化。
它不带 EAGLE 或 HiCache 参数，关闭 overlap；Attention DP 由 `ATTN_DP_SIZE` 控制。GLM-5.2 权重
不适用于当前后端，实际启动时会在构建扩展前检查 `model_type` 和 EP 可整除性。

默认 JSON 的 `offline_stats_path` 是占位符。非零 cache_ratio 必须提供真实 profile，
`offline_model_id` 与 profile 的 `model_id` 一致；相对路径以 JSON 所在目录解析。
脚本在构建扩展前检查文件和 ID，格式参见 [周期缓存配置](PERIODIC_CACHE_1022.zh-CN.md)。

日志默认落盘到新建的 `${TMPDIR:-/tmp}/sglang-moe-offload/<时间>-<PID>/server.log`，同时输出到终端。
同目录保存 offload 配置快照和 logging.json；可用 `LOG_DIR` 指定持久目录。脚本保留
server 的 exec、信号和退出码，不将 server 放入 shell pipeline。
默认 root/MoE 日志为 INFO，HTTP 为 WARNING，每 batch MoE 命中率已开启；
普通 SGLang decode 汇总保持原有 40 步间隔，`DECODE_LOG_INTERVAL=1` 可逐步打印。

首轮调通可只打开 MoE DEBUG，不增加其他模块的 DEBUG 输出：

```bash
MODEL_PATH=/models/DeepSeek-V4-Flash \
MOE_OFFLOAD_CONFIG=/configs/moe-offload.json \
MOE_LOG_LEVEL=debug LOG_DIR=/tmp/moe-offload-debug \
  bash scripts/launch_moe_offload_npu.sh
```

默认 `LOG_REQUESTS=1`、`LOG_REQUESTS_LEVEL=0`、`LOG_REQUESTS_FORMAT=json`，记录请求
元数据；`LOG_REQUESTS=0` 可关闭。`SGLANG_LOGGING_CONFIG_PATH` 如已设置则优先使用，
脚本不会覆盖它，此时 logger 级别由该文件控制。

需要请求链路追踪时，另设置 `ENABLE_TRACE=1 OTLP_TRACES_ENDPOINT=localhost:4317`；
须已有对应协议的 OTLP collector 和 OpenTelemetry SDK/exporters。trace 默认采用
异步 exporter（尊重已有 `SGLANG_TRACE_ASYNC`），不隐式安装依赖或启动 collector。
需要 NPU Graph warmup/capture 时间线时单独设置 `PROFILE_CAPTURE=1`，结果写入
`SGLANG_TORCH_PROFILER_DIR/graph_capture_profile`，默认位于本次 LOG_DIR/profiler 下。
请求 trace 与 capture profiling 默认关闭；capture profile 不代表稳定态 replay 的时间线。
日志与采集均有开销，性能测量时关闭 DEBUG/trace/profiling 并按需调低 batch 日志频率。
`ASCEND_LAUNCH_BLOCKING` 继续清除，`TASK_QUEUE_ENABLE=1` 保持不变。

`--dry-run` 打印计划路径、开关和命令，不创建日志目录，不加载 profile/权重或 NPU；
真实启动才检查 profile 文件及其 model ID，并在启用 trace 时检查 exporter 依赖。

### 手动构建与启动

```bash
python -m sglang.srt.offloading.moe.npu.native_acl --build-directory /tmp/sglang-moe-acl
export MOE_ENABLE_TOPK_NEG_ONE=1
export DEEP_USE_MODE=default
export MOE_EXPERT_TOKEN_NUMS_TYPE=1
python -m sglang.launch_server \
  --model-path /path/to/DeepSeek-V4-Flash \
  --device npu --tp-size 4 --ep-size 4 \
  --moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend ascend \
  --disable-overlap-schedule --disable-shared-experts-fusion \
  --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled \
  --moe-offload-config python/sglang/srt/offloading/moe/npu/examples/v4.json
```

示例中的 host_pool_gib、workspace/metadata 及 max_dispatch_rows 是明确的容量配置，
须按每 rank 模型大小、batch/EP 接收上界和机器资源调整；不是测得的最优值。
host_pool_gib 必须容纳探测后的原生物理 bundle（含 padding）与 MemFabric 的 GiB
保留粒度。max_dispatch_rows 须覆盖最坏 EP 接收容量，超限拒绝执行，不截断 token。
loader 一次只转换单专家并受 workspace_limit_bytes 限制。capture 从 KV/输入缓冲
分配完后的基线测量 NPU allocator peak，把 graph/临时内存增长保守计入 workspace；
这不能替代实机驱动内存、通信私有 workspace 和真实峰值监测。

当前支持 PP=1、MoE DP=1、MoE TP=1，TP+EP 和组内 Attention DP 可用；前三层及 hash 层 routed
专家和 shared experts 常驻。DeepEP default 策略需要 -1 路由屏蔽支持。未接入
FuseEP、DeepEP ops/alltoall、AscendTP、EPLB、elastic EP、LoRA、speculation、torch.compile、并发
forward、其他专家 pager 或在线权重更新；不满足条件在加载或绑定时拒绝。

## 目标机验收

缓存策略、原生装配与 Graph 生命周期的 CPU 契约测试已随缓存集成恢复，运行方法见
[周期缓存配置与验证](PERIODIC_CACHE_1022.zh-CN.md)。这些测试使用测试目录内的
字节复制、事件和通信替身，不启用生产代码的 CPU MemFabric 路径。

目标机仍需完成：真实扩展链接与 H2D 字节校验，host 同份存储 ACL/MemFabric 双路径
字节对比，原生 NZ FP4 bank 与常驻 FFN 数值对比，EP padding/空 rank/尾组，
多 bucket 连续 replay 与 prefill/decode 切换，workspace 峰值及 H2D/F 时间线。
`RuntimeCapabilities` 表示实现提供完整捕图路径，不表示已获得这些硬件结果。

W8A8 主模型适配的格式、预算、零 offset 限制、INT8 clamp 激活及实机验收见
[W8A8 设计与验证](W8A8_ADAPTATION.zh-CN.md)。DSpark 仍关闭。
