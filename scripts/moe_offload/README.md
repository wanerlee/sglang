# Ascend Ops MoE offload

## 怎么启动

前提：已安装匹配的 CANN、torch_npu、sgl-kernel-npu、DeepSeek recipe 和 MemFabric（含 A3 构建源码）。

修改 [serve_tp4.sh](serve_tp4.sh) 开头的权重路径、四个空闲 die 和端口。`CACHE_RATIO` 默认 `0.28125`，可改为 `0.5`；`ENABLE_MTP=1` 开启 MTP。然后在仓库根目录运行：

```bash
bash scripts/moe_offload/serve_tp4.sh
```

脚本自动准备本地编译缓存并启动服务，日志写入 `LOG_DIR`。更换依赖版本后更换 `RUNTIME_DIR`。所有保留的 offload 优化默认生效。

也可用 `scripts/launch_moe_offload_npu.sh` 启动，默认 `ATTN_DP_SIZE=2`，可设为 `WORLD_SIZE` 的任意正整数约数。例如 16 die 使用 TP=EP=16、Attention DP=2、Attention TP=8，模型副本 DP 和 MoE DP 仍为 1：

```bash
MODEL_PATH=/path/to/DeepSeek-V4-Flash-W8A8 \
WORLD_SIZE=16 ATTN_DP_SIZE=2 \
MOE_OFFLOAD_CONFIG=/path/to/offload.json \
bash scripts/launch_moe_offload_npu.sh
```

`WORLD_SIZE` 按 die/rank 计数，先确认可见设备及每 rank 的 Host 预算。脚本会按 Attention TP 对齐 decode graph 的 capture bucket；并发请求数仍由 `MAX_RUNNING_REQUESTS` 控制。`--chunked-prefill-size` 会由框架按 Attention DP 调整，比较性能时应核对日志中实际 `toks`。

服务 ready 后，可检查历史算术错题或测短请求性能（结果写入 `/tmp`；端口不同时加 `--port 端口`）：

```bash
python3 scripts/moe_offload/test_requests.py
python3 scripts/moe_offload/test_requests.py --bench
```

这不是 `sglang.bench_serving`：使用 8 条短自然语言提示，固定输出 128 token，并发 1/4/8 各两轮，轮换固定 DP rank；不额外预热、不自动重置缓存，首次与后续轮都记录。BS 指请求并发；TPOT 按首末内容流块间隔除以输出 token 数减一计算，含客户端/流式传输影响，不是设备逐 token 时间。历史错题和原始错误已标在脚本中，文本格式不匹配需人工检查。

## DSV4 单机混布：部分专家用于 Prefill 常驻

在现有 offload JSON 中增加 `"prefill_resident_ratio": 0.25`，即可启用本地实验功能；不配置或设为 `0` 保持原路径。范围为 DSV4 Flash ModelSlim W8A8、单机混布、PP=CP=1、无 MTP，通信后端仍为 `ascend_ops`。本功能尚待 NPU 数值和性能验收，不能用 CPU UT 代替设备验证。

每个卸载层、每个 EP rank 固定保留前 `floor(local_experts * ratio)` 个本地专家，其余专家继续双 bank 流水预取。沿用已有物理槽 GMM 和逻辑专家计数，不增加 GMM 次数，也不改通信、算子或 MemFabric。原有常驻 hash 层不参与这项比例计算。

Decode 的 Hot 容量、LFU、packed/Hot 直算选择和执行槽容量不变。新增 Prefill 常驻区不能被 Decode 覆写；Prefill 冷 bank 复用 Decode 执行区，并在不够时扩展底层分配，不借用 Hot，所以此模式无需回填 Hot。常驻专家仍可能同时存在于 Decode Hot 中：这是保留 Decode 行为的代价，不能把这项比例当成 Decode 命中率或 Hot 容量比例。

例如 256 专家、EP8、40 个卸载层、`execution_slots=32`、比例 0.25：每层本 rank 有 8 个常驻、24 个冷专家，Prefill 每层 H2D 权重字节数降低 25%。新增 HBM 为 `(40*8 + 2*24 - 32) * expert_nbytes = 336 * expert_nbytes`，另有少量 group metadata；`expert_nbytes` 含运行时权重及量化辅助张量。若每专家约 24 MiB，额外权重空间约 7.875 GiB/rank，实际以日志为准。EP4 Hot 直算布局还要保持 Hot 与 Decode 执行区相邻，额外空间另按布局计算。物理槽总数超过 1024 时明确拒绝，不切换 Decode 路径来规避限制。

**NPU 对照步骤：**

1. 沿用当前可运行启动命令，仅分别设置 `prefill_resident_ratio=0` 和 `0.25`，其余参数、权重、输入和实际 Prefill `toks` 保持相同。保留原 JSON 中的 `offline_stats_path` 等配置。启动脚本无需改动。
2. 对每组执行相同的“短请求 → 长请求（跨多个 Prefill chunk）→ 短请求”，固定采样参数，核对输出及错误日志；再测原并发。检查切回 Decode 后仍使用原 graph/cache 路径。
3. 启动日志的 `Mixed Prefill residency` 应列出常驻专家数、新增 HBM 和每层 Prefill H2D 字节数。Profiling 对比同一卸载层，确认只搬冷专家且每层 GMM 次数未增加。
4. 比较 TTFT、TPOT、端到端吞吐和可容纳的实际并发。常驻权重挤占 KV cache，不能只凭 H2D 减少判定混布收益。若内存不足，降低该比例；不要同时调其他变量后直接归因。

## 当前边界

以下“已验证”指合入前的 Ascend Ops 基线或上游测试；整层 prefill 与 Hot 直算的本次兼容合入尚待 NPU 验收。

- 已验证 A3、DeepSeek V4 Flash ModelSlim W8A8、TP=EP=Attention DP=4；MoE TP、CP、PP 均为 1。要求专家形状一致、量化 offset 为零。
- 原固定槽路径已验证请求并发 1～8、28.125%/50% 缓存及 MTP 开关；MTP 使用 EAGLE steps=2、topk=1、draft tokens=3。基础回归不替代完整精度评测。
- 新增 Decode Hot 直算与执行槽复用：四 rank、无 MTP、`execution_slots` 等于本地专家数且 Hot 加执行槽不超过 1024 时自动启用；默认 28.125% 缓存满足条件。其他配置（包括额外 execution 槽）保留本地已有的 packed 搬运路径，无需额外开关。上游快速对照主要改善 BS1/4，BS8 可能略慢；不保证所有 batch 都加速。
- 共享专家、最前面的 `max(3, num_hash_layers)` 层和 MTP draft 权重常驻。Host 预算每 rank 64 GiB，四 rank 合计 256 GiB。
- 每 rank 上限为 256 个本地专家、1280 个 Hot 槽、8192 个专家 key；有效行 mask 默认容纳 128 个本地 padded/验证 token，不等于请求 BS。
- 不支持与 EPLB、冗余专家、batch overlap、模块 CPU offloader 或 MemFabric HiCache 同时使用。

## 保留的实现

- Prefill 适配自 [codex/moe-prefill-opt-20261102](https://github.com/pachi-33/sglang/commit/2dc7903d0bebe6f003db2149f8a612e641a5e190)：每次全局 prefill forward 入口、任何模型层计算之前，提交前两个卸载层的全量专家权重，使用两个 bank 交替执行。第 i 层的回传及最终合并提交后，再提交同一 bank 的第 i+2 个卸载层；拷贝流等待第 i 层 combine 完成后才能覆盖 bank。dispatch 入口只校验层次顺序，不等待未来层权重；GMM1 前等待当前层 W13，GMM2 前等待当前层 W2。这样当前层的 dispatch / FFN / combine 可以与未来层 H2D 并行，W2 仍在传输时也能执行当前层 GMM1 和激活。每层仍使用原生一次 dispatch、一次 FFN、一次 combine，不按专家 wave 拆分。
- Prefill 是否启动由本轮 batch 的全局 `is_extend_in_batch` 判断；本地 IDLE / DECODE rank 仍参与其他 rank 的 prefill。不会读取上一轮残留的 RuntimeContext phase。每个 chunk 都重新全量预取；Decode / TARGET_VERIFY / 全局 IDLE 不启动 prefill 搬运；Hot 直算仅接入 decode 路径，prefill 仍使用整层 bank 和逐专家计数。
- Prefill 的 ACL 描述符在装配时按 W13/W2 分为两个阶段，每阶段包含对应的 weight、scale 和 offset，并独立记录完成事件。只合并同一源注册 owner / 目标 owner 中源、目标地址都连续的已验证范围；不能跨越 checkpoint FP32 scale 到运行时 BF16 scale 留下的源端间隙。对 32 专家、6 组件的典型 W8A8 布局，192 个 region 可合并为 68 个 ACL entry，实际取决于物理布局。所有原始 region 的越界、目标重叠和映射别名检查仍保留。原生 batch 继续通过 `OpCommand::RunOpApi` 有序提交，等待事件、H2D、完成事件依次入队。AllToAllV 使用紧凑接收布局；AllGather 保留原生 payload / scale 容量和无效行屏蔽。空接收跳过 FFN；同一 copy FIFO 仍保证旧 W2 完成后才覆盖 bank。
- Prefill 用一次 Triton 操作完成计数格式转换、错误屏蔽和访问统计，GMM 按完整本地专家顺序读取 bank。没有额外的专家打包、激活重排或重复量化；FFN 使用实验分支现有的 W13 → 激活 → W2 实现。
- 每 rank 有 E 个本地专家时，A 使用 Execution 前 E 槽；B 优先借用 Hot 前 E 槽。Hot 不足 E 时，需在 offload JSON 显式设置 `execution_slots >= 2*E`，由 Execution 后 E 槽提供 B。默认 `execution_slots=E`，容量不足直接报错，不额外分配权重池。
- 最后一个 prefill combine 之后回填借用的 Hot 前缀，异常退出也会在清理前回填并排空流。仅恢复被借用的槽位，保留缓存 owner / generation / LFU 状态，随后才执行缓存维护与 decode。
- Decode 在符合条件时直接读取 Hot；miss 只在执行槽未保留同层同专家权重时从 Host 搬入，使用完整专家 key 防止跨层误复用。其他配置保留固定执行区、活跃专家 W13/W2 和 scale 的稳定打包与清零尾部计数。有效 token mask 排除 padding 和空 rank。
- 每次全局 prefill 入口先使 execution key 失效，再开始整层预取，避免 decode 复用被 prefill 覆盖的权重；本地 IDLE / DECODE rank 同样处理。GMM 的物理槽索引与激活的逻辑专家计数分开，prefill 始终使用自己的计数。
- Attention DP 不再要求等于 EP，CP 仍限定为 1。有效行 mask 使用框架按 Attention TP 分片后的 `num_token_non_padded`；第一个卸载层生成一次，后续卸载层的 dispatch/combine 复用。图捕获记录这次比较，重放读取更新后的计数，不再用未分片的 `input_ids` 长度。该适配同时用于 packed 和 Hot 直算的 Decode 路径；P-only 不分配 Decode mask。
- 融合准备搬运地址、逐专家计数和访问统计，缓存更新后统一发布地址表；跳过已校验为零的 offset，各权重/scale 分别使用相同计数调用 `group_pack_copy`。
- 独立流搬运 W13/W2，W2 与第一段专家计算重叠，第二段计算前等待完成。
- 周期 LFU、固定槽缓存搬运、融合缓存更新及维护图重放。Hot 直算路径跳过命中的 D2D；packed 路径将命中专家 D2D 到执行区，miss 从 Host 搬入。

## 本次合入验证

CPU 契约测试：`python3 test/registered/unit/offloading/test_ascend_prefill.py`，覆盖 bank 容量、异步读写顺序、跨 forward event、逐专家计数、部分拷贝失败后的 Hot 恢复和 Prefill→Decode 切换。计数内核使用 Triton 的 CPU 解释模式执行生产函数；需要安装 Triton / NumPy。

本地隔离 serving / NPU 依赖后通过 31 项 CPU 契约测试；关键回归在旧实现上失败。测试执行真实生产 Python 生命周期与描述符逻辑，设备流 / collective / 内核使用测试边界。覆盖：未来层未就绪时当前层仍能完成计算并提交后续预取、W13 与 W2 各自的读取依赖、空 rank、独立通信流、combine 后 bank 复用、第二阶段提交失败时的 Hot 恢复，以及 decode 路径保持原有依赖。连续范围合并测试检查实际字节结果、源/目标间隙、owner 边界、越界和别名拒绝。CPU 测试不证明 NPU 数值或性能收益。

### 2026-11-04 trace 定位

输入为 `trace_2407_2471ms_analysis.html`，NPU 4、`EXTEND bs=1 toks=2048`、EP=8。时间以下均相对 2407 ms 窗口起点。HTML 的 track 7 实际为物理 stream 31（HCCL），主计算流是 track 3 / stream 32，DMA 是 stream 39。

| 观察 | 实测时间 | 含义 |
| --- | --- | --- |
| 主计算流等待未来权重 | 19.359–36.268 ms，16.909 ms | H2D 完成事件在 36.267 ms，随后 counts AllGather 才能启动 |
| 下一次同类等待 | 47.518–68.141 ms，20.623 ms | 跨出 64 ms 窗口；不能将整段加入窗口内统计 |
| 一层完整 H2D | 8.742–36.267 ms，27.525 ms | 192 个 DMA，计算气泡期间拷贝仍忙 |
| 下一层 H2D 启动 | 40.664 ms | 两批之间约 4.398 ms 的拷贝空档可通过恢复流水减少 |
| 两次回传 AllToAllV | 0.528 / 0.480 ms | 条目缩短，但未来权重等待被提前到了 dispatch 前 |
| 两次 native 队列任务 | 0.987 / 1.063 ms | Python 快速返回仍不能消除共享 launch worker 阻塞 |

按窗口裁剪后，Computing / Communication / Free 分别是 17.756 / 2.161 / 44.083 ms，合计 64 ms。DMA 活跃约 54.972 ms，与这些分类重叠，不能相加；Free 并不表示 DMA 空闲。根据代码布局推算，一层每 rank 约搬运 769.5 MiB（含量化元数据），27.525 ms 对应约 27.3 GiB/s；trace 没有直接给出拷贝字节数，此带宽是推算值。

本次取消 `dispatch(i) -> 等待 W(i+1)` 的过早依赖，将权重截止点恢复到实际使用它的投影。此举允许当前 combine 提前完成，使下一批 H2D 能更早排入 FIFO；同时用分阶段事件重叠 W2 和 GMM1，并合并连续范围减少提交开销。H2D 与通信会重新重叠，HCCL 条目可能变长，应比较稳定态 forward / TTFT 和完整层周期，而非单独优化通信条长度。27.5 ms 的整层传输表明当前也有带宽约束，不能承诺调度修改消除全部气泡。

目标设备验收应同时确认：当前 dispatch 不再等待未来 ticket；GMM1 不读取未完成的 W13、GMM2 不读取未完成的 W2；相邻 H2D 批次的空档是否缩短；native 提交耗时是否降低；两种 prefill 通信模式、空 rank、prefill→decode 数值和 Hot 恢复正确。`serve_tp4.sh` 不需要新增参数。

原生扩展需要重新构建：`python3 -m sglang.srt.offloading.moe.npu.native_acl --build-directory "$RUNTIME_DIR/native_acl"`（先按启动脚本设置环境）。`serve_tp4.sh` 已在每次启动时调用此构建步骤，无须修改启动参数。旧扩展的 `torch_npu_stream_flush` 协议会被拒绝，避免新 Python 静默加载旧提交路径。原生异步任务出错会由 torch_npu 队列在后续检查 / 同步时上报，owner 保留搬运资源直到流排空。

本地主机没有 `torch_npu`，原生 ACL H2D、NZ bank 视图、AllToAllV / AllGather 数值、空 rank、prefill→decode 图重放与 TTFT 仍需在目标 NPU 验收。可继续使用上述请求检查脚本；请同时对比 offload 关闭时的输出与真实设备时间线。

Attention DP=2 的 CPU 回归：`test/registered/unit/offloading/test_ascend_decode_mask.py` 使用真实 graph registry 更新计数，覆盖 TP4/8/16 下的 padding、空 rank、bucket 切换和 MC2 mask 传递；CPU tracing 检查计数不会固化为捕获时的值，不能替代 NPU graph 验收。`test_ascend_offload_launcher.py` 检查 dry-run 命令及 graph bucket 对齐。2026-11-04 合入实验分支 `0804dde563` 后，使用上述 CPU harness 运行相关 94 项 UT，全部通过；范围包括 Decode mask、启动脚本、Prefill 借池回填、配置、P-only 生命周期及 GLM 格式和模型目录等 CPU 测试，不包含原生 C++ 搬运队列及 NPU 验收。

NPU 验收先用同一模型、配置和设备，将 `ATTN_DP_SIZE` 设为 2，确认服务完成 decode graph capture；用短请求分别检查串行和并发 1/2/4/8、长请求后再发短请求，确认无错误或 hang、输出与关闭 offload 时一致。BS=1 时应覆盖其余 Attention DP 组空闲的情况；同时记录 commit、最终启动参数和实际 `toks`。这里尚未声明 Attention DP=2 的真实 NPU 测试通过。
INT8 专家激活遵守模型的 `swiglu_limit`，截断后再量化；该正确性修复与搬运优化分开对照。

### 上游 Hot 直算合入验收

来源为 fishAndShrimp/sglang 的 `f2f42f6c85`（INT8 SwiGLU clipping）和 `67d1f62fdf`（Hot 直算与执行槽复用）。合入保留当前 prefill 事件依赖及本地 packed 回退路径。

本次 CPU 回归额外验证全局 prefill 在提交前清除 execution key（包含本地 IDLE / DECODE）、decode 保留 key，以及激活按实际 GMM 分组格式选择逻辑计数。移除失效操作或恢复错误的计数选择后，对应用例均失败。

目标 NPU 上运行 `test/manual/test_ascend_clipped_int8_activation.py` 和 `test/manual/test_ascend_offload_fixed_slots.py`，并检查 prefill→decode、空 rank、额外 execution 槽、MTP 和大缓存回退。CPU 契约只能验证生命周期和调度选择，不能替代真实权重搬运、GMM 数值或 Graph 验收。

## DeepSeek V4 Flash ModelSlim W4A8

DSV4 Flash routed experts 已接入现有 `W4A8_DYNAMIC` 路径，支持 per-channel 和 `group_size` 声明的 per-group；使用 INT4 权重、INT8 激活。这与 `W4A8_MXFP` 不同，后者尚未接入 offload。W13/W2 和各卸载层的量化格式、分组及形状须一致，暂不支持 `activation_use_clip=True`。

```bash
MODEL_PATH=/mnt/weight/DeepSeek-V4-Flash-DSpark-w4a8 \
WORLD_SIZE=16 ATTN_DP_SIZE=2 \
MOE_OFFLOAD_CONFIG=/path/to/offload.json \
bash scripts/launch_moe_offload_npu.sh
```

启动脚本默认从 `quant_model_description.json` 的 routed expert 条目识别 W4/W8，非专家的 Attention 权重可以是 W8；仅在没有专家条目时使用 `model_quant_type`。也可设置 `MOE_EXPERT_FORMAT=w4a8` 检查格式是否匹配，实际 scheme 仍由 `--quantization modelslim` 选择。请使用实际 W4A8 checkpoint，改路径标签或环境变量不会转换权重。

上述 DSpark checkpoint 的顶层 `model_quant_type=W8A8_DYNAMIC`，但第 0–42 层专家条目均为 `W4A8_DYNAMIC`、`group_size=0`，因此选择 per-channel W4A8。其 W1/W3 为 INT8 `[1024,4096]`、W2 为 INT8 `[2048,2048]`，每字节打包两个 INT4；加载 `weight_scale`、`weight_offset`、`scale_bias`，不要求 `weight_scale_second`。识别主模型格式时排除 MTP 专家；`dspark_full_dequantized_to_bf16` 标记不会覆盖实际专家量化条目。此启动命令运行目标模型普通推理，不启用 DSpark 投机解码。

Prefill 复用整层双 bank 预取；Decode 使用 packed 执行区和原有缓存策略，W4 不启用仅适用于 INT8 权重的 Hot 直算。DSV4 的 INT8 SwiGLU 限幅沿用现有 runner；Attention DP=2 的有效行 mask 保持不变。Host/HBM 预算按 W4 权重及全部 scale/bias 元数据核算，不应直接将 W8 总预算除以二。

CPU 回归覆盖 DSV4 常驻前缀、按 EP rank 加载 packed 专家、per-channel/per-group scale/bias、Decode packed 路径和启动格式识别。尚需在目标 NPU 使用同一 W4A8 checkpoint 对照关闭 offload 的输出，检查 graph capture、短请求、并发 1/4、长请求后短请求及 prefill→decode 切换；不要用 W8A8 输出作为 W4A8 的等价基线。底层格式和 ACL reload 可先运行下文的 `test_ascend_glm_offload_formats.py --formats w4a8 w4a8-group128`，该脚本复用相同格式实现，不依赖 GLM 权重。

## GLM-5.2 ModelSlim W8A8 / W4A8

GLM-5.2 的 offload 接入已实现，尚未在目标 NPU 上完成整模型数值、Graph 和性能验证。上文 DeepSeek V4 的已验证结果不能直接作为 GLM 验收结果。本节使用独立启动脚本，不修改原 `serve_tp4.sh`，也不继续调整 prefill 调度。

支持 checkpoint 中的 `W8A8_DYNAMIC` 与 `W4A8_DYNAMIC`，均使用 INT8 激活。`--quantization modelslim` 根据 checkpoint 的量化描述选择格式，无须增加 W4/W8 启动开关。W4 支持 per-channel 和 metadata 声明 `group_size` 的 per-group；W13/W2 量化格式、分组和激活配置须一致。不支持 `W4A8_MXFP`、`activation_use_clip=True` 或混合 W4/W8 专家投影。W8 仍要求量化 offset 为零；W4 保留并加载其 scale、offset、`scale_bias` 及 per-group second scale/offset，按 resident 转换生成运行时权重与 scale/bias。

GLM 按实际 routed MoE 层分布建立专家目录，最前面的 routed MoE 层、dense 层及共享专家常驻。W4 decode 使用 packed 执行区路径；W8 在满足原有限制时可使用 Hot 直算。Prefill 沿用当前的双 bank 全层预取和 W13/W2 分阶段等待。

Triton-Ascend 3.2.2 的组件映射使用整型编译常量编码。该版本不能直接对 `tl.constexpr` 包装的 tuple 做下标取值；CPU interpreter 不经过这段 JIT 前端，不能替代编译验证。可在 NPU 上先运行 `python3 test/manual/test_ascend_offload_fixed_slots.py --metadata-only`，检查 W8、W4 per-channel/per-group 的两个生产搬运元数据内核；此模式只分配小型 NPU 张量，不需要模型权重或创建 MemFabric Host 池。

在仓库根目录启动 W8A8：

```bash
MODEL_PATH=/models/GLM-5.2-W8A8 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
HOST_POOL_GIB=192 CACHE_RATIO=0.25 \
bash scripts/moe_offload/serve_glm52_tp4.sh
```

W4A8 per-channel 使用同一脚本，并显式调整 Host 预算：

```bash
MODEL_PATH=/models/GLM-5.2-W4A8 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
HOST_POOL_GIB=96 CACHE_RATIO=0.25 \
bash scripts/moe_offload/serve_glm52_tp4.sh
```

`HOST_POOL_GIB` 默认 192，单位是 **每 rank 的 GiB**。脚本不从目录名推断量化位数；W4 的 96 GiB 仅为 per-channel 的配置参考。Per-group 还要计入 INT64 second scale 和 offset 等 metadata，例如该模型 group size 128 时估算约需 100 GiB/rank，96 GiB 不够。请按启动日志和实际 schema 核算调大；预算不足时 loader 会明确报告所需 GiB 并拒绝启动。

按本次 GLM-5.2 配置、TP=EP=4 估算，卸载专家的权重本体约为 W8 的 **166.5 GiB/rank**、W4 的 **83.25 GiB/rank**；这些数字尚未包含 scale、bias、offset、转换所需的 Host extent 和分配对齐，也不代表整个服务进程的内存。四个 rank 的 Host 池预算相加分别为 768 GiB（192 × 4）和 384 GiB（96 × 4），此外还要给常驻权重、checkpoint 读取和系统内存留空间。实际配置与 checkpoint 不同时，以 loader 预算核算为准。

默认脚本设置 `--attention-backend ascend --page-size 128`，TP=EP=Attention DP=4，MoE TP/CP/PP=1，服务模型名为 `glm52-ascend-offload`。使用独立的 `RUNTIME_DIR`（默认 `/tmp/sglang-glm52-offload-runtime`）和 `LOG_DIR`；需要与其他服务并行时显式更换 `SERVER_PORT`、`DIST_PORT` 和设备列表。

### GLM-5.2 MTP

`ENABLE_MTP=1` 开启同 checkpoint 内的单层 NextN。沿用 DSV4 offload 的执行方式：主模型的 routed experts 继续卸载；draft 权重全部常驻 NPU，不占用主模型 Host/Hot 池；主模型的 target verify 走 packed 路径，关闭 Hot 直算。Verify 不触发整层 prefill 预取，draft 也不参与主模型的缓存维护。

```bash
MODEL_PATH=/models/GLM-5.2-W8A8-with-MTP \
ENABLE_MTP=1 HOST_POOL_GIB=192 CACHE_RATIO=0.25 \
bash scripts/moe_offload/serve_glm52_tp4.sh
```

W4 主模型使用同一开关，Host 预算仍按上面的实际格式设置。脚本固定 EAGLE `steps=2 / topk=1 / draft_tokens=3`，target 与 draft 均使用 `ascend` attention、同一个 `MODEL_PATH`。默认 `ENABLE_MTP=0`。

默认 `MTP_DRAFT_QUANTIZATION=unquant`，对应仓库 GLM-5.2 NPU 部署示例中的不量化 draft；它不会继承主模型的 W8/W4 量化。必须使用实际包含 MTP 权重的 checkpoint。启动前的 `glm52_mtp_config.py` 会读取配置及 safetensors 的名称/头部信息，在分配主模型和 Host 池前检查 MTP 层是否存在，并拒绝将量化 draft 权重按浮点权重加载。若 draft 本身按 ModelSlim W8A8/W4A8 导出，显式设置 `MTP_DRAFT_QUANTIZATION=modelslim`，由 draft 层自己的量化描述选择 scheme。量化 draft 的 routed experts 支持 `W8A8_DYNAMIC` / `W4A8_DYNAMIC`，其余可量化 linear 支持 `FLOAT` / `W8A8_DYNAMIC`；静态 `W8A8` 未纳入该启动预检范围。暂不支持混合量化且 routed experts 为 FLOAT 的 draft、量化 `eh_proj`，也不支持本脚本中指定另一个 draft checkpoint。

不量化 draft 还需要额外 NPU 显存；`HOST_POOL_GIB` 仅预算卸载专家的 Host 存储，不能代替这部分显存。MTP 接入已做 CPU 契约与启动参数检查，尚未进行 GLM 的 NPU 整模型精度、accept length、Graph replay 或性能验收。实机上分别以 `ENABLE_MTP=0/1` 运行 `test_requests.py` 与 `test_requests.py --bench`，检查短请求、空 DP rank、chunked prefill→verify、缓存维护和并发 1/4/8，再比较相同 checkpoint 的输出与性能。

默认 [initial_profile_glm52.json](initial_profile_glm52.json) 的 `layers` 为空，表示所有专家初始热度为零，按既有确定性规则初始化 Hot；它不是采集得到的访问统计，也不代表推荐的热点分布。可用 `OFFLINE_STATS_PATH=/path/to/glm52-profile.json` 覆盖为该 GLM 模型的实测 profile，不应复用 DeepSeek V4 的 `initial_profile.json`。脚本生成的 `offload.json`、profile 和日志路径都会写入启动输出。

目标 NPU 验收前，先在相同 CANN / torch_npu / MemFabric 环境运行单设备格式与 GMM 对照：

```bash
MOE_NATIVE_ACL_DIR=/tmp/sglang-glm52-offload-runtime/native_acl \
python3 test/manual/test_ascend_glm_offload_formats.py --device 0
```

该脚本使用合成 checkpoint 分片，检查 resident 与 offload Host repack / ACL reload 后的投影结果，覆盖 W8、W4 per-channel 和 per-group；不需要下载模型。它不替代四 rank 的模型加载、长短请求、空 DP rank、prefill→decode、缓存维护和 Graph replay 验证。整模型验收还应与关闭 offload 时的同一量化 checkpoint 对照输出和精度，再评估 TTFT/TPOT。

## GLM-5.2：单机 P 卸载 + 两机 D

[serve_glm52_pd.sh](serve_glm52_pd.sh) 提供 `prefill`、`decode`、`router` 三个入口。原混部脚本和未指定 `mode` 的 offload 配置保持原有行为。新模式只卸载 P 主模型的全部 routed experts；dense、attention、shared experts 和 MTP draft 常驻。D 不配置 offload。

并行划分沿用 [GLM-5.2 A3 PD 手册](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/glm_5_2#prefill-decode-disaggregation-deployment)，D 扩为两台组成一个实例。脚本使用当前分支的 `--attn-dp-size`，与手册的 `--dp-size N --enable-dp-attention` 对应。

| 角色 | 机器 / die | TP | Attention DP / TP | MoE EP / TP | 专家通信 |
| --- | --- | --- | --- | --- | --- |
| P | 一台 A3 / 16 | 16 | 4 / 4 | 16 / 1 | Ascend Ops AllToAllV |
| D | 两台 A3 / 32 | 32 | 32 / 1 | 32 / 1 | DeepEP low_latency |

P 复用现有双 bank 逐层预取及 W13/W2 独立等待，关闭图和 overlap schedule；D 保留 Decode Graph。P 使用当前已接入 offload 的 Ascend Ops 后端，未切换为手册的 DeepEP normal。当前方案不启用 KV cache sharding，P Attention TP4 与 D Attention TP1 的 KV 交接沿用现有 PD 传输实现。

### 启动

三台机器需要相同版本的代码和兼容依赖，checkpoint 内容和 MTP 开关一致。在每台机器设置以下环境变量；`COMM_IFACE` 是**本机**承载通信地址的真实网卡，不能照抄混部脚本中的 `lo`：

```bash
export MODEL_PATH=/models/GLM-5.2-ModelSlim
export P_IP=10.0.0.11
export D0_IP=10.0.0.12
export COMM_IFACE=eth0
export ENABLE_MTP=0
```

先启动 P，然后在 D0、D1 上分别启动同一 Decode 实例：

```bash
# P 主机
HOST_POOL_GIB=48 bash scripts/moe_offload/serve_glm52_pd.sh prefill

# D0 主机
NODE_RANK=0 bash scripts/moe_offload/serve_glm52_pd.sh decode

# D1 主机
NODE_RANK=1 bash scripts/moe_offload/serve_glm52_pd.sh decode

# P/D ready 后，在可连接 P、D0 的主机运行；此角色无须模型与 NPU 环境
bash scripts/moe_offload/serve_glm52_pd.sh router
```

Router 只登记 `D0_IP`，D1 通过共同的 `D0_IP:DIST_PORT` 加入 32-rank 实例，不是第二个 Decode 服务。默认 P HTTP 为 8000、bootstrap 为 8998、D HTTP 为 8003、D rendezvous 为 5000、Router 为 6688；可分别用 `P_PORT`、`BOOTSTRAP_PORT`、`D_PORT`、`DIST_PORT`、`ROUTER_PORT` 覆盖，关联节点须保持一致。

三台 `ASCEND_MF_STORE_URL` 默认均为 `tcp://$P_IP:24707`（可用 `MF_STORE_PORT` 改端口或显式覆盖 URL）。这个 KV store 与 D 的 rendezvous 是两个独立端口。当前 PD 默认使用 SDMA；安装版本支持时可设置 `ASCEND_MF_TRANSFER_PROTOCOL` 选择其他协议。部署前需确认 A3 fabric 支持所选跨机协议，普通 IP 可达不等于 SDMA 可达。

脚本默认使用本机 16 个 die；可用 `ASCEND_RT_VISIBLE_DEVICES` 指定另一个含 16 个不同设备 ID 的列表。P 会复用既有 MemFabric 搬运扩展和 native ACL 构建流程。Ascend 环境安装位置不同，可设置 `CANN_ENV`、`ATB_ENV`、`CUSTOMIZE_ENV`、`CUSTOM_TRANSFORMER_ENV`。`RUNTIME_DIR` 放构建缓存，`LOG_DIR` 放本次日志和 P 的 `offload.json`，默认均在 `/tmp` 下；两者须为绝对路径。切换依赖版本时使用新的 `RUNTIME_DIR`。

### 配置与容量

P 自动生成最小配置，不读取混部的热点 profile：

```json
{
  "mode": "prefill_only",
  "cache_ratio": 0,
  "host_pool_gib": 48,
  "native_extension_directory": "/tmp/sglang-glm52-pd-prefill-runtime/native_acl"
}
```

`prefill_only` 自动选择全部 routed MoE 层，零 Hot 缓存，也不创建 LFU 或 Decode 缓存维护状态。默认执行槽由 runtime 计算为 `2 × 本 rank 专家数`；256 experts / EP16 时是 32 槽。脚本会从 checkpoint 核验专家数，若显式设置 `EXECUTION_SLOTS`，不得低于双 bank 所需容量。该模式只接受纯 Prefill 角色，误用于 Decode 会在装载前失败。

代码使用两个执行策略：`CachedExecution` 保留混部缓存行为，`PrefillOnlyExecution` 管理 P 的初始化和 forward。两者共用 Host 装载、格式转换和 `LayerPrefetchPipeline`，不修改模型、dispatcher 或 PD 协议。P 要求开启 DP 调度同步，不能设置 `SGLANG_SCHEDULER_SKIP_ALL_GATHER=1`；全局空闲由 scheduler 跳过，局部空 rank 仍参与全局 Prefill。

`HOST_POOL_GIB=48` 是 **每 rank** 的初始预算，P 整机预留 768 GiB，此外还需 checkpoint 装载、常驻对象和系统内存。GLM-5.2 的 75 个 routed MoE 层在 W8 下仅权重本体约 42.1875 GiB/rank，元数据与分配对齐另计；loader 会按实际格式检查预算。W4 不能仅用 W8 总预算除以二，尤其要计入 per-group 元数据。D 无此 Host 池。

保守起点为 `CONTEXT_LENGTH=16384`、P `CHUNKED_PREFILL_SIZE=8192`、P `MAX_RUNNING_REQUESTS=16`、D `MAX_RUNNING_REQUESTS=128`、`MEM_FRACTION_STATIC=0.8`。P 的 chunk 参数会按 Attention DP4 归一为每 DP 2048。每一个 chunk 都重新搬运全部 routed 专家；双 bank 重叠搬运与计算，不减少搬运字节数。先测 P 吞吐与 D 等待比例，再增加 chunk 或并发，不能直接套用手册的大 chunk。

MTP 默认关闭，因为 checkpoint 可能不包含 NextN 权重。需要时在 **P、D0、D1 同时**设置 `ENABLE_MTP=1`；脚本复用 `glm52_mtp_config.py`，先读取配置和 safetensors 头部验证 MTP。默认 `MTP_DRAFT_QUANTIZATION=unquant`，必须确实包含浮点 draft。参数沿用 PD 手册：P NEXTN steps=1 / topk=1 / draft tokens=2，D steps=3 / topk=1 / draft tokens=4；draft 始终常驻。若实际使用受支持的量化 draft，三台统一显式设置 `MTP_DRAFT_QUANTIZATION=modelslim`。

### 预检与设备验收

给任一入口增加 `--dry-run`，只做本地配置 / MTP 头部检查，并显示 P JSON 与完整启动命令；不加载模型、不 source CANN、不构建扩展、不创建日志或配置文件：

```bash
bash scripts/moe_offload/serve_glm52_pd.sh prefill --dry-run
NODE_RANK=0 bash scripts/moe_offload/serve_glm52_pd.sh decode --dry-run
NODE_RANK=1 bash scripts/moe_offload/serve_glm52_pd.sh decode --dry-run
bash scripts/moe_offload/serve_glm52_pd.sh router --dry-run
```

CPU 检查和 dry-run 不代表三机 NPU 部署已验证。实机验收先跑相同 checkpoint 的常驻 PD 基线，再启用 P-only，覆盖空 DP rank、零收包、多个 chunk、长短 prompt、并发及 MTP 开关；对比 logits / 首 token / 端到端输出。确认 D0/D1 同属一个实例、D 无权重 H2D，并观察 Host/HBM 峰值、LOCAL offload 与 PD TransferEngine 共存、异常退出清理、TTFT、P tokens/s 和 D TPOT。P 每个完整 chunk 的 W8 权重本体搬运约 675 GiB/整机，H2D 与 PD KV 传输的带宽争用需要单独测量。

本地隔离外围 serving / NPU 依赖后，配置与目录 19 项、纯 P 生命周期 13 项、原预取/dispatcher 31 项和 GLM 格式契约 17 项通过；设备拷贝与 collective 使用测试边界。新增的两个无 usage 计数测试依赖 Triton interpreter，本地主机缺少 Triton，尚未执行。可在目标环境先运行 `python3 test/manual/test_ascend_offload_fixed_slots.py --metadata-only`，它已加入 `TRACK_USAGE=False` 且统计指针为空时的真实 NPU 编译/计数检查，不加载模型权重。
