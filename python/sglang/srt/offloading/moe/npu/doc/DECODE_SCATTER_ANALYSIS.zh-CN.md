# Decode 中 ScatterElements 的用途与耗时分析

> 历史设计记录：本文中的 wave 调度、分组配置及相关接口已在分层分支删除。当前执行方式和配置迁移见[分层实现说明](../../../../../../../scripts/moe_layer/README.md)。缓存策略设计仍供参考。

代码基线：`codex/moe-offload-dev-20261027` / `531e228a`。

第 1–3 节解释优化前的七次 scatter；第 4 节记录 `61bec1e8` 的共享 slot 压紧，
第 5 节提供实机验证命令；第 6 节记录当前 P0-1、P0-2 和输出回填回滚。

以下文件路径均相对于仓库根目录；行号对应上述代码基线。每处定位同时列出
类名、函数名和关键语句，后续代码行号变化时可按函数或语句搜索。

**这 7 次 ScatterElements，主要是在重复整理六份权重 component 的复制地址。
它们处理的是小型 INT64 元数据，但每次都是独立操作，并且串行挡在权重搬运前。**

这里的七次对应 W8A8、单个 wave 不超过 256 个专家的搬运路径。代码能解释
“为什么有七次、为什么总开销会累加”；单次为什么达到截图中的约 75.9 μs，
还需要结合实际执行类型和输入规模判断。

## 1. 第一次：检查目标 slot 是否被重复写入

代码位置：

- 文件：`python/sglang/srt/offloading/moe/npu/transfer.py`
- 函数：`MemFabricDeviceTransferOps._mask()`，定义在第 419 行。
- 相关逻辑：第 442–445 行；其中第 444 行调用 `destination_writes.scatter_add_()`。

```python
bindings.destination_writes.zero_()
indices = torch.where(destination_in_range, plan.destination_slot + 1, 0)

bindings.destination_writes.scatter_add_(
    0, indices, plan.valid.to(torch.int64)
)

invalid |= (bindings.destination_writes[1:] > 1).any()
```

每个有效复制条目给目标 slot 的计数加 1。例如，忽略第 0 项 sentinel 后：

```text
目标 slot： [0, 1, 2, 3]
有效条目：  [1, 0, 1, 0]
统计结果：  [1, 0, 1, 0] → 没有重复写入
```

它是正确性校验，不搬权重，也不计算命中。

当前 decode 的 `destination_slot` 在绑定时由 `arange` 生成，本身就是固定、
不重复的序列。初始化代码位置：

- 文件：`python/sglang/srt/offloading/moe/residency.py`
- 函数：`TensorResidencyOps._bind_wave()`，定义在第 168 行。
- 相关逻辑：第 203–207 行，使用 `torch.arange(bound.execution_start,
  bound.execution_start + size, out=buf["plan"].destination_slot)` 初始化目标 slot。

因此，这个通用检查在当前固定 decode 路径中，有改为绑定期校验的优化空间。
需要通过专用路径保证该约束，不能直接删除通用接口的保护。

## 2. 后六次：分别整理六个 component 的目标地址

W8A8 的完整专家 bundle 包含：

```text
w13_weight、w13_weight_scale、w13_weight_offset
w2_weight、 w2_weight_scale、 w2_weight_offset
```

component 定义位置：文件 `python/sglang/srt/offloading/moe/npu/w8a8_format.py`，
函数 `NpuV4W8A8Format.__init__()`；第 74–77 行遍历 `w13`、`w2` 两个 projection，
第 90–106 行分别定义 `weight`、`weight_scale`、`weight_offset` 三种 component。

每个 component 都执行一次下面的逻辑，省略参数的示意代码如下：

```python
# 查源地址，选择 Host 或 hot cache
host_ptr = host_table.gather(...)
hot_ptr = hot_table.gather(...)
src.copy_(torch.where(selected, torch.where(from_host, host_ptr, hot_ptr), 0))

# 查目标地址
destinations = destination_table.gather(...)

# 压紧目标地址
dst.zero_()
dst.scatter_add_(
    0, packed_index,
    torch.where(selected, destinations, 0),
)

# 实际搬运
group_pack_copy(...)
```

上述逻辑的代码位置：

- 文件：`python/sglang/srt/offloading/moe/npu/transfer.py`
- 函数：`MemFabricDeviceTransferOps.materialize_into()`，定义在第 449 行。
- 第 485–487 行：在 component 循环外计算共享的 `packed_indices`。
- 第 489–492 行：遍历每个 component 及其 batch。
- 第 493–505 行：查询 Host / hot cache 源地址，并写入 `src`。
- 第 508–511 行：查询 execution buffer 的目标地址。
- 第 515–518 行：清零 `dst`，再通过 `dst.scatter_add_()` 压紧目标地址。
- 第 520–522 行：调用 `group_pack_copy()` 执行该 component 的实际搬运。

命中时搬运 hot cache → execution buffer（D2D），未命中时搬运
Host master → execution buffer（H2D）；两者都需要准备地址。

例如，只需要第 0、2 个专家：

```text
selected       = [ 1,  0,  1,  0]
目标地址       = [A0, A1, A2, A3]
packed_index   = [ 0,  0,  1,  1]
scatter 写入值 = [A0,  0, A2,  0]

得到 dst       = [A0, A2, 0, 0]
```

换到 scale component，又需要：

```text
目标地址 = [S0, S1, S2, S3]
得到 dst = [S0, S2, 0, 0]
```

**选择的是同一批专家，但每份 component 的地址不同，所以当前代码重复做了
六次地址压紧。** `packed_index` 的 `cumsum` 已经在循环外共享，重复的是后面的
地址 gather、where、scatter 和复制。

压紧的是地址列表，不改变专家的物理目标 slot。使用 scatter-add 是为了让无效
条目的零不覆盖有效地址；不能简单替换成覆盖式 scatter，因为索引可能重复。

## 3. 为什么这么小的工作也可能很贵

从代码能确定的有三点：

- **它用了通用的 INT64 scatter-add。** `dst` 保存的是 64 位指针值，并非普通
  浮点权重。地址数组的类型定义位于
  `python/sglang/srt/offloading/moe/npu/transfer.py` 的
  `MemFabricTransferBindings.__init__()`，第 328–331 行；第 331 行将目标指针数组
  分配为 `torch.int64`。这个操作需要支持重复索引的累加语义，而不是简单连续写入。
  官方语义参考：CANN Community Edition 9.1.0 API 文档，`aclnnScatterAdd`。
- **无效条目仍参与调用。** 上例中的零也被传给 scatter，且索引重复。全空 mask
  时，`packed_index` 甚至全为零；代码仍会调用 scatter。零值不会自动让整次
  设备任务消失。
- **六套准备与搬运严格串行。** 当前顺序如下：

```text
校验 scatter
    ↓
准备 weight 地址 → scatter → copy
    ↓
准备 scale 地址  → scatter → copy
    ↓
准备 offset 地址 → scatter → copy
    ↓
……另外三个 component……
    ↓
计算流才能执行 FFN
```

所以它的代价来自重复的小任务及其依赖链，不能用专家权重有多大来解释。
假设 wave 有 32 个专家，每个地址数组仅 `32 × 8 = 256` 字节。

如果七次实际都约为 75.9 μs，那么仅 scatter 的累计耗时就约为：

```text
7 × 75.9 μs ≈ 531 μs / wave
```

这是条件估算，不代表已经测得七次的总时长；还没算周围的 Gather、Where、清零
和真正的权重复制。固定图 replay 也不意味着这些设备任务会消失。

目前不能把单次 75.9 μs 确定归因于原子冲突或 AI CPU 回退。应先看该任务的
Task Type、INT64 输入 shape，以及不同 mask 下的耗时。如果从几十个元素变到
几百个元素，耗时仍接近常数，更支持固定执行开销占主导；若随重复索引情况明显
变化，才更支持冲突处理开销的解释。

**最直接的优化方向，是减少元数据操作的重复次数：固定目标 slot 的唯一性在
绑定期验证；有效 slot 顺序只压紧一次，再为六个 component 查地址，或者融合成
一个专用地址准备算子。** 当前用通用 scatter-add 实现每份地址列表的压紧，给
decode 引入了可以进一步压缩的开销；具体收益仍需实机测量。

## 4. 已实现：共享 slot 压紧，保留复制接口和动态校验

本节记录的 `61bec1e8` 保持 `MemFabricDeviceTransferOps._mask()` 的动态校验不变，
也保持 `group_pack_copy(src, dst, lengths, scan, groups, packed, device)` 接口不变。
当前 decode 又通过第 6 节的固定目标 binding 去掉了运行时目标校验；通用路径不变。

优化后的代码位置（本节行号对应优化后的实现）：

- 文件：`python/sglang/srt/offloading/moe/npu/transfer.py`
- `MemFabricTransferBindings.__init__()` 第 342–348 行：为每个 batch 预分配
  `packed_destination_slots`。这些 INT64 张量由 binding 独占，额外有效字节数为
  `8 × capacity`，通过 `_tensors` 纳入 `allocations()` 记账。
- `MemFabricDeviceTransferOps._mask()` 第 427–455 行：保留来源、目标、generation、
  有效条目计数、重复目标和错误状态检查；其中第 452 行仍执行一次校验 scatter。
- `MemFabricDeviceTransferOps.materialize_into()` 第 493–507 行：每个 batch 只压紧
  一次 `destination_slot + 1`；零表示地址表的 sentinel，而不是物理 slot 0。
- 同一函数第 528–533 行：每个 component 按共享 slot 列表 gather 到自己的 `dst`，
  覆盖完整目标数组及零尾部，替代逐 component 的 `dst.zero_()` 和 `dst.scatter_add_()`。

```text
动态校验 scatter（保留）
    ↓
每个 batch：共享 slot 压紧 scatter（一次）
    ↓
六个 component：各自查询源地址、gather 目标地址、调用原有 copy
    ↓
完整 bundle 搬运结束后记录完成事件
```

`src` 和 `groups` 保持原始输入顺序，目标地址按有效输入顺序压紧，继续满足
原生接口的 `src[i] → dst[j]` 语义。每个 component 的 `groups`、`packed` 等
可变原生 scratch 仍独占；全空 mask 不通过 CPU 分支跳过调用。

新增测试：`test/registered/unit/offloading/test_memfabric_compaction.py`。
测试运行真实 `materialize_into()`；仅设备分配、事件及原生 copy 使用 CPU 替身，
按原生索引约定复制真实字节，并与未压紧的逻辑映射比较。

使用同一测试夹具对比优化前 `5182d4d5` 与当前实现，得到以下 CPU ATen 调用计数：

| 场景 | 优化前 scatter_add_ | 优化后 scatter_add_ | copy 调用 |
| --- | ---: | ---: | ---: |
| 32 个专家，全有效 | 7 | 2 | 均为 6 |
| 32 个专家，稀疏有效 | 7 | 2 | 均为 6 |
| 32 个专家，全空 | 7 | 2 | 均为 6 |
| 257 个专家，跨两个 batch | 13 | 3 | 均为 12 |

对六个 component，设固定 batch 数为 B，scatter 调用由 `1 + 6B` 变成 `1 + B`。
这些是 CPU 执行真实 lowering 时的调用计数和字节正确性结果，
不是 NPU ScatterElements 的实测数量或耗时，也不代表 TPOT 已改善。

## 5. NPU 实机验证

先在仓库根目录运行 CPU 契约测试（不加载模型）：

```bash
PYTHONPATH="$PWD/python:$PWD/test/registered/unit/offloading${PYTHONPATH:+:$PYTHONPATH}" \
  python -m unittest -v \
  test_memfabric_compaction \
  test_fixed_decode_transfer \
  test_moe_decode_layer_plan \
  test_npu_backend.TestAddressCatalogMetadata
```

更新代码后重启原服务，沿用原模型、EP/TP、execution_slots 和缓存配置，让 decode
graph 重新捕获。已经捕获的旧 graph 不会因为磁盘代码更新而自动采用新实现。

以下命令以服务地址 `http://127.0.0.1:8818` 为例；基线与优化版本应使用相同请求、
预热过程和采样配置。先预热，再分别采集 prefill 与 32 个 decode step：

```bash
BASE_URL=http://127.0.0.1:8818

curl --fail --silent --show-error "$BASE_URL/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Explain how a compiler works.","sampling_params":{"temperature":0,"max_new_tokens":64,"ignore_eos":true}}' \
  > /tmp/moe-shared-slot-warmup.json

curl --fail --silent --show-error "$BASE_URL/start_profile" \
  -H 'Content-Type: application/json' \
  -d '{"output_dir":"/tmp/moe-shared-slot-profile","num_steps":32,"profile_by_stage":true,"activities":["CPU","GPU"],"record_shapes":true,"with_stack":false}'

curl --fail --silent --show-error "$BASE_URL/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Explain how a compiler works.","sampling_params":{"temperature":0,"max_new_tokens":64,"ignore_eos":true}}' \
  > /tmp/moe-shared-slot-profile-response.json
```

当前服务接口使用 `GPU` 活动名称；NPU 后端会将对应 profiler 活动映射为 NPU。
`profile_by_stage=true` 将 prefill 与 decode 分开；足够长的请求触发 32 个 decode
step 后自动停止采样。这里采集的是请求执行阶段，不是 `PROFILE_CAPTURE=1` 的建图时间线。

在同一 rank 的 decode 时间线中，用一次复制计划就绪到完整 bundle 就绪的事件依赖
定位搬运段；对六个 component、单 batch 的 wave，核对：

- 当前固定 decode：目标校验 scatter 消失，只剩共享 slot 压紧 scatter 一次。
  `61bec1e8` 基线仍有校验和共享压紧各一次；hot.update 的动态校验继续保留。
- 命中/source 查询在 dispatch/layout.bind 后按层执行一次；各 wave 读取对应切片。
- 六个 component 的 `OffloadGroupPackCopyOps` 调用保留；具体设备任务数量以 profiler 为准。
- 目标地址 gather、源地址选择及错误屏蔽正常，输出与基线一致。
- 同时记录搬运段总时长、实际 copy 时长、计算流等待，以及完整 decode step 时长。
  普通 step 与缓存维护 step 分开统计，不能按整张 trace 的 scatter 总数直接除以 wave 数。

关闭 profiler 后再比较 TPOT。NPU 实际算子数、graph replay 正确性和性能收益待实机验收。

## 6. 当前实现：P0-1、P0-2 与输出回填回滚

已通过 `1fb14334` 回滚 `fdab5c61` 的输出 `index_copy_` 改动，恢复原有输出
MaskedFill / IndexSelect / Where 路径。用户实测没有收益，因此此次只推进命中计划
与固定目标校验，保留 `61bec1e8` 的共享 slot 压紧。

### P0-1：每次调用一层时，只生成一次整层来源计划

例如一层有 96 个本地专家，每个 wave 容纳 32 个专家。原先三个 wave 各自查询
key_to_slot、slot_valid、slot_to_key、slot_generation，再选择 Host/hot 来源。
现在 dispatch 得到本层全部 counts 后，一次处理 96 个专家；三个 wave 分别使用
`[0:32]`、`[32:64]`、`[64:96]` 的固定视图。命中查询的数据总量没有减少，
减少的是重复的算子提交与依赖链；有 hot slots 时，这四种查表的调用总数从
`4 × wave 数` 降到每层 4 次。这里仅统计命中计划查表，不包含激活整理的 IndexSelect。

代码位置（本节对应 P0 修改后的源码，路径均相对仓库根目录）：

- `python/sglang/srt/offloading/moe/programs.py`：`DispatchOnceProgram.run()`
  第 67–76 行，在 dispatch/layout.bind 后 prepare，通过 finally 关闭本次计划；
  `_run_waves()` 第 101–110 行让各 wave 显式使用 prepared plan。
- `python/sglang/srt/offloading/moe/npu/experts.py`：`build_npu_layer_programs()`
  第 632–639 行启用本层 counts adapter，并指定 compute stream scope。
- `python/sglang/srt/offloading/moe/residency.py`：`_bind_wave()` 第 222–240 行
  创建来源与 generation 的持久切片；`_allocate_layer_sources()` 第 243 行
  预分配整层元数据；`prepare_decode_layer()` 第 344 行生成本次计划；
  `_prepare_sources()` 第 471 行执行统一查表和访问计数。
- 同文件 `resolve_execution()` 第 437–469 行，每 wave 仍刷新最新错误屏蔽、
  valid/kind/count、execution_valid，并记录当前计算前沿；`publish_execution()`
  第 537 行仍等待完整复制完成，再检查错误并发布有效权重。

计划每次层调用都会刷新，graph replay 也重新执行捕获的 tensor 操作，不跨 step
缓存旧命中结果。临时元数据的物理存储在绑定时分配并计入资源预算，切片不重复记账。
当前层关闭前不能更新 hot set；前一 wave 的 reader 等待和复制依赖没有移到层首。

成功执行时，整层访问次数与原逐 wave 累加一致。失败路径的时序有变化：后续 wave
的非法映射可能在层首更早被发现；cache stats 也在层首记录路由需求，因此不能把
失败调用中的命中数解读为“已经成功搬运或执行的专家数”。错误后的复制与输出仍受屏蔽。

### P0-2：由 decode binding 持有固定目标，初始化时证明合法

decode 的每个 wave 都绑定 `execution.view(0, count)`，目标 slot 固定为
`(0, 1, ..., count - 1)`。绑定时检查长度、整数类型、范围和唯一性，之后实际复制
使用 binding 独立持有的 slot 索引。它不是简单关闭检查：即使 plan 的目标内容
后来被改成重复或越界，固定路径也不能被重定向；plan 字段仍需满足 shape/dtype/device 契约。

- `python/sglang/srt/offloading/moe/npu/backend.py`：`NpuMoEBackend.finalize()`
  内第 217–225 行，将固定 slot 顺序与 FFN bank/validity 的顺序绑定。
- `python/sglang/srt/offloading/moe/npu/transfer.py`：
  `MemFabricTransferBindings.__init__()` 第 313–345 行验证并保存固定目标；
  `MemFabricDeviceTransferOps._mask()` 第 471–490 行只让动态 binding 执行目标
  范围/重复检查；`materialize_into()` 第 524–529 行使用固定索引。
- 同文件第 540 行的共享 slot 压紧 scatter 仍保留。六份 component 仍需各自
  gather 地址和 copy；来源范围、generation、count、共享错误状态检查也保留。
  `hot.update` 不使用固定目标，继续保留运行时目标校验。

对六份 component、单 batch 的 decode wave，CPU 真实 lowering 的 scatter_add_
调用数由共享压紧版本的 2 次降为 1 次；跨 B 个 batch 时由 `1 + B` 降为 `B`。
这不是 NPU kernel 数或耗时承诺。

新增回归测试位于 `test/registered/unit/offloading/test_moe_decode_layer_plan.py`
和 `test/registered/unit/offloading/test_fixed_decode_transfer.py`，覆盖计划差分、
动态输入的 CPU 算子 trace 重放、重复层调用、晚期错误、generation 复核、固定目标
篡改、跨 batch 压紧和字节复制正确性。CPU trace 不替代真实 NPU graph replay。

本次本地验证：15 个 offloading CPU 测试模块与 launcher 的 4 项 Python preflight，
共 214 项通过，未跳过；另外依赖 POSIX shebang 的 Bash 启动测试不在本次 Windows
执行范围内。ruff、isort 和 git diff 空白检查通过。回归中修正了旧测试的跨层 LFU
预期、统计开关与 fake replay 协议，未改变对应生产策略。
