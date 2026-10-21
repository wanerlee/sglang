# Cache 调用链实现与 H2D 交接表

接口更新：已合并同事 `59cbb5d`；cache 执行统一接入公共 `ExpertWeightAccess` / `CacheUpdateCoordinator`。算法参数及层内配对不变，详见 [合并适配设计](MERGE_ADAPTATION_DESIGN.zh-CN.md)。

适用分支：`codex/moe-offload-dev`。

本轮全部修改概览见 [修改总结](CHANGE_SUMMARY.zh-CN.md)。

当前已实现 cache 调用方：初始化、step 生命周期、dispatch 后计数、按实际需求准备 execution bank、LFU 更新、slot 计划、复制完成后发布、prefill 借池前快照及回填后恢复。decode 通过公共 access 调用 `materialize_into`，prefill pipeline 调用 `materialize_host_into`。

CPU 测试使用延迟执行的测试搬运器和两分量专家权重，验证实际权重复制与计算结果。另新增 NPU 完整流程脚本，使用真实 stream/event 和小矩阵权重，见第 6 节；本机没有 NPU，尚未实机验证。测试实现只位于测试目录，不作为 serving fallback。尚未提供生产 NPU transport、原生专家算子或 EP dispatcher；也未验证完整 NPU graph 或重叠性能。

## 1. 代码落点与调用路径

| 文件 | 本次实现 |
| --- | --- |
| [cache_runtime.py](cache_runtime.py) | `CacheRuntime` 实现 `BoundMoERuntime` 的 cache 生命周期和捕图 hook；`make_execution_request` 接 D 后统计，公共 `DecodeLocalExpertProgram` / `ExpertWeightAccess` 串联 bank 准备和原生 F。 |
| [cache_residency.py](cache_residency.py) | `TensorResidencyOps` 实现实际需求计划、保留幸存 slot、分配替换位置、失效旧读者映射及就绪后发布。 |
| [cache_metadata.py](cache_metadata.py) | 保持原职责：计数、策略状态和元数据快照；LFU 收尾由公共 coordinator 统一调用。 |
| [ModelRunner](../../model_executor/model_runner.py) / [BaseOffloader](../base.py) / [MoEOffloaderV2](offloader.py) | 新增 eager step scope 的实际调用链；decode graph 分支直接返回，不重复执行 eager step hook。 |
| [test_cache_runtime.py](../../../../../test/registered/unit/offloading/test_cache_runtime.py) | 测试 backend、延迟搬运、D/F/C 程序及框架调用链验证。 |

```text
backend.finalize
  -> CacheRuntime(... backend bindings / streams / programs ...)
  -> 组装 decode_program = DispatchOnceProgram(..., DecodeLocalExpertProgram(...))
  -> runtime.initialize(model_id)
       offline TopK -> prepare_refill -> materialize_into -> wait -> commit
  -> 返回 runtime，创建现有 owner

offloader.bind_model_runner
  -> runtime.install_model_hooks
       安装模型程序 + 注册 capture head/tail

ModelRunner.forward -> offloader.forward_scope
  -> owner.begin_forward：阶段切换 / 前序完成依赖
  -> 写固定地址的统计开关
  -> eager：ModelRunner 调用 eager_step_scope 的 begin/end
     graph：已有 capture head/tail 记录 begin/end，replay 不重新调用 Python
  -> 每层一次 D -> 各 wave 的 LocalRoutes -> 计数 -> ExecutionRequest
       -> access.stage -> wait_ready -> native F -> release_after -> 一次 C
  -> 完整 step 末尾：coordinator 累计 LFU、选择、finish_step
       -> access.apply_update -> prepare_update -> 搬运 -> commit_update
  -> 记录本次提交的新完成事件 -> owner.finish_forward
```

decode 使用 compute -> copy -> compute 依赖串行保护复用内存，不做下一 wave 预取。prefill hook 已接到公共 pipeline，真实优化流水由同事提供。传输 API 仍为异步，不用每步 host synchronize。初始化和阶段切换允许主机等待。

阶段判断不依赖 TBO 才提供的 `global_forward_mode`：结合 `is_extend_in_batch` 与全局 token 数推导。在全局 prefill 中，本地 IDLE 或 DECODE rank 都进入 prefill；在有效全局 decode 中，本地 IDLE rank 仍统计 dispatch 后的访问并推进周期。优先使用 `original_global_num_tokens_cpu`，避免把图 padding 当成有效 token；全局空闲时保持当前阶段，不触发无谓回填。

Eager 的周期由 runtime 中一个 host 整数跟随完整有效 forward 推进，无需读取设备 tensor。`forward_scope` 必须包围一次实际 eager 执行或 graph replay；成功提交后只推进一次，prefill、全局 IDLE、capture/warmup 不推进，失败后沿用 owner 的停止执行约束。Host 周期跨 prefill 保持，设备 LFU 周期随快照恢复，因此 eager 与 graph 交替执行时保持一致。非周期 eager 调用 `cache_updates.emit_step_end(context, adjust=False)`，累计 usage 后直接结束，不排序、不生成热池计划、不提交空搬运、不重建映射。到期但无合格交换时仍按规则衰减。图路径保持默认 `adjust=True` 的固定操作序列，设备门控实际交换；此轮没有宣称已消除图路径的非周期排序开销。

## 2. 已落地的调用方数据约定

下列约定已用于本次代码及测试，同事实现 transport 时可以照此适配。如果同事已有不同 ABI，应在 binding/transport 适配层转换，不修改 LFU 规则。

### 2.1 DeviceCopyPlan

| 字段 | 类型 / 含义 |
| --- | --- |
| `source_kind` | int64[C]，NONE=0、HOST=1、HOT_SLOT=2。 |
| `source_index` | int64[C]；HOST 是本 rank 的 canonical expert key；HOT_SLOT 是热池 slot ID。 |
| `destination_slot` | int64[C]；索引本次 bindings 指定的目标池。热更新目标为热池，执行加载目标为完整 execution 池。 |
| `valid` | bool[C]；可能有空洞，必须逐条尊重 mask，不能把前 count 项全部当有效。 |
| `count` | int64 标量，等于 valid.sum()；零条目是正常空操作。 |
| `expected_generation` | int64[C]；HOT_SLOT 来源的预期 generation，HOST/NONE 为 0。 |
| `capacity` | Python 静态整数 C；计划 Tensor 在构造时分配，在线更新内容。 |

一次条目表示一个完整专家 bundle，包括所有权重、scale 等必要组件。所有组件就绪才能返回覆盖本次操作的完成依赖。`bindings` 是后端对象，本次未假定它必须是 dict 或特定内存布局。

### 2.2 公共 reservation 与 UpdatePlan

`ExecutionReservation` 和 `UpdatePlan` 复用公共 `copy_plan / bindings / copy_dependencies / keepalive / state`；原私有 CacheUpdatePlan 已删除。

- `resolve_execution(ExecutionRequest, context)` 返回 execution reservation。
- `publish_execution(reservation, weights_ready, context)` 返回执行映射发布完成的 `TransferTicket`。
- `finish_execution(reservation, read_done, context)` 记录最终读者，保护下一次计划与 bank 复用。
- `prepare_update(target_keys, context)` 返回 `UpdatePlan`；`commit_update(plan, weights_ready, context)` 返回热映射发布票据。
- 热更新的 `state` 保存 `slot_keys` 和设备 `publish_enabled`，不再复制一份搬运计划。
- `prepare_refill(slot_keys, context)` 是精确位置初始化/恢复的 cache 生命周期方法，不调用 LFU。

热更新保留幸存专家位置，将新专家按 key 顺序分配到空出的 slot。LFU 层内配对，各层驻留数量不变；同质物理 slot 本身不绑定层。

默认 `TensorResidencyOps` 生成 decode 的 DeviceCopyPlan。真实 prefill backend 在 pipeline 的 access 中提供 HOST_ONLY 的 HostCopyPlan/reservation 和 bank 布局。runtime 可注入扩展的 cache residency，其需保留 `prepare_refill` 和 `allocations`；不新增 host-copy 生产 fallback。

### 2.3 路由与执行 bank

`DecodeLocalExpertProgram` 接收 `make_request(layer, dispatch_output, wave, context)`；后端转换成 `LocalRoutes` 后调用 `runtime.make_execution_request(layer.layer_id, routes, wave, context)`：

- IDs 是该层、本 EP rank 内的 local expert ID，int32/int64；valid 为同形状 bool。
- 每个有效 token-expert 分配计数一次，不能把去重后的专家列表交给计数器。
- 当前 wave 外的分配及 padding 不计数。装载需求在计数之后去重。
- bank 内 lane = local_id - wave.local_start；物理目标 = wave.bank * bank_slots + lane。
- 只准备实际有需求的 lane，空 wave 仍保留原 D/F/C 顺序，但不搬运专家。
- `run_experts(layer, dispatch_output, bindings, context)` 读取已就绪的 bank，返回 `LocalExecution(output, read_done)`；不重复统计或准备权重。read_done 覆盖全部权重读取流，公共 Program 将其 join 回 compute。bindings 提供原生 F 使用的视图/映射信息，不依赖旧签名的 wave。

## 3. 需要与 H2D 同事确认的事项

| 对接项 | Cache 侧已提供 | 同事需提供 / 双方待确认 |
| --- | --- | --- |
| 权重地址与格式 | canonical key、来源类别、目标 slot | `hot_bindings` / `execution_bindings` 的具体对象；host 权重、量化参数、stride、注册句柄和池地址的绑定方式。 |
| 计划格式 | 上述 DeviceCopyPlan 编码、dtype、mask、固定容量 | 是否直接支持；如不同，由哪边做转换。空 mask / count=0 必须可执行。 |
| H2D / D2D 职责 | HOST->bank、HOT_SLOT->bank、HOST->hot 三种调用 | 同事是否同时实现 HOT_SLOT 的 D2D；否则另指定实现者，不能把 cache hit 当作无需准备 bank。 |
| 完成凭据 | 使用 TransferTicket；compute 流 wait 后再读权重或发布 | 返回的 completion 覆盖整个 bundle 和所有内部流；wait_on 仅建立设备依赖，不阻塞主机。 |
| 内存覆盖保护 | 复制前 compute fence；copy 流等待；更新前失效受影响 slot | 同事内部流必须继承传入流的前序依赖；不能绕过等待提前覆盖。 |
| Prefill 复用 | owner 排空前序工作，保存 snapshot；回填读取原 slot_to_key | `borrow_hot_arena` 实际可借范围、prefill 程序、内部流完成依赖；归还时按原 slot 回填完整 bundle。 |
| Graph 动态复制 | 计划地址固定、内容设备侧变化；不在 caller 用 .item() 等读取路由 | NPU `DeviceTransferOps` 是否支持捕图后随输入变化的计划；普通 host copy loop 不能替代。 |
| 权重存储生命周期 | 调用方持有 bindings；ticket 的 keepalive 随提交保活 | host pin/register 由谁管理，backend 关闭时谁释放。 |

驻留表由 cache/runtime 统一发布；transport 不直接修改 `key_to_slot`、`slot_to_key`、`slot_valid` 或 generation。

## 4. 框架 / 原生后端仍需提供的绑定

这些不是等 H2D 联调才能写的 cache 算法，也不能全部归给 H2D 同事：

1. 在真实 `MoEBackend.finalize()` 中构造并返回 `CacheRuntime`，提供真实权重 allocations、streams、bindings 和通过验证的 capabilities。
2. 提供 `install_programs(model_runner, prefill_program, decode_program)`，将程序绑定到目标模型实际 routed-expert 路径。
3. 将所用 EP dispatcher 输出转换成上述 `LocalRoutes`，实现 native F 对 execution bank 的绑定。当前测试 adapter 不适用于直接读取任意 DeepEP/NPU 输出。
4. 提供原 prefill 流式程序，并把其后台工作接回 compute 的完成依赖。当前 runtime 对 decode 的每次复制都等待 ticket；不能替外部 prefill 程序推断未报告的流。
5. 真实 graph backend 仍负责 graph bucket/owner lease 生命周期、固定地址与 workspace 预算、capture/replay 验证。正常图构建在 serving scope 外进行，统计开关默认为 0；不要在 serving scope 内临时 warmup/capture。

当前支持普通 DECODE / IDLE、EXTEND / MIXED 的阶段编排；verify、split prefill 和其他原分支未支持模式仍拒绝，不通过新 runtime 绕过原限制。全局模式和原始 token 数在 host scope 读取；全局有工作而本 rank 无路由时仍推进周期。

## 5. 验证范围

`test_cache_runtime.py` 使用真实 CPU Tensor 存储两分量专家权重，测试搬运器在 `wait_on` 时才执行复制，并在复制前检查新映射尚未发布。

- 完整初始化、4 步 LFU 替换、下一步命中与实际计算结果。
- batch 重复路由累计、多波去重加载、空 wave、零热缓存、全热缓存、多 execution bank。
- 正常 decode、全局空闲、本 rank 空闲、warmup/capture 关闭统计、prefill 覆盖与原位置恢复。
- 搬运失败时不发布新专家；恢复失败时不宣称快照已恢复。
- 执行真实 `ModelRunner._forward_raw` 方法体的 eager / graph 分支，验证 scope 接线及 replay 不重复执行 eager hooks。模型、graph executor 和硬件依赖使用测试替身。
- cache 在线计数、计划与发布不进行 Tensor.item/tolist/cpu/bool 的 host 读取。

这证明调用方流程及 CPU 语义，不证明真实 NPU H2D、完整 graph、分布式正确性或性能收益。

当前本地运行：双方原有测试 108 项，加上 `test_cache_integration.py` 的 9 项读者依赖、调用链和 prefill/恢复衔接测试，合计 117 项 CPU 测试通过；Ruff 检查 / 格式及 isort 通过。测试环境有 PyTorch CPU，没有 torch_npu。

在具备项目依赖的环境中，可单独运行：

```bash
PYTHONPATH=python python -m unittest discover -s test/registered/unit/offloading -p 'test_cache_runtime.py' -v
```

## 6. NPU 小规模完整流程测试

脚本：[test_moe_cache_runtime_npu.py](../../../../../test/manual/test_moe_cache_runtime_npu.py)。在具备匹配的 PyTorch、torch_npu、CANN 和项目依赖的 Ascend 环境，从仓库根目录运行：

```bash
PYTHONPATH=python python test/manual/test_moe_cache_runtime_npu.py -v

# 可选：选择其他可见设备，默认 npu:0。
MOE_TEST_NPU_DEVICE=1 PYTHONPATH=python python test/manual/test_moe_cache_runtime_npu.py -v
```

不下载模型。两个逻辑层，每层 4 个本地专家，热池 4 个 slot，execution bank 2 个 slot。每个专家含两块 8×8 FP32 矩阵；host 权重使用 pinned memory。NPU 通过 execution bank 执行两次矩阵乘法和 ReLU，与独立 CPU 逐 token/expert 参考实现对比。

| 测试 | 检查内容 |
| --- | --- |
| `test_real_h2d_d2d_lfu_and_next_step_hits` | 初始 H2D；miss 的 HOST→bank；命中的 HOT→bank D2D；batch 重复路由及空 wave；4 次 forward 后替换到 `[1,2,5,7]`，generation 和 usage 正确；下一步全部命中。 |
| `test_copy_completion_and_mapping_publication_are_separate` | 只提交并完成复制，暂不 commit：新权重已到达，但映射仍为旧映射，victim 有效位为 false；显式 commit 后才出现新映射。 |
| `test_prefill_borrow_and_exact_refill_restore` | 保存非零 LFU 周期状态，覆盖借出的热池，按原 slot H2D 回填；恢复映射、generation、usage、周期；execution bank 失效；随后 decode 计算仍正确。 |

测试 transport 将设备计划读回 host 后，调用已有 `TorchTransferBackend.copy_into()` 发起真实 `copy_(non_blocking=True)`。HOST 和 HOT_SLOT 均支持；完成事件在所有专家组件复制后记录。completion 的 `wait_on()` 调用真实 NPU event wait，然后在 compute stream 上记录权重/元数据快照，验证“权重可读但新映射尚未发布”的顺序。比较结果时才将快照读回 CPU。

事件语义参考 [TorchNPU 上游 Stream / Event 实现](https://github.com/Ascend/pytorch/blob/master/torch_npu/npu/streams.py)。本测试不依赖“事件一定尚未完成”的时序假设，小权重复制可能很快完成；检查的是实际 wait 调用、等待后的权重和发布前状态，不证明重叠收益。

该测试驱动直接调用 `CacheRuntime` 的初始化、step、D/F/C、快照及恢复接口。它不创建需要完整 graph 能力的 serving owner，也不修改生产 capability gate：`full_decode_graph=False`、`supports_device_plans=False`。主机读计划会增加同步，因此不能用于性能测试或替代同事的动态图流水。dispatch 为单卡脚本路由，借池覆盖为模拟 prefill 写入，非真实模型 / HCCL / prefill 流水验证。

成功应显示三个测试 `ok`、`Ran 3 tests` 和 `OK`。缺少 torch_npu 或设备时，脚本明确打印 `NPU flow was NOT validated`，直接执行返回码为 **2**；断言或执行失败返回 **1**，全部通过返回 **0**。不将 `skipped` 视为 NPU 验证通过。当前只在无 NPU 的本机验证了缺失环境分支、语法、格式及 CPU 参考计算，尚无实机通过记录。
