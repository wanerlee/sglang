# Cache 调用链实现与 H2D 交接表

接口更新：已合并同事 `59cbb5d`；cache 执行统一接入公共 `ExpertWeightAccess` / `CacheUpdateCoordinator`。算法参数及层内配对不变，详见 [合并适配设计](MERGE_ADAPTATION_DESIGN.zh-CN.md)。

适用分支：`codex/moe-offload-dev`。

核对基线：`9da07a0`（2026-10-21）；本轮在此之上补齐真实框架适配，见 [合并适配设计第 14 节](MERGE_ADAPTATION_DESIGN.zh-CN.md#14-本轮补齐我方真实框架适配)。下面区分代码实现与硬件验收；目前没有 NPU 实机通过记录。

本轮全部修改概览见 [修改总结](CHANGE_SUMMARY.zh-CN.md)。

当前已实现 cache 调用方：初始化、step 生命周期、dispatch 后计数、按实际需求准备 execution bank、LFU 更新、slot 计划、复制完成后发布、prefill 借池前快照及回填后恢复。decode 通过公共 access 调用 `materialize_into`，prefill pipeline 调用 `materialize_host_into`。

CPU 测试使用延迟执行的测试搬运器和两分量专家权重，验证实际权重复制与计算结果。NPU 小权重脚本见第 6 节，本机没有 NPU，尚未实机验证。测试实现只位于测试目录，不作为 serving fallback。本轮已提供 DeepSeek V4 模型 installer、Ascend/DeepEP 适配和 CacheBackend 组装，复用已有原生专家算子与 EP dispatcher；实际 NPU transport、prefill pipeline 和物理 bindings 仍由同事交付。

## 1. 代码落点与调用路径

| 文件 | 本次实现 |
| --- | --- |
| [cache_backend.py](cache_backend.py) | 将物理 binding 组装为 CacheRuntime、每层 decode/prefill Program 和真实模型 installer。 |
| [model_binding.py](model_binding.py) | 安装 decoder 层首 hook 和 routed experts forward，关闭时恢复原方法。 |
| [ascend_adapter.py](ascend_adapter.py) | 原生 DeepEP 通信接入、compact receive 行分组/恢复、原生 Ascend F 读取 bank 视图。 |
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

公共方法签名已确定，不需要重做 LFU 或再加一套封装。下一步要确认的是实际 backend 能否按这些语义交付。下表中的“你”指 cache / 框架适配负责人，“同事”指 H2D / prefill 流水负责人。

| 对接项 | 已定接口 / 语义 | 双方需要落定的内容与责任 |
| --- | --- | --- |
| 权重布局与 bindings | canonical key 定位完整专家 bundle；`hot_bindings` / `execution_bindings` 指定物理池 | 双方定一份具体布局：各权重与 scale 的 shape、dtype、stride、每 slot 字节数、kernel 视图。同事负责可供传输读取的 host 地址/注册，你负责 F 使用对应视图。host pin/register 的创建、保活与释放归属一并明确。 |
| Decode 搬运 | `materialize_into(DeviceCopyPlan, bindings, stream) -> TransferTicket`；HOST→bank、HOT_SLOT→bank、HOST→hot | 同事提供三类复制及设备侧计划转换，明确 D2D 也包含在交付范围。支持任意 valid 空洞、零条目、实际 destination_slot；计划内容变化不能在 replay 时依赖 Python 读回。 |
| Prefill 搬运 | `materialize_host_into(HostCopyPlan, bindings, stream) -> TransferTicket` | 同事提供 host 区间计划到实际复制的实现；双方明确字节偏移和组件布局与 decode 使用同一份权重存储。默认 `TensorResidencyOps` 只产生 decode 设备计划，prefill 的 HOST_ONLY reservation/provider 仍需后端提供。 |
| 完成事件与内存复用 | transfer 返回复制完成票据；cache 发布映射；F 返回 `LocalExecution(output, read_done)` | 同事保证票据覆盖整个 bundle、所有内部复制流，并继承传入 stream 的前序依赖。你保证 F 的 read_done 覆盖全部权重读取；双方用真实 event 验证“读完才覆盖、复制完才发布”。 |
| Prefill 流水与借池 | `start / validate_session / seal / acquire / finish / abort`；`borrow_hot_arena`，快照与原位置回填 | 同事交付具体 pipeline 和 bank 布局，明确热池可借字节范围、A/B 等区域是否别名、何时不再有后台写入。你接 owner 阶段切换和快照恢复。`finish/abort` 必须覆盖后台提交及设备工作，才能归还池。 |
| Dispatch 与原生 F | `dispatch_completion`、`DispatchLayoutAdapter`、`LocalRoutes`、`run_experts(layer, dispatch_output, bindings, context)` | 主要由你实现目标模型/EP 适配；同事确认 pipeline 使用真实通信完成依赖。固定 wave 划分、local expert ID、padding mask、行顺序恢复与路由权重施加位置必须一致。Python dispatch 返回不等于通信完成。 |
| Backend 组装与 graph | `MoEBackend.finalize()`、`install_programs(...)`，固定地址的计划/池/状态 | 你组装实际 backend、原生程序和模型 hooks；同事提供可捕图的 device transfer。共同验证后再声明能力，不能降低 serving capability gate 来绕过未完成的 graph 接入。 |

驻留表由 cache/runtime 统一发布；transport 不直接修改 `key_to_slot`、`slot_to_key`、`slot_valid` 或 generation。

一个必须提前对齐的例子：`valid=[False, True, False, True]`、`destination_slot=[0,1,2,3]`、`count=2` 表示写 slot 1 和 3，不是写 slot 0 和 1。同事 H2D 文档所引用的本地 `group_pack_copy` 有选中项压紧语义，device adapter 必须保留公共计划的目标位置；不能直接把压紧序号当 slot。这里描述的是该文档指定实现，不能据此假定所有 MemFabric 版本相同。

另一个需要实机确认的点：pinned host Tensor 不自动证明其地址可供设备 kernel 直接访问。按照同事的方案，需要确认同一份 host 权重既能用于 ACL 异步复制，也能合法注册/映射给 decode 的设备搬运实现，不能直接把 CPU `data_ptr()` 当设备可访问地址。

## 4. 我方框架接线已补齐，剩余物理交付

此前下列框架功能确实没有实现，不能仅称为“待联调”。本轮已补齐：

1. `CacheBackend.finalize()` 创建 cache 状态、runtime 和逐层 Program，调用初始化；不再让同事实现该组装。
2. `DeepSeekV4ProgramBinding` 将程序安装到真实 routed experts 路径，并保留原模型 shared/scaling 行为。
3. `AscendDispatchLayout` 将本分支 NPU compact DeepEP receive counts 转为有效本地路由，`AscendExpertRunner` 复用原生 F；并非任意厂商 dispatcher 的通用适配。
4. 使用公共 `PrefillDispatchOnceProgram` 接同事的 pipeline，经过 acquire/wait/F/read_done/finish；正常完成与失败清理都能到达流水。
5. NPU graph backend 已接 owner lease，在 graph 销毁后释放。正常图构建在 serving scope 外，统计开关默认为 0。

同事现在提供 `prepare_weights`、`bind_weights -> CacheBackendBinding`、`close_weights` 三个物理入口：host 加载/原生格式、权重池与注册、真实 transfer、prefill pipeline 及其 host-plan/bank 绑定。字段与示例见设计第 14 节。这些函数名是新明确的交接约定，不是声称同事已经实现同名函数；缺失的物理交付不再包含模型 hook、dispatch adapter 或 native F adapter。

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

此前 review 基线为 119 项 CPU 测试通过。本轮新增真实模块安装/组装与 graph lease 测试，最终结果见修改总结；测试环境有 PyTorch CPU，没有 torch_npu。

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

## 7. 当前完成情况与 NPU 联调顺序

### 一条命令运行现有验证

在已安装匹配 PyTorch / torch_npu / CANN 的 Ascend 环境，用该环境的 Python 执行：

```bash
python test/manual/validate_moe_cache_npu.py --device 0
```

入口：[validate_moe_cache_npu.py](../../../../../test/manual/validate_moe_cache_npu.py)。无需下载模型；自动使用当前 checkout 的源码，以独立进程运行四组测试。轻量 package bootstrap 绕过 serving API 初始化，但不替换 cache 实现；CPU 测试内部仍使用其明确声明的硬件/通信替身。

| 分组 | 当前用例数 | 验证范围 |
| --- | --- | --- |
| `cpu` | 133 | 全部 offloading CPU UT，包括模型 installer、CacheBackend、graph lease 的测试替身验证 |
| `npu_runtime` | 3 | 真实小权重 H2D/D2D、event 等待、LFU 替换、映射发布、快照回填 |
| `npu_metadata_graph` | 1 | 真实 NPU metadata/LFU capture/replay，变化路由与 CPU 对照 |
| `npu_layout` | 2 | 生产 AscendDispatchLayout 在 NPU eager/graph 下的行整理、scale、空 wave、尾组、NaN padding 屏蔽与固定输出地址；F 为简单算术 |

默认生成带时间戳的 `moe-cache-validation-*` 目录，包含 `report.json`、每组 JSON 结果和完整日志。可通过 `--output-dir /path/to/new-empty-dir` 指定空目录，`--timeout 900` 设置每组超时秒数。所有测试必须实际运行且无 skip/expected failure 才算通过：退出码 `0` 为上述范围内通过，`1` 为失败或超时，`2` 为缺失硬件或验证不完整。

交给执行 AI 时要求：使用当前分支，保留原始断言和日志；先运行脚本，不把失败改成 skip，不用 CPU fallback 冒充 NPU。返回提交号、`report.json`、失败组日志和环境版本。若修复代码，单独说明修复，并使用新的输出目录重跑。

本入口不验证真实 DeepEP/HCCL、多卡 EP、原生 MoE 量化/GMM kernel、同事生产 transfer/prefill pipeline 或完整模型 graph；这些范围写入每份报告。不能把四组通过解释为整模型已联调通过。

### 7.1 已完成到哪一层

| 部分 | 当前状态 | 尚未证明的内容 |
| --- | --- | --- |
| LFU 算法 | 已实现离线初始化、逐路由计数、每 4 个有效 decode forward 层内冷热配对、全 rank 最多 96 次交换、每轮 ×0.7 衰减；候选 usage≥2，热度差≥1.5 | 真实模型的命中率和性能收益；当前 verify 模式仍未开放，不能把算法支持 batch 计数等同于 verify 已接入 |
| Cache 元数据与驻留管理 | 已实现实际需求计划、临时 execution 加载、幸存 slot 保留、复制后发布、读者完成保护、精确位置恢复 | NPU 算子支持和真实传输下的完整时序 |
| 公共调用链与生命周期 | 公共 hooks、阶段切换、实际模型安装、Ascend/DeepEP 适配、native F bank 绑定、CacheBackend 组装及 NPU graph lease 均已实现 | 同事物理 binding 接入后的真实模型结果与多卡时序 |
| 测试 | 119 项原有 CPU UT，加本轮模块调用与 graph 生命周期测试；两个既有 NPU 手工脚本已准备 | 真实 transport、完整 decode graph、真实 prefill pipeline 和多卡 EP 尚未实机验收 |

LFU 的 1.5 是热度的差值，不是 1.5 倍。当前仍为层内配对，再跨层按热度差排序选出本轮交换；不是全局任意冷热配对。

### 7.2 第一步：现在即可跑的 NPU 基线

双方使用同一代码基线，记录提交号、NPU 型号、CANN、PyTorch、torch_npu 版本；接入同事 backend 时再记录其扩展和 MemFabric 构建版本。以下命令均从仓库根目录执行，环境需已安装对应依赖：

```bash
# CPU 语义与框架适配基线。
PYTHONPATH=python python -m unittest discover -s test/registered/unit/offloading -p 'test_*.py' -v

# 真实小矩阵 H2D/D2D、event 等待、发布和恢复：3 项。
PYTHONPATH=python python test/manual/test_moe_cache_runtime_npu.py -v

# 仅 metadata/LFU 的 NPU graph capture/replay：1 项。
PYTHONPATH=python python test/manual/test_moe_cache_metadata_npu.py -v
```

两个独立 NPU 脚本都可通过 `MOE_TEST_NPU_DEVICE` 选择可见设备，默认 0；统一入口用 `--device` 设置。两者都必须实际运行测试，不能将 skipped 计作通过。metadata 脚本的成功也不表示 H2D 或整个模型已经入图。

小权重流程的可检查结果：初始热池 `[0,2,4,7]`，测试给定的四次 forward 后为 `[1,2,5,7]`；实际计算与独立 CPU 参考一致；新映射只在复制完成后发布；借池覆盖后按原 slot 恢复，并能继续 decode。这一步不依赖同事优化流水，可以先排查当前 NPU 对所用 tensor 操作和 event 的支持。

### 7.3 第二步：替换测试搬运器，验证同事真实 backend

保留 `test_moe_cache_runtime_npu.py` 的小权重、路由序列、CPU 参考与发布顺序断言，将测试 fixture 中的 `EagerNPUTransfer` 和 bindings 接到双方确定的真实 transfer。当前脚本没有选择生产 backend 的命令行参数，需要修改或参数化 fixture，不能只换一个启动参数。

最小验收包括：HOST→bank、HOT→bank、HOST→hot，完整多组件 bundle，valid 有空洞和 count=0，原位置回填，以及等待前后映射状态。先验证结果与依赖，不要求这一步已经获得流水重叠收益。transfer 仍只复制，映射发布仍走现有 ResidencyOps。

### 7.4 第三步：接单层真实 D/F/C 和 prefill 流水

使用本轮已完成的 native F / dispatch adapter，接同事的真实物理 binding。使用相同权重、输入和路由，对比常驻权重执行与 offload 执行的结果。检查每个 batch 的有效 token-expert 分配只计一次、padding 不计数、路由权重不重复施加。

随后与同事接真实 `PrefillWeightPipeline`：层开始启动，D 完成后 seal，逐 wave acquire/F，finish 归还资源。在同一进程中运行“若干 decode → prefill → decode”，验证一次 D/一次 C、后台工作结束后才回填、原热专家与 LFU 周期恢复、实际输出一致。现有模拟覆盖热池的测试不能替代这一步。

### 7.5 第四步：验证完整单层 decode graph

使用同事的 device-plan 实现，将真实权重准备、F、LFU 更新和映射发布一起 capture/replay。保持 tensor/池地址固定，改变路由、valid 和命中情况，至少跨过一个四步更新周期；与 eager 参考比较输出、usage 和最终驻留映射。检查空 wave、无交换轮次，以及替换后下一步命中。

backend 注册、首次 kernel 加载等初始化放在 capture 前。不能用测试搬运器的 host 读计划替代动态图执行，也不能用只覆盖 metadata 的 graph 测试作为整层通过证据。功能正确后，再检查复制与计算的实际重叠。

### 7.6 第五步：真实模型与两卡 EP

在通过上述验证后，使用本轮已实现的 `CacheBackend` 和模型 installer，在符合现有能力门槛的配置下运行真实推理。先验证单卡/单层，再启用两卡 EP，检查全局 expert ID 到各 rank local ID 的映射、dispatch/combine 输出以及每 rank 独立 cache。本轮 native adapter 需要 DeepEP，单卡阶段若所用 DeepEP 构建不支持单 rank，就使用小权重单层测试后直接进入两卡。

最小多卡用例包含：路由落到远端 rank、本地无输入但收到远端专家任务、某 rank 无有效路由但全局 decode 有工作、全局 prefill 下本地 IDLE。核对统计按 dispatch 后有效分配计数，周期按全局有效 forward 推进，阶段切换一致。现有小权重脚本虽然构造了 EP catalog，但没有真实跨卡通信，不能代替该验收。

本轮联调先以结果正确、生命周期完整为完成标准。确认这些之后再测延迟、带宽和重叠收益；不为基本功能增加备用传输栈或新的抽象层。
