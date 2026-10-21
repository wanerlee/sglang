# MoE Cache 修改总结

适用分支：`codex/moe-offload-dev`

整理日期：2026-10-21

本次合并同事 `59cbb5d` 的 dispatch-once、H2D 公共接口和资源重命名；详细边界见 [合并适配设计](MERGE_ADAPTATION_DESIGN.zh-CN.md)。累计工作实现了周期 LFU、cache 元数据管理、权重加载与替换计划，以及通过现有搬运接口串联的 cache 调用方流程。同时补充了 CPU 调用链测试和独立 NPU 测试脚本。

当前结论：**cache 侧算法与调用方已经实现，119 项 CPU 测试通过；生产 NPU 后端和真实模型绑定尚未完成，NPU 脚本尚未实机验证。** 同事的公共 Program/transfer 接口沿用其提交；我方负责本文列出的 cache 功能和接口适配。

## 1. 周期 LFU 算法

新增 [periodic_lfu.py](periodic_lfu.py)，提供 `PeriodicLFUConfig`、`PeriodicLFUState` 和 `PeriodicLFUPolicy`。

| 功能 | 实现行为 |
| --- | --- |
| 离线热启动 | 读取路由频次，筛选当前 EP rank 管理且需要卸载的专家，选择初始 TopK。 |
| 在线热度 | 每条有效 token→expert 分配累计一次；命中和 miss 都计数。在线 usage 从 0 开始，不用离线频次初始化。 |
| 更新周期 | 每 4 个有效完整 forward 检查一次；实际 runtime 当前接入普通 decode。 |
| 配对范围 | 每层内部将最热非驻留专家与最冷驻留专家配对。 |
| 替换门槛 | 候选 usage ≥2，且候选比被替换专家至少高 1.5；这里是差值，不是倍数。 |
| 换入限量 | 所有层合格配对按热度差排序，每 rank 每轮最多替换 96 个专家。 |
| 热度遗忘 | 每个调整周期选择完成后，所有 usage 统一乘 0.7。 |
| 状态保存 | 支持原位复制热度、周期位置和更新标记。 |

保留原有 `TensorLFUPolicy`，通过配置显式选择新策略。后续替换保持层内配对，没有改成全局冷热配对。Verify 的算法计数语义与实际推理支持分开：当前 runtime 不解除原分支的投机解码限制。

## 2. Cache 元数据管理

新增 [cache_metadata.py](cache_metadata.py)。

- `CacheMetadataAdapter`：负责 `begin_step → record_local_routes`、状态与快照；策略收尾由公共 `CacheUpdateCoordinator` 统一调用。
- 路由适配：支持层内、rank 内专家编号；连续 EP 分片的 global ID 可转换为 local ID。过滤 padding、无效编号和不属于当前 wave 的分配。
- Batch 统计：同一专家被多个 token 选中时累计多次；多 wave 不重复统计同一分配。
- `CacheRouteStats`：记录本步 accesses、hits、misses；这些是路由分配次数，不是搬运次数。
- `target_keys`：LFU 选出的目标集合；与当前映射的差分仅由 residency 计算，不维护重复的 keep/admit/evict 数组。
- `HotMetadataSnapshot`：保存热池映射、有效位、generation，以及配套 LFU 状态；恢复时失效临时 execution bank。
- 模式开关：正常 decode 可以统计；prefill、warmup/capture、全局无 token 时关闭。全局有工作但本 rank 无路由时仍推进周期。
- 资源核算：报告组件自身持有的元数据内存，不重复统计外部权重池。

## 3. 实际需求计划与驻留发布

新增 [cache_residency.py](cache_residency.py)，实现 `TensorResidencyOps`。

| 环节 | 已实现内容 |
| --- | --- |
| 当前 batch 的权重准备 | 根据 dispatch 后实际需求生成 execution bank 加载计划，不把整个 wave 的全部专家都作为加载对象。 |
| 命中与 miss | 命中使用 HOT_SLOT 来源；miss 使用 HOST 来源。 |
| 加载去重 | 重复路由仍分别计数，但同一 wave 对同一专家只生成一次加载需求。 |
| 热缓存替换 | 保留幸存专家的物理 slot，将新专家分配到空出的位置。 |
| 覆盖保护 | 更新前失效受影响 slot；runtime 为复制建立前序读者完成依赖。 |
| 映射发布 | 等待完整专家权重就绪后，更新 key↔slot、有效位和 generation。 |
| 初始化与回填 | 支持将指定专家装入指定 slot，用于初始加载和快照原位置恢复。 |

复用现有 `DeviceCopyPlan`，没有新增生产 H2D 引擎。采用公共 `ExecutionReservation` / `UpdatePlan`；热更新的 state 包含期望 slot 映射和发布开关，已删除重复私有计划类型。

## 4. Runtime 调用链与框架接线

新增 [cache_runtime.py](cache_runtime.py)：

- `CacheRuntime`：初始化、step 生命周期、搬运调用、映射发布、阶段切换和完成事件编排。
- 删除旧 `CacheLocalExpertProgram`，统一使用公共 `DecodeLocalExpertProgram` / `ExpertWeightAccess`；我方提供 make_request 统计适配。
- `CacheExecutionContext`：phase/capturing、线程安全 retain、compute 完成事件、resources_idle 汇合；每个 forward 使用独立上下文，capture 资源保活到 runtime 关闭。
- `LocalRoutes`：约定路由 adapter 向 cache 提供的本地专家编号和有效 mask。

完整调用方路径：

```text
离线 TopK → 初始装载 → 等待 → 建立驻留映射
                         ↓
forward 开始 → 清空本步计数
                         ↓
每层 / 每波：dispatch → 计数 → 需求计划
             → materialize_into → 等待 → 准备 bank → 专家计算 → combine
                         ↓
完整 forward 结束 → 累计 LFU → 周期到期时生成替换计划
                 → materialize_into → 等待 → 发布新驻留映射
```

阶段判断不依赖 TBO 才提供的 `global_forward_mode`：结合 `is_extend_in_batch` 与全局 token 数推导。在全局 prefill 中，本地 IDLE 或 DECODE rank 都进入 prefill；在有效全局 decode 中，本地 IDLE rank 仍统计 dispatch 后的访问并推进周期。优先使用 `original_global_num_tokens_cpu`，避免把图 padding 当成有效 token；全局空闲时保持当前阶段，不触发无谓回填。

Eager 的周期由 runtime 中一个 host 整数跟随完整有效 forward 推进，无需读取设备 tensor。`forward_scope` 必须包围一次实际 eager 执行或 graph replay；成功提交后只推进一次，prefill、全局 IDLE、capture/warmup 不推进，失败后沿用 owner 的停止执行约束。Host 周期跨 prefill 保持，设备 LFU 周期随快照恢复，因此 eager 与 graph 交替执行时保持一致。非周期 eager 调用 `cache_updates.emit_step_end(context, adjust=False)`，累计 usage 后直接结束，不排序、不生成热池计划、不提交空搬运、不重建映射。到期但无合格交换时仍按规则衰减。图路径保持默认 `adjust=True` 的固定操作序列，设备门控实际交换；此轮没有宣称已消除图路径的非周期排序开销。

阶段切换路径：

```text
前序工作完成 → 保存 decode 元数据与 LFU 状态 → prefill 借用热池
           → 原专家按原 slot 回填 → 等待 → 恢复元数据与 LFU 状态
```

相关框架文件修改：

| 文件 | 修改 |
| --- | --- |
| [offloader.py](offloader.py) | 增加周期 LFU 配置注入，以及 eager step scope 转发。 |
| [BaseOffloader](../base.py) | 增加默认无操作的 `eager_step_scope()`，保持其他 offloader 的默认行为。 |
| [interfaces.py](interfaces.py) | 补充 runtime 的默认 eager scope，以及初始化、LFU 收尾、搬运和发布调用契约。 |
| [ModelRunner](../../model_executor/model_runner.py) | 在实际 eager 执行分支调用 step scope；decode graph 分支绕过它，避免重复计数。 |
| [__init__.py](__init__.py) | 导出新算法、元数据、residency 和 runtime 组件。 |

当前基础调度以明确的 compute→copy→compute 依赖保证正确性，尚未实现跨 wave 的搬运计算重叠。已有 step/capture 接线不等于完整 NPU graph 已验证。

本次新增接口衔接：

- `CachePolicy.finish_step` 提供默认空实现；coordinator 调用周期 LFU 衰减/清理，eager 非周期步跳过排序和热池搬运。
- residency 返回公共 reservation 和发布票据，记录 F 最终读依赖；未 release 时不能覆盖 bank 或执行热替换。
- prefill layer-start 委托公共 Program；CPU 测试 pipeline 验证 host 复制、分组 F、单次 D/C 和原 slot 回填，生产流水仍由同事提供。
- 初始化和恢复保留 exact-slot 路径；删除旧重复执行搬运 helper，更新 NPU 手工脚本接口。

## 5. 测试与验证结果

### 5.1 CPU 测试

| 测试文件 | 主要覆盖 |
| --- | --- |
| [test_periodic_lfu.py](../../../../../test/registered/unit/offloading/test_periodic_lfu.py) | 离线初始化、周期、层内配对、阈值、限量和衰减。 |
| [test_cache_metadata.py](../../../../../test/registered/unit/offloading/test_cache_metadata.py) | Batch/EP 计数、padding、多 wave、目标集合、快照、资源核算、固定地址数据流。 |
| [test_moe_framework.py](../../../../../test/registered/unit/offloading/test_moe_framework.py) | 在原框架测试基础上增加 LFU 策略注入验证。 |
| [test_cache_runtime.py](../../../../../test/registered/unit/offloading/test_cache_runtime.py) | 22 项调用链测试：无 TBO 的空闲 rank、全局 prefill、周期跳过、图/eager 交替、延迟复制、实际权重计算、替换后命中、回填、失败处理、零缓存、多 bank，以及真实 ModelRunner 方法体的分支接线。 |
| [test_cache_npu_test_support.py](../../../../../test/registered/unit/offloading/test_cache_npu_test_support.py) | 2 项 CPU 检查：NPU 脚本的独立参考结果和小模型 bank 计算逻辑。 |

本地合计 **119 项 CPU 测试通过**（合并后原有 117 项，本轮删除 1 项无调用方测试、增加 3 项回归）。Ruff 检查与格式、isort、空白、Python 语法及文档本地链接检查通过。CPU 测试不构成 NPU 搬运或硬件图验证。

### 5.2 NPU 元数据测试

[test_moe_cache_metadata_npu.py](../../../../../test/manual/test_moe_cache_metadata_npu.py) 验证 LFU、计数和目标集合的 NPU 捕图及变化输入 replay，与 CPU 结果比较。它不搬运专家权重。

### 5.3 NPU 小规模完整流程测试

[test_moe_cache_runtime_npu.py](../../../../../test/manual/test_moe_cache_runtime_npu.py) 包含三组用例：

1. 真实 H2D/D2D、4 轮 LFU 替换、下一轮命中，以及 NPU 计算结果与 CPU 参考结果比较。
2. 在复制完成、尚未 commit 的中间位置验证映射仍未发布，commit 后才更新。
3. 覆盖借出的热池后，按原位置回填，恢复映射、generation、热度和周期，随后继续 decode。

配置为两个逻辑层、每层 4 个本地专家、4 个热 slot、2 个 execution slot。每个专家包含两块 8×8 FP32 矩阵，host 权重使用 pinned memory。测试使用真实 NPU stream/event 和复制操作，但通过主机读取计划进行 eager 调度。

```bash
# 在具有匹配 PyTorch、torch_npu、CANN 和项目依赖的 Ascend 环境，仓库根目录执行：
PYTHONPATH=python python test/manual/test_moe_cache_runtime_npu.py -v

# 可选，默认使用 npu:0：
MOE_TEST_NPU_DEVICE=1 PYTHONPATH=python python test/manual/test_moe_cache_runtime_npu.py -v
```

不下载模型，不依赖同事的优化流水。该测试不注册成 serving backend，不宣称动态搬运入图、分布式 EP 或真实模型 prefill 支持。

本机没有 `torch_npu`，**两个 NPU 脚本尚未实机验证**。已检查完整流程脚本的缺失环境分支：直接执行时缺少硬件返回码为 2，失败为 1，全部通过为 0，不将 skipped 当作验证通过。

## 6. 文档交付

| 文档 | 用途 |
| --- | --- |
| [LFU_DESIGN.zh-CN.md](LFU_DESIGN.zh-CN.md) | 缓存算法、状态、接口及具体例子。 |
| [CACHE_DESIGN.zh-CN.md](CACHE_DESIGN.zh-CN.md) | 总体设计、五项需求、实现边界与验证状态。 |
| [CACHE_RUNTIME_HANDOFF.zh-CN.md](CACHE_RUNTIME_HANDOFF.zh-CN.md) | 调用链、复制计划字段、双方职责、待对齐接口及 NPU 测试验收方式。 |
| 本文 | 汇总本轮全部修改，供分工、评审和交接使用。 |

## 7. 尚待完成与对接边界

本节为此前交付时的缺口记录；模型/dispatcher 绑定已在第 9 节补齐，当前边界以第 9 节及合并适配设计第 14 节为准。

| 待完成项 | 说明 |
| --- | --- |
| 生产 NPU transport | 真实权重布局、host 注册、目标池 bindings、H2D/D2D 实现，以及覆盖所有相关流的完成凭据。测试搬运器不能替代它。 |
| 模型 / dispatcher 绑定 | 在真实 backend 中构造 runtime；安装目标模型程序；从实际 EP 输出提取 LocalRoutes；让原生专家算子读取 execution bank。 |
| 原生 prefill 流水 | 实际借池范围、流式加载和后台流完成依赖。当前 NPU 用例只模拟覆盖借用空间并验证恢复。 |
| 完整 NPU graph | 动态复制计划的硬件实现、捕图 / replay、地址与 workspace、graph 生命周期验证。 |
| 分布式与性能 | 真实多卡 EP 正确性、跨波重叠、吞吐与时延收益。 |
| Verify | 仍受原分支限制，未接入实际投机解码推理。 |

Cache/runtime 统一维护和发布驻留映射；搬运实现接收计划并报告完成依赖。无需同事实现 LFU 算法，具体接口清单见交接文档。


## 8. 合并后的代码 Review 与精简

Review 范围：相对同事 `59cbb5d` 的 cache/LFU、框架接入、公共接口适配，以及 CPU/NPU 测试调用链。按基础功能范围检查，不增加 fallback 或新的抽象层。

| 发现 | 处理 |
| --- | --- |
| runtime 在 decode 清理时也调用 prefill abort，可能把 decode context 交给只接受 prefill 的流水。 | 只在 prefill context 下停止提交线程；保留“先停止线程、再 drain 设备”的顺序。新增两项回归。 |
| 同一 wave 在 metadata 和 runtime 重复计算路由直方图。 | metadata 返回每专家需求，runtime 复用；需求不受统计开关影响。新增 warmup/capture 关闭统计仍有加载需求的回归。 |
| 独立 cache_update_enabled 没有推理调用方，预留 verify 开关与当前拒绝投机解码的范围不一致。 | 删除函数、导出和仅覆盖它的 UT；保留真实 runtime 的 phase/capture/IDLE 回归。 |
| runtime 重复保存 phase 和未使用的 bindings 引用。 | phase 统一读取 context，bindings 由 residency 持有，删除冗余字段。 |

LFU 的层内配对、周期 4、门槛 2/差值 1.5、最多 96 次交换、衰减 0.7 均未变化。发布完成票据、F 最后读者依赖和异步资源保活仍是正确性所需，予以保留。

验证：原 117 项测试删除 1 项无调用方辅助 gate 测试，增加 3 项针对性回归，**119 项 CPU UT 全部通过**。真实 NPU H2D/D2D、生产模型绑定及完整 graph 尚未在本机验证。

## 9. 补齐我方模型与框架适配

纠正之前“框架适配已完成”的表述：此前完成的是公共调用链，真实模型 installer 和原生 dispatch/F 接线仍缺失。本轮补齐这些我方任务，实际驱动搬运及 prefill 流水继续由同事负责。

| 代码 | 本轮实现 |
| --- | --- |
| [cache_backend.py](cache_backend.py) | CacheBackend 接物理权重 binding，构造 cache 状态、runtime、每层 decode/prefill Program、installer，初始化热缓存；不要求同事组装模型执行链。 |
| [model_binding.py](model_binding.py) | 接 DeepSeek V4 routed experts；覆盖 NPU 直接调用的 `forward_hc_pre_from_prev` 层首入口，保留 shared/scaling 与常驻层；关闭时恢复原方法。 |
| [ascend_adapter.py](ascend_adapter.py) | 兼容原生 DeepEP 的外层关键字 wrapper；compact receive 行分 wave、scale 同步整理、padding 排除、原顺序 combine；复用 AscendRunnerCore 与原生 bank 权重/scale 视图。 |
| [cache_runtime.py](cache_runtime.py) | 安装器和物理权重释放的关闭路径，先结束 graph/设备使用，再恢复模型方法和释放物理绑定。 |
| [NPU graph backend](../../hardware_backend/npu/graph_runner/npu_cudagraph_backend.py) | shape 级 owner lease 与 graph cleanup 接线；保留不启用 MoE offload 时的原构造行为。 |

新增 10 项模型调用测试及 4 项 graph 生命周期/兼容测试，**133 项 CPU UT 全部通过**。测试执行实际生产 backend/adapter/installer，并抽取原生 Ascend F 与 dispatcher wrapper 的方法体；HCCL、NPU kernel 和搬运由 CPU 测试替身提供。覆盖 decode→prefill→decode、LFU、读取依赖、恢复方法、动态 receive counts 和 padding，不以此宣称 NPU 已跑通。

物理交付仍需同事提供 host 加载/注册、权重池、真实 transfer、prefill pipeline 及其物理 host-plan/bank 视图；我方的模型 hook、dispatch adapter、F adapter、runtime 组装不再列为对方任务。具体字段及注入示例见 [合并适配设计第 14 节](MERGE_ADAPTATION_DESIGN.zh-CN.md#14-本轮补齐我方真实框架适配)。LFU 参数和层内配对规则未改。

本轮提交前 review 修正了图捕获失败路径：在 warmup 前获取 lease，重复 shape 在执行模型前拒绝；创建后立即由 backend 持有图，只有成功生成输出才允许 replay。失败后仍由 cleanup 排空设备、销毁未完成图，再释放 lease，不添加自动重试或 fallback。补充回归覆盖捕获失败、重复捕获、未加载 bank lane 含 NaN、零接收行的输出形状/类型，以及关闭后恢复 NPU 直接入口。
