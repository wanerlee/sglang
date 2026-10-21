# MoE 周期 LFU 策略设计

适用分支：`codex/moe-offload-dev`

状态：已实现独立的 `PeriodicLFUPolicy`、离线初始集合选择和 offloader 策略注入入口；CPU 算法测试通过。现已通过 `CacheRuntime` 串联计划、搬运调用与就绪后发布，见 [调用链与 H2D 交接表](CACHE_RUNTIME_HANDOFF.zh-CN.md)；真实 NPU 搬运、模型绑定与 graph 验证仍待后端接入。原有 `TensorLFUPolicy` 保留，算法阈值及周期不变。

本策略决定哪些专家应该留在热缓存中：离线频次决定初始集合，在线路由累积 usage，每 4 个有效 forward 按层配对替换，最多换入 96 个专家，再统一衰减热度。这里只设计算法、状态与调用接口；slot 分配、权重搬运和发布由现有 residency/runtime 承担。

本文与 [Cache 总体设计](CACHE_DESIGN.zh-CN.md) 一致；接口基于原分支 `5b13541` 的 [CachePolicy](interfaces.py) 和 [ExpertCatalog](config.py)。

本策略参考 Strata adaptive tier 的选取规则，并增加确定性排序、EP rank 独立统计和设备图内执行约束。固定源码基线及差异见第 13 节；这些规则与 Strata 的完整执行引擎、异步模式不等价。

## 1. 算法规则

| 项目 | 确定规则 |
| --- | --- |
| 初始化 | 本 EP rank 所有可卸载 `(layer, expert)` 按离线频次排序，选择前 K 个。 |
| 在线热度 | 每个有效 token-expert 路由使该专家 usage 加 1，命中和 miss 都统计。 |
| 更新周期 | 默认每 4 个目标模型有效 decode/verify 完整 forward 触发一次。 |
| 候选配对 | 每层非驻留专家按 usage 降序，驻留专家按 usage 升序，一一配对。 |
| 准入门槛 | `candidate_usage >= 2` 且 `candidate_usage - victim_usage >= 1.5`。 |
| 换入上限 | 所有层合格配对按热度差降序排列，每 rank 每轮最多选择 96 对。 |
| 衰减 | 周期轮使用衰减前热度选取，然后所有专家 `usage *= 0.7`；无交换也衰减。 |
| 输出 | 输出目标专家集合，不给专家分配物理 slot，不搬运或发布权重。 |

**1.5 是热度差，不是倍数。**热度 12 可以替换热度 10；热度 3 不能替换热度 2。

这是带周期更新、准入门槛和衰减的 LFU，不是在每次 miss 时淘汰一个专家。临时加载的 miss 是否进入热池，要等周期选取结果。

## 2. 专家身份与容量

使用 `ExpertCatalog.key(layer_id, local_id)` 将当前 rank 的可卸载专家映射到 `[0, N)`。常驻层、shared experts 不在此范围内。

```text
N = 当前 rank 的全部可卸载专家数量
K = floor(cache_ratio * N)
```

保持原 `MoEOffloadConfig.hot_slots()` 公式。各 rank 独立统计、独立选取，不合并全局热度。

初始化 Top-K 可以跨层竞争；之后只做同层替换。因此每层驻留数量由初始化形成，并保持不变。某层零驻留时没有可替换对象，继续通过 miss 路径执行，不增加补配额算法。

这也是需要验收的容量取舍：离线频次全部为零时，按 key 排序会优先填满前面的层，后面的层可能永久没有热 slot。初始化应报告每层驻留数量 `K_l` 及离线统计覆盖情况；不把这种行为误判为在线计数失效，也不隐式改变既定的层内替换规则。

算法只需知道每个 key 的层号，以及 `key_to_slot[key] >= 0` 是否成立；不要求每层拥有连续的 slot 区间。

## 3. 配置与状态

沿用总体设计建议的配置：

```python
@dataclass(frozen=True)
class PeriodicLFUConfig:
    offline_stats_path: str       # 初始化读取，runtime/backend 使用
    update_interval: int = 4
    min_usage: float = 2.0
    replacement_min_gain: float = 1.5
    max_swaps: int = 96           # 每个 EP rank 的全层上限
    usage_decay: float = 0.7
```

周期为正整数、上限为非负整数、阈值为非负有限数、衰减系数在 `[0, 1]`。配置在初始化时检查；不增加运行期参数调优逻辑。

上述范围描述当前实现的参数检查，不等于所有组合都已获得长期运行的数值精度保证。`usage_decay=1` 的限制及待修复要求见第 14 节。

```python
@dataclass
class PeriodicLFUState:
    usage: object       # device float32[N]，初始全零
    step_mod: object    # device int64 标量，初始 0
    update_due: object  # device bool 标量，初始 False
```

| 输入或输出 | 形状与含义 | 所有者 |
| --- | --- | --- |
| `step_counts` | int64[N]，本次完整 forward 的有效路由次数 | runtime，复用 `step_access_count` |
| `update_enabled` | device bool 标量，本次是否统计和推进周期 | runtime |
| `key_to_slot` | int64[N]，-1 表示非驻留 | residency |
| `target_keys` | int64[K]，目标专家 key，空项为 -1 | runtime 预分配，policy 写入 |
| `key_to_layer` 或层 key 区间 | 固定专家身份信息 | 初始化时由 catalog 建立 |

`target_keys` 是集合的定长表示。可以按 key 排列以保证结果确定，但**数组位置不代表物理 slot**。正常初始化装满 K 项，同层一换一后仍是 K 项；K=0 时输出为空。

## 4. 离线初始化

离线文件示例，专家编号为模型的全局 expert ID：

```json
{
  "model_id": "checkpoint-used-for-profiling",
  "layers": {
    "3": {"4": 120, "5": 10, "6": 110, "7": 5},
    "4": {"4": 100, "5": 20, "6": 2, "7": 90}
  }
}
```

初始化属于 host 加载阶段，不进入 decode graph：

```python
# 参考伪代码。
# offline_count 已按 EP 归属和 catalog 转换为 rank-local key。
keys = list(range(N))
keys.sort(key=lambda key: (-offline_count[key], key))
initial_keys = keys[:K]

# runtime/backend 执行：加载 initial_keys，完成后建立驻留映射。
# LFU 不负责这些搬运，也不把离线计数写入在线 usage。
usage = zeros(N)
step_mod = 0
update_due = False
```

相同频次按 key 升序排序；未记录的专家离线频次为零。只读取当前模型、当前 rank 的可卸载专家。文件读取和模型匹配由初始化组件完成，不属于在线 LFU 循环。

离线计数只决定初始集合，在线 usage 从零开始。基础版不保存“学习后的 profile”，不持久化在线 usage。

## 5. 计数与周期

### 5.1 什么算一次访问

例如一个 batch 中，T0、T1、T2 都选择 E5：

```text
usage[E5] += 3
```

即使 E5 权重只加载了一次，也计 3 次；即使 E5 没有缓存，也计 3 次。不按 router 分数加权，不按权重搬运次数计数。

因此热度衡量的是路由频次，不直接等于节省的 H2D 次数或字节数。性能验收须同时记录实际 H2D 字节数、搬运等待和 decode 延迟，不能只以按 token 统计的命中率判定收益；基础版仍保留上述计数口径。

runtime 在 dispatch 后统计有效本地 token-expert 分配，跨波累加到 `step_counts`，每个分配只记录一次。padding、空路由、预取但没有实际执行的专家不计数。

### 5.2 什么算一步

- 一个 batch 的完整目标模型 forward 算一步，不按 token、层或 wave 推进。
- 一个有效 batch 在某 rank 没有本地专家需求时，该 rank 增量为零，但周期仍推进。
- prefill、warmup、capture 和全局无有效 token 的 IDLE 不推进。
- `update_enabled` 由 runtime 根据实际执行模式生成，不通过本 rank 的计数和是否为零推断。

当前分支未支持 verify。接通后，一个 verify forward 推进一步，统计目标模型实际执行的路由；拒绝的候选 token 不回滚 usage，draft 模型不计入本状态。本文不解除原分支的投机解码限制。

从初始化开始，周期如下：

| 有效 forward | step_mod | 本次替换和衰减 |
| --- | --- | --- |
| 1 | 1 | 否 |
| 2 | 2 | 否 |
| 3 | 3 | 否 |
| 4 | 0 | 是 |
| 5 | 1 | 否 |

## 6. 替换算法

### 6.1 参考伪代码

下面仅说明算法语义。动态列表、集合和 Python 分支不能直接作为 graph replay 实现。

```python
def update_statistics(state, step_counts, update_enabled):
    # 每次调用刷新标志，避免初始 step_mod=0 被误认为到期。
    state.update_due = False
    if update_enabled:
        state.usage += step_counts
        state.step_mod = (state.step_mod + 1) % update_interval
        state.update_due = (state.step_mod == 0)


def select_hot_set(state, key_to_slot, target_keys):
    resident = {key for key in range(N) if key_to_slot[key] >= 0}
    target = set(resident)

    if state.update_due:
        pairs = []
        for layer in offloaded_layers:
            hot = [e for e in layer.keys if e in resident]
            cold = [e for e in layer.keys if e not in resident]

            hot.sort(key=lambda e: (state.usage[e], e))
            cold.sort(key=lambda e: (-state.usage[e], e))

            for candidate, victim in zip(cold, hot):
                gain = state.usage[candidate] - state.usage[victim]
                if (state.usage[candidate] >= min_usage
                        and gain >= replacement_min_gain):
                    pairs.append((gain, layer.id, candidate, victim))

        # 稳定、可复现的同分顺序。
        pairs.sort(key=lambda p: (-p[0], p[1], p[2], p[3]))
        for gain, layer_id, candidate, victim in pairs[:max_swaps]:
            target.remove(victim)
            target.add(candidate)

    # 只输出集合，不改 key_to_slot，不分配 slot。
    result = sorted(target)
    target_keys[:] = result + [-1] * (K - len(result))


def finish_step(state):
    # 选择使用衰减前热度；每轮只衰减一次。
    if state.update_due:
        state.usage *= usage_decay
    state.update_due = False
```

一个候选和一个 victim 最多参与一对。所有配对基于本轮同一份 usage 和驻留集合，不边换边重新排名。无候选的层不产生配对；K=0 或所有专家驻留时，仍按周期衰减。

`max_swaps=0` 表示本轮不交换，计数和周期衰减照常执行，不额外增加另一套运行模式。

### 6.2 为什么层内替换和跨层排序不冲突

层内配对决定“谁替换谁”；跨层排序决定“本轮有限的换入预算优先给哪些配对”。

```text
L3 产生配对 A，热度差 8
L4 产生配对 B，热度差 12

若本轮最多换 1 个，选择 B。
L4 的新专家占 L4 被淘汰专家的空间，不占 L3 的 slot。
```

## 7. 完整例子

rank 1 负责每层 E4～E7，两层共 N=8，K=4。第 4 节离线频次选出：

```text
L3/E4、L3/E6、L4/E4、L4/E7
```

前 3 个有效 forward 只累加 usage，热集合不变。第 4 个 forward 计数后，假设热度如下：

| 层 | 驻留专家及 usage | 非驻留专家及 usage |
| --- | --- | --- |
| L3 | E4=2，E6=5 | E5=8，E7=3 |
| L4 | E4=1，E7=4 | E5=7，E6=2 |

排序配对：

```text
L3：E5(8) → E4(2)，差值 6，合格。
L3：E7(3) → E6(5)，差值 -2，不合格。
L4：E5(7) → E4(1)，差值 6，合格。
L4：E6(2) → E7(4)，差值 -2，不合格。
```

默认上限 96，本轮选择两对，目标集合变成：

```text
L3/E5、L3/E6、L4/E5、L4/E7
```

随后所有 usage 乘 0.7：

```text
L3：E4=1.4、E5=5.6、E6=3.5、E7=2.1
L4：E4=0.7、E5=4.9、E6=1.4、E7=2.8
```

被淘汰专家的 usage 不清零，换入专家也不重置；所有专家共同衰减。若将上限设为 1，两对差值相同，按层号选择 L3 的交换。

此时 LFU 只给出目标集合。新专家权重尚未就绪时，不能把这个输出直接当作已经生效的驻留表。

## 8. 接口与集成边界

实现位于 [periodic_lfu.py](periodic_lfu.py)，保留现有 `TensorLFUPolicy`：

```python
class PeriodicLFUPolicy(CachePolicy):
    def __init__(self, config, catalog):
        # 保存静态配置及专家层号信息，不持有或分配权重。
        ...

    def initial_keys(self, capacity: int, model_id: str) -> tuple[int, ...]:
        # Host 初始化：读取离线文件，检查 model_id，返回当前 rank 的 Top-K。
        # 仅返回专家集合；实际装入和驻留映射仍由 backend 完成。
        ...

    def allocate_state(self, num_experts, device):
        # 分配 usage、step_mod、update_due。
        ...

    def update_statistics(self, state, step_counts, update_enabled):
        ...

    def select_hot_set(self, state, key_to_slot, target_keys):
        ...

    def finish_step(self, state):
        # 本策略新增方法，不要求修改其他 CachePolicy 实现。
        ...

    def copy_state(self, source, destination):
        # 将三个字段 copy_ 到预分配状态，不能替换 tensor 地址。
        ...
```

`allocate_state`、`update_statistics`、`select_hot_set`、`copy_state` 保持 `CachePolicy` 的公共契约；`finish_step()` 是新增的具体策略调用点，不是原 runtime 已有功能。绑定新策略的 runtime 必须显式接入它：

```text
累计完整 forward 的 step_counts
    ↓
update_statistics
    ↓
select_hot_set
    ↓
finish_step
    ↓
ResidencyOps.prepare_update(target_keys)
    ↓
实际权重换入 → weights_ready → commit_update
```

图路径每个完整 forward 执行一次这条尾部流程，非周期轮由设备 due 屏蔽交换。实际 eager 路径每步累计 usage 和周期，只在周期到期时执行选集、计划、搬运及发布；其他步骤不产生新的 target_keys，也不消费旧目标。关闭更新时不推进周期。

Eager 的周期由 runtime 中一个 host 整数跟随完整有效 forward 推进，无需读取设备 tensor。`forward_scope` 必须包围一次实际 eager 执行或 graph replay；成功提交后只推进一次，prefill、全局 IDLE、capture/warmup 不推进，失败后沿用 owner 的停止执行约束。Host 周期跨 prefill 保持，设备 LFU 周期随快照恢复，因此 eager 与 graph 交替执行时保持一致。非周期 eager 调用 `cache.finish_step(select_hot_set=False)`，累计 usage 后直接结束，不排序、不生成热池计划、不提交空搬运、不重建映射。到期但无合格交换时仍按规则衰减。图路径保持默认 `select_hot_set=True` 的固定操作序列，设备门控实际交换；此轮没有宣称已消除图路径的非周期排序开销。

职责不越界：

| 组件 | 负责 | 不负责 |
| --- | --- | --- |
| LFU policy | usage、周期、候选配对、目标集合、衰减 | slot 分配、H2D/D2D、通信 |
| ResidencyOps | 集合差分、保留幸存 slot、同层安排换入、发布 | 重新决定热度排名 |
| Runtime / transfer | 路由统计、调用顺序、搬运、事件依赖 | 用未完成复制的目标集合冒充当前缓存 |

Prefill 不更新本策略。快照/恢复复制 usage、step_mod、update_due；正常 forward 结束后 due 为假。恢复周期位置后继续累计，不从第 1 轮重新计数。

## 9. Decode 入图约束

原分支要求完整 decode graph，本策略沿用该约束：

1. usage、周期标量、目标集合及中间缓冲区在捕图前分配，地址和容量固定。
2. `update_due` 是设备 bool；不能取回 CPU 决定是否排序、搬运或发布。
3. 每层最多处理已知数量的本地专家。用固定形状数组、有效 mask 和设备排序表达配对；无效项不参与全层选择。
4. 跨层候选缓冲区按静态专家数量定容量，按热度差和固定 tie-break 选取最多 96 对；不在 replay 中创建 Python list/set。
5. 非周期轮输出当前集合。物理槽位、程序波次和通信顺序不因 LFU 输出而重新构建。
6. `finish_step` 在选取所需的 usage 读取结束后衰减；更新状态用原位写入，排序 workspace 由后端申报预算。

参考实现可用 CPU 验证算法。实际 NPU 排序、固定形状集合构造、graph 支持和性能仍需验证，不能把本文伪代码视为可直接捕获的实现。

## 10. 与已有 TensorLFU 的差异

| 项目 | 现有 `TensorLFUPolicy` | 新 `PeriodicLFUPolicy` |
| --- | --- | --- |
| 在线热度 | int64 频次 | float32 usage |
| 初始装入 | 策略本身不提供离线热启动 | 初始化组件按离线 Top-K 装入 |
| 选取 | 按热度跨层 Top-K | 到周期时层内配对，再全层限量 |
| 层间容量 | 每次选取可改变各层驻留数量 | 初始化后保持每层驻留数量 |
| 替换门槛 | 无 2 / 1.5 门槛 | 最低热度 2，最小差值 1.5 |
| 换入上限 | 由目标集合差异决定 | 每 rank 每周期最多 96 |
| 衰减 | 默认关闭，可配置周期减半 | 默认每 4 步乘 0.7 |
| 同分 | resident 优先，再按 key | 配对内按 key，全层按层号与 key |
| 物理位置 | 不分配 slot | 同样不分配 slot |

例如 K=2，L3/E0、L4/E0 当前驻留且热度均为 1，L3/E1=10、L3/E2=9。在相同热度输入下，原策略会选 L3/E1 和 L3/E2；新策略到期时只将 L3/E0 换成 L3/E1，保留 L4/E0，因为不能借用 L4 的配额。非周期轮则保持原集合。

新策略是显式可选项：`MoEOffloaderV2(..., periodic_lfu_config=...)` 才在 `post_init()` 构造它；未传该配置且未显式指定 `policy` 时，仍使用 `TensorLFUPolicy`。策略实现位于工作树不意味着默认行为已切换，也不等于变更已经提交到分支。

## 11. 最小验收

| 场景 | 预期 |
| --- | --- |
| 离线 Top-K | 跨层选择准确，在线 usage 从零开始。 |
| Batch 与 miss 计数 | 3 个有效路由加 3，miss 计数，padding 不计。 |
| 周期 | 第 1～3 步无替换，第 4 步选择；无本地需求的有效 rank 仍推进。 |
| 加法阈值 | 3.5 对 2 可换，3 对 2 不换，12 对 10 可换，候选小于 2 不换。 |
| 全层限量 | 超过 96 对时取最大差值的 96 对，且无重复 candidate/victim。 |
| 零缓存/全驻留/零热度 | 无交换，但有效周期轮照常衰减。 |
| 无交换的周期轮 | usage 仍乘 0.7；被淘汰和换入专家不特殊清零。 |
| 纯选取 | 调用 select_hot_set 不改变 usage 或 key_to_slot，改变输出排列不改变物理布局。 |
| 禁用更新 | warmup/capture 不改变 usage、周期或驻留；初始 step_mod=0 不误触发。 |
| 恢复 | prefill 前后完整策略状态相同，周期接续。 |
| 连续 replay | 同一 graph 中 usage、到期标志、目标集合能随输入变化，不靠 CPU 分支。 |
| 离线覆盖与层配额 | 全零或缺失统计下的 Top-K、每层 K_l 和零配额层行为可观察；不隐式跨层补配额。 |
| 热度与实际收益 | batch 内重复路由累计热度，权重加载按专家复用；分别报告路由命中与 H2D 字节、等待和延迟。 |
| 大热度、小增量（待修复验收） | 覆盖第 14 节 float32 反例；实现数值保护或明确拒绝不受支持配置后再判为通过。 |

Verify 计数在模型接入后另验：每个目标模型 verify forward 推进一步，拒绝 token 已执行的路由保留计数，draft 不计数。此项不能由 LFU 单测替代。

## 12. 已实现入口与验证范围

使用现有 backend 注入入口，显式选择新策略：

```python
from sglang.srt.offloading.moe import MoEOffloaderV2, PeriodicLFUConfig

offloader = MoEOffloaderV2(
    offload_config,
    backend,
    periodic_lfu_config=PeriodicLFUConfig(
        offline_stats_path="/path/to/expert_frequency.json",
    ),
)
```

`wrap_modules()` 建立 catalog 后，`post_init()` 创建 `PeriodicLFUPolicy` 并传入 `backend.finalize()`。不传 `periodic_lfu_config` 时仍使用原默认策略；它与显式 `policy=` 互斥。也可在已有 catalog 时自行构造策略并通过 `policy=` 注入。

后端接入顺序：

1. 初始化调用 `policy.initial_keys(config.hot_slots(catalog), model_id)`，得到本 rank 的初始专家 key；装入完整权重并发布映射。此方法不会改在线 usage。
2. 通过 `allocate_state()` 分配当前状态及快照状态；目标集合和相关工作区由后端管理并计入预算。
3. 图路径每个完整 forward 尾部调用 `update_statistics → select_hot_set → finish_step`，随后执行现有 residency/transfer 接口；eager 非周期轮跳过选集与 residency/transfer，只累计统计。有效访问计数和更新开关由 runtime 提供。
4. prefill 快照和恢复使用 `copy_state()`，三个状态 tensor 均原位复制。

在线实现按静态层范围发出固定形状的排序、mask、索引和 scatter 操作，不使用 `.item()`、`.tolist()` 或 `.cpu()` 决定选取。所有状态更新保持原 tensor 地址；排序和其他临时 tensor 的实际设备内存峰值需由后端纳入预算。

CPU 验证覆盖离线 rank 筛选、文档例子、层份额不变、加法阈值、全层 96 对上限、禁用与空需求步骤、快照恢复、100 组随机参考对照及动态 tensor 回放。测试文件为 `test/registered/unit/offloading/test_periodic_lfu.py`，并在现有 `test_moe_framework.py` 验证策略注入及调用顺序。

当前本地使用 CPU PyTorch 运行两份测试，共 45 项通过。为避免加载无关的服务端/GPU 依赖，本地测试跳过 `sglang` 顶层公共 API 初始化，直接加载实际 offloading 模块；没有替换 LFU 张量实现。CPU trace 检查固定形状下的动态数据流，不等于 NPU graph 验证；实际权重装入、按需执行、异步发布及 verify 接入不在这次 LFU 实现中完成。

## 13. 与 Strata 的对应关系及差异

核对日期：2026-10-21。源码基线固定为 [Niko1221/Strata commit 9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4](https://github.com/Niko1221/Strata/commit/9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4)。以下主要对照 `generate.cpp` 中 serve 的基础 `adapt()` 路径，不将所有可选模式视为相同实现；后续上游变化不自动改变本文契约。

### 13.1 对应规则

| 本文规则 | 固定 Strata 源码依据 | 对应范围 |
| --- | --- | --- |
| 离线热启动 | [profile 排名格式](https://github.com/Niko1221/Strata/blob/9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4/include/strata/core/expert_cache.hpp#L40-L55) | 都用离线路由排名初始化；本文使用带 model_id 的 JSON 和 rank-local catalog，不兼容其二进制 profile 格式。 |
| 默认 4 轮、衰减 0.7、最小差值 1.5、上限 96 | [默认参数](https://github.com/Niko1221/Strata/blob/9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4/src/program/generate.cpp#L680-L712) | 默认数值对应；轮次与预算的作用范围见下表。 |
| 每个有效 token-expert 路由加 1 | [窗口路由累计](https://github.com/Niko1221/Strata/blob/9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4/src/core/expert_source.cpp#L4536-L4538) | Strata 遍历 n_tok × k；本文在 dispatch 后按本 rank 有效分配跨 wave 累计。 |
| 同层候选、victim 配对；候选至少 2；差值至少 1.5；按收益限量 | [基础 adapt 选取](https://github.com/Niko1221/Strata/blob/9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4/src/program/generate.cpp#L10073-L10099) | 数学规则对应；Strata 还排除由 peer/helper/remote 持有的候选，本文按 EP 所有权定义候选范围。 |
| 选择后统一衰减 | [基础路径衰减](https://github.com/Niko1221/Strata/blob/9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4/src/program/generate.cpp#L10231) | 本文用单独的 finish_step 表达，不将选集和衰减副作用混在一起。 |
| 层内替换维持配额 | [slot 大小与层归属约束](https://github.com/Niko1221/Strata/blob/9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4/include/strata/core/expert_cache.hpp#L96-L98) | Strata 可利用此约束支持不同层 bundle 大小；本文保持现有容量预算，不因此声明支持异构 bundle。 |

Strata 的准入判断使用 `candidate < victim + min_gain` 时跳过，本文使用 `candidate - victim >= min_gain`。两者表达相同数学门槛，但浮点边界的运算顺序不同；不能据此宣称逐 bit 一致，边界验收以本文公式为准。

### 13.2 实现差异

| 方面 | 固定基线的 Strata | 本文及当前工作树 |
| --- | --- | --- |
| 策略运行位置 | 基础 adapt 用主机 C++ vector、sort 和控制分支。 | torch 设备张量与静态层循环；完整 NPU graph 能力仍需验证，不能照搬其 host 排序路径。 |
| 同分顺序 | 基础配对比较器只比较热度或 gain，没有本文的显式层号/key 次序。 | 按层号和 key 定义确定性顺序；相同热度不保证与 Strata 选择相同专家。 |
| 周期与在途更新 | 基础 adapt 在 pending 非空时直接返回，不发生该次选取及衰减。 | 每个有效完整 forward 推进一步；更新在尾部完成，下一 forward 等待资源就绪，不照搬跳轮逻辑。 |
| 多卡与上限 | 基础 adapt 对其管理的层合并排序；还有 layer-split、peer/helper 等路径。 | 每个 EP rank 独立统计、配对和限制 96；不是所有 rank 合计 96，也不声称复现其多卡调度。 |
| 零交换配置 | 基线存在以 adapt_every、adapt_swaps 是否大于零决定启用 usage 的路径。 | update_interval 必须为正；max_swaps=0 仍累计、推进周期并衰减。 |
| miss 执行 | 可由 CPU 计算，或分配给 GPU 的 PCIe 路径。 | miss 权重进入临时执行区后由 NPU 计算；周期准入是另外一件事。 |
| RAM 副本与换出 | complement 模式可在 RAM 只保留 GPU 未驻留专家，因此交换需要 D2H；arena 模式不同。 | CPU 保留完整只读权重，淘汰无需 D2H 回写。 |
| 异步更新 | 可选异步模式用快照、事件和状态机跨窗口推进。 | 基础方案不做跨 forward 的热池并发更新，不承诺尾部换入与下一 forward 重叠。 |

依据：[基础 adapt 及 pending 判断](https://github.com/Niko1221/Strata/blob/9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4/src/program/generate.cpp#L10073-L10099)、[CPU/GPU 分工](https://github.com/Niko1221/Strata/blob/9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4/src/core/expert_source.cpp#L4539-L4542)、[异步交换与 RAM 生命周期](https://github.com/Niko1221/Strata/blob/9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4/src/program/generate.cpp#L10234-L10272)。这些实现差异是适配边界，不是策略等价测试可以覆盖的内容。

## 14. Review 后的数值限制与待修复项

当前 `PeriodicLFUPolicy` 使用 float32 usage，配置检查仍允许 `usage_decay=1`；尚未实现长期累计的精度保护。在 float32 下，从 `2**24 = 16777216` 开始逐次加 1，重复 100 次仍可能停在 16777216。候选因此无法体现新增访问，可能不能超过已不再被访问的 victim。固定 Strata 基线同样使用 [float usage](https://github.com/Niko1221/Strata/blob/9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4/include/strata/core/expert_source.hpp#L618-L620) 和逐次加 1；该问题不是相对 Strata 新增的算法差异。

默认 0.7 衰减通常限制热度规模，但不构成任意 batch 大小、更新间隔和衰减参数下的精度证明。长期运行验收前，需选择并实现一种明确方案：为不衰减模式提供足够宽的整数累计及对应比较，或根据最大计数范围限制并拒绝不受支持的参数组合；不能只截断热度并声称仍保持原 LFU 排序语义。此项当前未修复，本次文档更新不代表代码已具备该保护。

第 12 节的既有 CPU 验证记录不覆盖本节大热度边界，也不替代新增的 H2D/延迟与离线覆盖验收。
