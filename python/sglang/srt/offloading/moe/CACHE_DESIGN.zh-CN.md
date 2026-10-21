# MoE 专家 Cache 基础功能设计

适用分支：`codex/moe-offload-dev`

状态：周期 LFU、元数据适配，以及 cache 调用方的初始化、实际需求计划、slot 替换、传输调用和就绪后发布已实现。见 [LFU 独立设计](LFU_DESIGN.zh-CN.md)、第 14 节及 [调用链与 H2D 交接表](CACHE_RUNTIME_HANDOFF.zh-CN.md)。CPU 测试搬运器验证调用链；真实 NPU transport、模型原生绑定和完整 graph 联调尚未完成。

本设计实现离线热启动、在线计数、周期层内替换、换入量限制与热度衰减，以及搬运完成后发布映射。复用现有专家目录、缓存状态、执行 bank 和 owner 生命周期，以基本功能跑通为目标。

## 0. 原分支基线与变更边界

核对基线为 `moe-offload-proto-v1` 的 `5b13541370f4b3f4746ea4b88054801b6a278c81`，框架代码提交为 `65e5c53ebbced0b06a8c3a1a253593a4caebe14f`。以下区分已有契约与本设计新增内容：

| 项目 | 原分支事实 | 本设计处理 |
| --- | --- | --- |
| Decode 权重准备 | dispatch 后得到本 rank 实际需求，再调用 `resolve_execution(local_demand)`。 | 保持按需加载；不扩大为 wave 全专家加载。 |
| 固定 waves | 固定 D/F/C 次数和顺序，`LocalExpertProgram` 接收 dispatch 结果。 | 保持；wave 范围不等于加载清单。 |
| 热池容量 | `floor(cache_ratio * catalog.num_experts)`。 | 原样保留。 |
| `target_keys` | 目标专家集合，不分配 slot。 | 原样保留；物理位置由 residency 决定。 |
| 执行区 | 总容量显式预算，`fixed_waves()` 默认一个 bank。 | 不强制双 bank、正偶数或新增容量。 |
| 默认 LFU | 跨层选 Top-K，默认不衰减，可选周期减半。 | 新增独立周期策略实现五点需求，不将其误写成已有行为。 |
| Prefill | 层开始全量流式预取、借热池、恢复原 slot 和策略状态。 | 保持执行路径，只补充新策略状态的保存恢复。 |
| Verify | 原适配器和 ModelRunner 明确拒绝投机解码。 | 只定义待接入语义，不声明当前已支持。 |
| NPU 后端 | 动态传输、residency、真实专家执行仍为接口。 | 列出实现落点，不声称已跑通。 |

依据：[原 README 执行时序](https://github.com/pachi-33/sglang/blob/5b13541370f4b3f4746ea4b88054801b6a278c81/python/sglang/srt/offloading/moe/README.zh-CN.md#3-uml从当前框架生长)、[config.py](config.py)、[interfaces.py](interfaces.py)、[programs.py](programs.py)、[owner.py](owner.py)、[policy.py](policy.py)。

## 1. 五项需求与确定行为

| 需求 | 基础版行为 |
| --- | --- |
| 离线初始化 | 每个 EP rank 将全部可卸载 `(layer, expert)` 按离线路由频次排序，选前 K 个装入 device 热池。 |
| 在线计数 | 每个有效 token-expert 路由令 `usage[key] += 1`，命中和 miss 都计数。 |
| 周期替换 | 每 4 个目标模型 decode/verify 完整 forward 检查一次；每层将最热非驻留专家与最冷驻留专家一一配对。 |
| 门槛与限量 | 候选 `usage >= 2`，且 `usage >= victim_usage + 1.5`；所有层合格配对按热度差排序，每 rank 每轮最多交换 96 个专家。 |
| 衰减与发布 | 每轮选择完成后，所有 usage 乘 0.7；新权重完整复制后才发布驻留映射。异步搬运沿用后端 stream/event；计算重叠只在真实数据依赖允许时安排，见第 8 节。 |

“高 1.5”定义为热度差至少 **1.5**，不是倍数。两个阈值都包含等号。这与核对到的 [Strata adaptive 判断](https://github.com/Niko1221/Strata/blob/main/src/program/generate.cpp#L9952-L9978) 一致；原分支默认 TensorLFU 本身没有该门槛。

初始化跨层竞争 K 个位置；后续按用户需求只做层内一换一，因此各层驻留数量保持初始化形成的分配。这里有意改变原 TensorLFU 每轮跨层竞争的策略，不改变热池的物理容量和公共接口。某层初始化没有驻留专家时，该层仍能通过 miss 路径计算，但本策略不会为它新增热 slot。

## 2. 范围与基本约束

- 只缓存可卸载 routed experts；前三层、hash 层及 shared experts 沿用现有常驻规则。
- 每个 EP rank 只管理自己负责的专家。热度、周期和 96 个换入上限均按 rank 独立计算，不同步全局排行榜。
- CPU pinned DRAM 保存完整、只读的权重 bundle；淘汰不做 D2H 写回。bundle 包含权重、scales 和格式要求的其他分量。
- forward 内热集合不变；miss 使用临时执行区，不能立即挤入热池。
- 两个 forward 不重叠使用资源。cache 替换放在 forward 尾部，下一次 forward 等待完成。
- prefill 借用热池，不更新热度、不推进替换周期；退出时恢复 decode 的精确布局和策略状态。
- 基础版不增加预测器、跨 rank 热度协调、在线容量调整、复制失败重试、回滚事务或额外权重备份池。正常读写依赖仍必须满足。

当前分支拒绝投机解码。本设计定义 verify 的 cache 行为，但实际启用 verify 需要同步实现模型/runtime 接入；不能只删除限制就认为支持。先完成 decode 路径，再用同一套接口接通 verify。

## 3. 配置与容量

建议新增 `PeriodicLFUConfig`，由策略对象持有，不改变现有 offload 参数的含义：

```python
@dataclass(frozen=True)
class PeriodicLFUConfig:
    offline_stats_path: str
    update_interval: int = 4
    min_usage: float = 2.0
    replacement_min_gain: float = 1.5
    max_swaps: int = 96
    usage_decay: float = 0.7
```

设当前 rank 在可卸载层 `l` 有 `N_l` 个专家，缓存比例为现有 `cache_ratio`：

```text
N = sum(N_l)
K = floor(cache_ratio * N)
```

沿用 `MoEOffloadConfig.hot_slots()` 的结果。初始化全局 Top-K 后，令 `K_l` 为其中属于层 l 的专家数量；不是预先按比例计算 `K_l`。运行期一换一保持该数量，`K_l=0` 的层只走临时执行，`K_l=N_l` 的层没有换入候选。

例如 N=8、K=4，离线排序前四名为 L3/E4、L3/E5、L3/E6、L4/E5，则 L3 有 3 个热专家、L4 有 1 个。保持原 K 的计算，不修改 config 和 owner 的容量公式。

临时执行区沿用原分支总预算：

```text
所有物理执行 bank 的 slot 总数 = execution_slots
```

基础 decode 可用一个 bank。若后端采用双缓冲，两个 bank 都必须计入该总数；本策略不要求正偶数配置，也不额外申请权重池。prefill 可按原设计借用热池视图，视图不新增物理分配。仍沿用原分支同构 bundle 的字节预算。

## 4. 离线统计与热启动

离线文件以层号和全局专家编号保存频次。例如：

```json
{
  "model_id": "checkpoint-used-for-profiling",
  "layers": {
    "3": {"4": 1200, "5": 900, "6": 500, "7": 100},
    "4": {"4": 300, "5": 1800, "6": 600, "7": 200}
  }
}
```

`model_id` 标识对应 checkpoint；部署时提供与模型匹配的文件。各 rank 从中筛选自身专家，缺少的专家频次按 0 处理。只需初始化时读取一次；文件无法读取则报配置错误，不增加备用初始化流程。

初始化顺序：

1. 建立 `ExpertCatalog`，确定可卸载层及本 rank 专家。
2. 使用原 `hot_slots(catalog)` 计算 K。
3. 在本 rank 所有可卸载专家间，按离线频次降序、专家 key 升序排序，选择前 K 个。
4. 从 CPU 加载这些专家到固定热 slot，等待完整 bundle 就绪。
5. 写入双向映射和有效位；初始化完成后才允许模型 forward。
6. 在线 `usage` 全部置 0，`step_mod` 置 0。

离线频次只用于初始装入，不直接加到在线 usage，避免离线大样本计数长期压制在线变化。频次相同或全为零时，按专家 key 决定顺序，初始化填满总计 K 个 slot。各层 slot 不要求连续；通过 `slot_to_key` 和专家层号即可确定归属。

## 5. 最小数据结构

复用 [state.py](state.py) 的 `ResidencyState`：

| 字段 | 含义 |
| --- | --- |
| `key_to_slot[N]` | 专家 key 到热 slot，未驻留为 -1。 |
| `slot_to_key[K]` | 热 slot 当前存放的专家。 |
| `slot_valid[K]` | 完整权重是否可读。 |
| `slot_generation[K]` | slot 复用版本，沿用现有框架。 |
| `execution_key[S]`、`execution_valid[S]` | 执行 bank 的内容标识。 |
| `step_access_count[N]` | 本次完整 forward 的有效访问次数。 |
| `update_enabled` | warmup/capture 时禁止更新，真实执行时允许更新。 |

策略新增状态：

```python
@dataclass
class PeriodicLFUState:
    usage: object       # device float32[N]，支持乘 0.7
    step_mod: object    # device int64 标量，值为 0..update_interval-1
    update_due: object  # device bool 标量：本次是否执行周期替换/衰减
```

初始化固定 `layer_key_ranges` 或等价的 `key_to_layer`，供策略和 residency 判断专家层号。无需新增连续的层 slot 配额表。

预分配 `target_keys[K]`，语义仍是目标专家集合；数组下标不表示物理 slot。非更新轮输出当前驻留集合，runtime 差分得到零交换。策略内部排序或输出顺序不能移动幸存专家。

没有单独的“待准入专家缓存”。临时执行区可以被后续波次覆盖，因此尾部晋升热专家时允许再次 H2D。

## 6. 在线计数与 miss 执行

### 6.1 计数规则

一次 step 是目标模型一个 batch 的一次完整 decode/verify forward。

在 EP dispatch 后，根据有效本地 token-expert 分配统计：

```python
# 说明语义的伪代码；实际实现为设备算子。
for assignment in valid_local_assignments:
    step_access_count[assignment.expert_key] += 1
```

- 多个 token 选择同一个专家，分别加一。
- top-k 中每个有效专家分配分别计数，不按 router 权重加权。
- 不按搬运次数计数；一个专家加载一次、处理 5 个 token，计 5 次。
- 每个分配只在它所属波次记录一次，padding 和无效项不计数。
- prefill、warmup、capture 不计数，也不推进周期；没有有效 token 的全局 IDLE 不推进。
- 周期按 batch 的 forward 推进；某个 rank 没有本地路由时计数为零，但仍随同一有效 batch 推进一步。不能用本 rank 的计数和是否为零判断全局 IDLE。

verify 统计目标模型实际执行的有效路由，拒绝的候选 token 不回滚计数，因为计算已经发生。draft 模型不计入本状态。一个 verify batch 只推进一步，不按候选 token 数推进。

### 6.2 本次计算的权重来源

```text
命中：热 slot → D2D → 执行 bank → 专家计算
miss：CPU pinned DRAM → H2D → 执行 bank → 专家计算
```

基础版采用连续 bank，兼容现有算子输入；直接读取任意热 slot 暂不实现。同一波内同一专家只准备一份权重，供它的多个 token 共用。

本地专家数大于 bank 容量时，沿用固定 waves 和 `ComposedDFCProgram`；所有 rank 保持相同 D/F/C 波次顺序，包括空波。完整顺序保持：

```text
mask 当前 wave 路由 → dispatch → 本 rank 有效需求
→ resolve_execution(local_demand) → 按需准备权重 → F → combine
```

例如 wave 范围为 E4/E5，但实际只收到发往 E5 的 token，则只准备 E5；E4 的计划项无效，不触发搬运或读取未就绪权重。空波保留通信顺序，不加载无关专家。按需加载是基础行为，不是后续优化。

## 7. 周期层内替换算法

### 7.1 每个完整 forward 的尾部

```python
# 算法参考伪代码，不是 graph 内的 Python 执行实现。
# update_statistics：每次都刷新 due，禁止沿用上一步的触发标志。
update_due = False
if update_enabled:  # 已包含真实有效 decode/已接通 verify 的 forward 条件
    usage += step_access_count
    step_mod = (step_mod + 1) % update_interval
    update_due = (step_mod == 0)

# select_hot_set：只决定目标集合，不写任何物理 slot。
target = set(current_resident_keys)
if not update_due:
    target_keys[:] = sorted(target)
    return

eligible_pairs = []

for layer in offloaded_layers:
    # 只看这个 rank、这个层的专家。
    hot = resident_experts(layer)
    cold = nonresident_experts(layer)

    # 同分时按 expert key 升序，保证确定性。
    hot.sort(key=lambda e: (usage[e], e))
    cold.sort(key=lambda e: (-usage[e], e))

    # 一一配对，不重复使用候选或被替换专家。
    for candidate, victim in zip(cold, hot):
        if (usage[candidate] >= min_usage
                and usage[candidate] >= usage[victim] + replacement_min_gain):
            eligible_pairs.append((
                usage[candidate] - usage[victim],
                layer.id,
                candidate,
                victim,
            ))

# 本 rank 所有层合并后排序，最多选择 max_swaps 对。
eligible_pairs.sort(key=lambda p: (-p[0], p[1], p[2], p[3]))
for gap, layer_id, candidate, victim in eligible_pairs[:max_swaps]:
    target.remove(victim)
    target.add(candidate)

target_keys[:] = sorted(target)

# runtime 在本次选取之后调用新策略的 finish_step()，每步一次。
# 非周期轮和禁用更新时不衰减。
if update_due:
    usage *= usage_decay  # 所有专家，无交换也执行。
update_due = False
```

配对和排序始终基于同一次热度快照；本轮不会“换入后再拿新成员参与下一次配对”。热池在初始化时已填满，正常运行通过一换一保持各层驻留数量，不需要额外补洞算法。上面的非周期 `return` 仅表示提前结束选取；runtime 仍调用 `finish_step()`，此时它是无操作。

### 7.2 具体例子

第 4 个 forward 计数完成后：

| 层 | 当前驻留专家及 usage | 非驻留专家及 usage |
| --- | --- | --- |
| L3 | E4=2、E6=5 | E5=8、E7=3 |
| L4 | E4=1、E7=4 | E5=7、E6=2 |

配对结果：

```text
L3：E5(8) 替换 E4(2)，满足门槛，差值 6。
L3：E7(3) 对 E6(5)，不满足差值至少 1.5，不换。
L4：E5(7) 替换 E4(1)，满足门槛，差值 6。
L4：E6(2) 对 E7(4)，不满足差值至少 1.5，不换。
```

默认上限 96，本轮两对都执行，各层 slot 数量不变。如果测试把上限设为 1，差值相同时按层号选 L3。随后所有 usage 乘 0.7，例如 L3/E5 从 8 变成 5.6。

## 8. 异步搬运与映射发布

### 8.1 按需执行与异步重叠的边界

Decode 采用原分支的顺序式 D/F/C，最小执行路径为：

```text
dispatch 得到实际需求
copy stream：等待旧 bank 读完 → 搬运本波所需权重 → weights_ready
compute stream：等待 weights_ready → 本波 F → read_done
combine 完成本波，再进入下一波
```

原 `ComposedDFCProgram` 不会在 F0 前先完成 D1，因此不能保证“计算 wave0 时已知 wave1 的完整实际需求”。本 cache 策略不修改这个顺序，不传 `local_demand=None`，也不为获得重叠而搬运 wave 全部专家。

第五点中的异步搬运与计算重叠按下面边界落实：

- 搬运使用异步接口和事件依赖；异步提交本身不代表 GPU/NPU 上有计算重叠。
- Prefill 沿用原分支 layer-start 全专家流式预取及有预算的 bank 流水；这条路径有提前已知的加载清单，可与独立计算重叠，具体后端仍待实现。
- Decode 首先按需打通。后端若要在一次本地 F 内将已知需求进一步分组、流水搬运与计算，可在 `LocalExpertProgram` 实现，但必须计入执行区预算，并保持外层 D/F/C 配对。该优化不是本 cache 策略已实现的能力。
- 尾部缓存换入等待旧读者并在下一 forward 前完成，不为保证它与计算重叠另建 staging 热池。

因此基础版先验收按需加载、异步正确性和发布顺序；如果交付要求 decode 搬运必须实际隐藏在计算后面，需要单列本地执行流水任务并用设备时间线验收，不能把该项当成现有框架自动满足。

### 8.2 热集合更新

缓存替换只在当前 forward 尾部进行：

```text
准备交换计划
    ↓
copy stream 等待 victim slot 的旧读者结束
    ↓
将 candidate 完整 bundle 写入同层待替换 slot
    ↓
记录 weights_ready
    ↓
发布流等待 weights_ready，提交映射
    ↓
下一次 forward 等待更新及其他资源使用结束
```

`prepare_update()` 对当前集合和目标集合做差：幸存专家保留原 slot；每层新增专家与该层被移除专家的 slot 配对，生成实际搬运及发布计划。策略已确定淘汰哪些专家，不要求策略数组位置决定物理配对。相同层 bundle 格式一致，按 key/slot 固定顺序配对即可。

发布时，使用该计划中的 candidate、victim、slot：

```python
key_to_slot[victim] = -1
key_to_slot[candidate] = slot
slot_to_key[slot] = candidate
slot_valid[slot] = True
slot_generation[slot] += 1
```

这些写入作为同一个受依赖保护的发布阶段完成；期间没有新 forward 读取映射。不要求提供跨 forward 并发读写能力。旧映射在复制时保留，但旧读者已排空，不能再次据此执行。

`residency_ready` 表示发布结束，`resources_idle` 汇合全部 copy、compute、communication 和 prefetch。warmup/capture 的禁用开关同时覆盖 residency commit，不能只关闭 usage 更新。基础模式没有跨 forward 并发读取，不需要另加事务回滚设计。

## 9. 与现有接口的对应

策略新增 `PeriodicLFUPolicy(CachePolicy)`，保留现有四个公共方法的含义：

```python
allocate_state(num_experts, device) -> PeriodicLFUState
# 分配 usage、step_mod、update_due；专家层号由构造参数固定。

update_statistics(state, step_counts, update_enabled)
# 允许更新时累加 usage 并推进 step_mod；写 update_due，不在此处衰减。
# 禁用时保持 usage/step_mod 不变，并清空 update_due。

select_hot_set(state, key_to_slot, target_keys)
# 每次 forward 调用一次。
# 普通轮输出当前驻留集合；周期轮执行配对、限量，输出新目标集合。
# 不改变 usage、不分配 slot、不搬运、不发布。

copy_state(source, destination)
# 复制完整策略状态，供 prefill 快照与恢复使用。
```

新策略额外提供一个具体方法，不改动 `CachePolicy` 抽象基类：

```python
finish_step(state)
# update_due 为真时对所有 usage 乘 usage_decay；然后清空 update_due。
# runtime 在 select_hot_set 之后调用一次，无交换也衰减。
```

这是本策略新增的明确接入点，现有 runtime 没有此调用。它避免把衰减副作用藏进 `select_hot_set()`，也不需要让策略绑定整个 `ResidencyState`。runtime 组装该策略时采用：

```text
update_statistics → select_hot_set → finish_step
→ prepare_update → materialize_into → commit_update
```

这些设备操作按依赖顺序录入图。每次真实 forward 执行一次；关闭更新时目标保持当前集合、计划为空且 commit 禁用，初始 `step_mod=0` 不会误触发替换。

复用 `ResidencyOps`：

```python
resolve_execution(local_demand, context) -> DeviceCopyPlan
# dispatch 后调用，只对本波有效本地需求生成加载计划。
# 没有需求的专家不搬运；不新增 None 预取调用约定。

prepare_update(target_keys, context) -> update_plan
# 对比当前驻留集合和 target_keys；幸存专家原位保留。
# 本策略下把每层新增专家放到本层被淘汰专家腾出的 slot。

commit_update(update_plan, weights_ready, context)
# 等完整 bundle 就绪后更新映射。
```

`DeviceTransferOps.materialize_into()` 执行计划，不决定热门专家，也不修改 LFU。`LocalExpertProgram` 负责 bank 绑定、计算和读者结束事件。`BoundMoERuntime.emit_step_end()` 负责组合统计、选择、搬运、提交。

`resolve_execution()` 和 `target_keys` 的公共语义不变。新的 residency 后端需要知道专家所属层，才能把同层更换落实到物理 slot；不能复用任意其他层的 victim slot。

算法伪代码中的动态列表和 Python 分支只用于说明语义。当前框架要求完整 decode graph，因此实际计数、配对、Top-K 选取、计划生成和发布使用固定地址设备 tensor/算子；不能 `.item()`/`.tolist()` 后由 CPU 决策，也不能每轮重新捕图。离线初始化和模式切换可在 host 执行。

## 10. Prefill 借池与恢复

沿用现有 owner 的切换顺序：

1. 进入 prefill 前排空前序工作，保存映射、有效位、generation 及完整策略状态（usage、step_mod、update_due；已结束 step 的 due 为假）。
2. 借用热池做流式执行；连续 prefill 不重复保存，不计数、不衰减。
3. 切回 decode 时从 CPU 将原专家加载回原 slot，再恢复全部元数据和策略状态。
4. 等待恢复完成，继续已有 decode graph；例如进入前 `step_mod=2`，恢复后仍为 2。

快照只保存元数据和策略状态，不复制一份热权重。基础版不额外维护 prefill 热度。

## 11. 代码修改清单

| 位置 | 基础版修改 |
| --- | --- |
| `config.py` / `owner.py` | 保留热池容量公式、执行区总预算和现有生命周期，不因策略要求修改。 |
| 新增 `periodic_lfu.py` | `PeriodicLFUConfig`、状态、周期配对与限量衰减算法；现有 `TensorLFUPolicy` 可保留用于比较。 |
| `offloader.py` / 具体 backend | 显式注入新策略；读取离线统计，在 finalize 阶段加载初始热集合。 |
| `state.py` | 复用已有状态，不把 usage 放进公共 ResidencyState。 |
| 具体 `ResidencyOps` | 按真实需求生成 miss 计划；目标集合差分、同层分配 victim slot、完成后发布。 |
| 具体 `DeviceTransferOps` / `LocalExpertProgram` | 固定 bank 的按需 H2D/D2D、就绪与读完事件；prefill 预取沿用原执行程序。 |
| 具体 `BoundMoERuntime` | 有效路由计数、step-end 接入新策略 `finish_step`、完整更新开关、策略快照/恢复。 |
| 模型和执行入口 | 接通 decode；接通目标模型 verify 的计数和模式识别后，再调整现有投机解码限制。 |
| 测试 | 策略参考测试及 NPU 搬运/执行联调。 |

本设计不实现新的 CPU offload 框架或新的内存池管理器；继续使用 `HostExpertStore`、`ResourceLedger`、`MoEOffloadOwner`。新增具体后端来填补接口。

## 12. 最小验收用例

| 用例 | 预期结果 |
| --- | --- |
| 离线初始化 | 每 rank 跨层 Top-K 正确装入，K 保持原公式；允许某层零热 slot，在线 usage=0。 |
| Batch 计数 | 3 个有效 token 使用同一专家，计 3；miss 同样计数，padding 不计数。 |
| 周期 | 前 3 个有效 forward 不换；第 4 个尾部更新；prefill/warmup/IDLE 不推进。 |
| 门槛 | candidate=3.5、victim=2 可换；candidate=3、victim=2 不换；candidate=12、victim=10 可换；candidate<2 不换。 |
| 层内配对和全层限量 | 每层 slot 数量不变；构造超过 96 对，只选择本 rank 差值最大的 96 对。 |
| 衰减 | 使用衰减前热度选取；周期轮所有 usage 乘 0.7，无交换也衰减。 |
| 按需执行 | wave 覆盖 E4/E5、实际只需 E5 时只加载 E5；空波无多余权重搬运，保持 D/F/C 顺序。 |
| Miss 执行 | 临时加载后能正确计算，但周期准入前仍显示非驻留。 |
| 集合与物理位置 | 改变 target_keys 的排列不移动幸存专家；新专家只占本层被淘汰专家的 slot。 |
| 发布顺序 | 复制完成前没有新映射可被下一轮使用；执行 bank 不覆盖未读完的权重。 |
| Graph | 同一 graph 多次 replay 改变路由，hit/miss、替换和 usage 随之改变；warmup/capture 不改变 usage、周期或热布局。 |
| 模式切换 | decode → prefill → prefill → decode 后，恢复原 slot、usage 和周期位置。 |
| Verify | 实际目标模型路由计数；拒绝 token 不回滚；draft 不计数；每个 verify forward 只推进一步。 |
| EP 周期 | 一个有效 batch 在某 rank 没有本地路由时，该 rank usage 不增加，但周期仍推进。 |
| 异步及预取 | 就绪/读完依赖正确；prefill 后端的真实重叠单独用时间线验收，不用 decode 全 wave 搬运凑出重叠。 |

实现顺序：离线装入与固定缓存 → decode miss 和计数 → 周期替换与衰减 → prefill 恢复 → verify 接入。动态搬运和分波计算的同图可行性应在最小单层路径先验证；无需先完成整模型才能发现该限制。

## 13. 本次基线 Review 修正记录

| 原文问题 | 修正结果 | 原分支依据或需求来源 |
| --- | --- | --- |
| Decode 为重叠而准备整个 wave 的权重，并允许 local_demand=None。 | 删除；恢复 dispatch 后按实际需求加载。 | `interfaces.py` 的 `emit_after_router`、`resolve_execution`，`programs.py` 调用顺序。 |
| target_keys 的下标被解释成物理 slot。 | 恢复目标集合语义，由 residency 保留幸存 slot、分配同层 victim slot。 | `CachePolicy.select_hot_set` 明确不分配 slot。 |
| 每层提前按比例取整，改变热池总容量。 | 恢复原 K 公式，跨层离线排序，层占比由初始化结果形成。 | `config.py::hot_slots` 和用户离线频次排序需求。 |
| 强制双 bank 和正偶数 execution_slots。 | 删除新增限制，沿用总预算，按实际执行后端选择 bank。 | `config.py` 只要求正数，`fixed_waves` 默认单 bank。 |
| 门槛写成 1.5 倍。 | 改为热度差至少 1.5，并增加区别两种公式的例子。 | 用户“高 1.5”及核对到的 Strata 加法判断。 |
| 声称靠异步 copy 就能在现有 decode 顺序下实现跨波重叠。 | 明确真实依赖和实现缺口，不声称框架已提供重叠。 | 原顺序 D0/F0/C0 → D1/F1/C1。 |
| 在 select_hot_set 中隐式衰减，并额外绑定公共状态。 | 新策略显式 finish_step；保持选集和 slot 职责分离。 | 新周期衰减需求的最小 runtime 接入扩展。 |
| 未区分策略新增、原框架既有及未实现后端。 | 第 0 节列清边界，verify 和 NPU 路径标明待接入。 | 原 README 实现状态及 ModelRunner/适配器限制。 |

## 14. 无需 H2D 联调即可使用的元数据组件

[cache_metadata.py](cache_metadata.py) 提供 `CacheMetadataAdapter`，接收已有 catalog、residency 和 `PeriodicLFUPolicy`。它不实现物理 slot 分配、`DeviceCopyPlan`、权重复制、映射发布或事件编排，不预先约束传输流水接口。

### 14.1 已实现的调用序列

```python
cache = CacheMetadataAdapter(catalog, residency, policy)  # 捕图前分配

# 以下调用作为一次完整 forward 内的设备操作。
# enabled 是固定地址的 device bool 输入，replay 前由 runtime 更新其内容。
cache.begin_step(enabled)

# 在 EP dispatch 后提供有效 token-expert 分配：
# local_ids 是该层、本 rank 内的专家编号，valid 标记真实行而非 padding。
cache.record_local_routes(layer_id, local_ids, valid)

# 所有层、所有 wave 结束后只调用一次。
delta = cache.finish_step()
# delta.keep / admit / evict 都是 bool[N]，按 expert key 索引。
# cache.target_keys 是目标集合；当前 residency 没有被修改。
```

`record_global_routes()` 可将连续 EP 分片下的全局专家编号转换为 local ID；其他 rank 的专家、负编号、越界项及 padding 不计数。若调用方持有的是每专家聚合 token 数，需要先另接计数适配，不能把一个专家编号当成只有一次访问。

两种分波计数接法二选一：

- 每波只传该波 dispatch 后的有效分配，每条分配只提交一次。
- 同一份完整路由张量用于多个 wave 时，传入互不重叠的 `local_start/local_end` 区间。区间由未来实际执行程序提供；组件不预设 bank 数或 wave 大小。

`stats.accesses/hits/misses` 是本次有效 token-expert 分配数。命中检查 key 到 slot、有效位及反向映射；不把“一个专家只搬一次”误计成一次访问。统计保留在设备 tensor 中，不在 replay 内读回 host。它不提供搬运字节数或流水时间等依赖后端的指标。

### 14.2 模式开关与集合差分

`cache_update_enabled(forward_mode, has_global_tokens=..., warmup=..., capture=..., verify_enabled=False)` 在 host 根据真实调度信息计算开关，随后写入 replay 输入 tensor。不要把捕图期间的 False 固化为图内常量。

传入 EP 协同执行所用的逻辑模式；不能直接用某个 rank 局部 IDLE 替代全组 decode 模式，也不能用本地计数和为零判断无有效 batch。prefill、warmup/capture、全组无 token 均关闭更新。verify 默认关闭；显式开启只提供计数语义，不解除 ModelRunner 现有的投机解码限制。

`CacheSetDelta` 只计算当前集合与目标集合的交、差集，不决定专家放到哪个 slot。层内一换一由 LFU 保证，后续具体 residency 再将 admit/evict 转为双方约定的物理更新计划。

### 14.3 快照及资源归属

- `save_decode_metadata()`：在前序工作已排空时保存热池双向映射、有效位、generation 和完整 LFU 状态。保存空间已预分配。
- 后续权重恢复实现可读取 `cache.snapshot.slot_to_key/slot_valid`，确定原专家及原位置；本组件不生成 H2D 计划。
- `restore_decode_metadata_after_refill()`：**仅在原权重已恢复到原 slot 后**还原元数据和 LFU 状态。方法本身不等待搬运，也不证明权重就绪。
- 临时执行区内容不随快照保存，恢复时将 `execution_key/execution_valid` 失效，避免宣称旧权重仍在 bank 中。
- `cache.allocations()` 报告组件持有的状态、快照、集合差分和统计 tensor。`residency.allocations()` 单独报告。排序等算子临时 workspace 仍由 backend 计入预算。

### 14.4 框架接入位置与尚待对齐部分

| 已有位置 | 当前接法 |
| --- | --- |
| `MoEBackend.finalize()` | 可构造 `CacheRuntime` 并调用 `initialize(model_id)`；真实 backend 注入权重 bindings、程序和 transport。 |
| `BoundMoERuntime.emit_step_begin()` | `CacheRuntime` 调用 `cache.begin_step(enabled)`。 |
| EP dispatch 后的实际分配 | `CacheLocalExpertProgram` 调用路由 adapter、计数、按实际需求准备 bank，再调用原生 F。具体 dispatcher 的 adapter 仍需提供。 |
| `BoundMoERuntime.emit_step_end()` | 已接 `finish_step -> prepare_update -> materialize_into -> commit_update`。 |
| `ModelRunner.forward_scope()` | `CacheRuntime.forward_scope` 处理全局模式、owner 生命周期及本次提交的新完成事件。 |
| `ModelRunner` eager 分支 | 新 `eager_step_scope` 包围实际 eager 调用，graph 分支绕过此入口。 |
| `capture_head_hooks/capture_tail_hooks` | `CacheRuntime.install_model_hooks` 注册一次；图 replay 不重新执行 Python 计数。真实 NPU capture 验证仍待完成。 |
| owner 的 prefill/decode 切换 | 已接快照、借池 callback、原位置回填调用、完成依赖、元数据恢复。实际借池程序及 bindings 由后端提供。 |

没有安装假硬件 backend 或默认 serving 开关。调用方已接线，测试实现验证流程；真实物理布局、NPU 动态复制与原生模型程序仍通过后端注入。调用方已经采用的计划格式及待确认项单独列于 [H2D 交接表](CACHE_RUNTIME_HANDOFF.zh-CN.md)。

### 14.5 验证

- CPU：`test/registered/unit/offloading/test_cache_metadata.py` 覆盖 batch/EP 计数、padding、互斥 wave、模式开关、同层差分、快照、资源核算和动态 tensor 回放。
- NPU：`test/manual/test_moe_cache_metadata_npu.py` 可单独验证 LFU、计数和集合差分的 `NPUGraph` 捕获及变化输入回放，不依赖专家权重 H2D。在 Ascend 环境执行 `python test/manual/test_moe_cache_metadata_npu.py`。
- NPU 小规模完整流程：`test/manual/test_moe_cache_runtime_npu.py` 新增 3 组测试，使用真实 H2D/D2D、事件等待、两块矩阵组成的专家权重，验证 LFU 替换、发布和回填恢复。它通过主机读取计划进行 eager 测试，不支持动态搬运入图；运行方式及验收标准见 [交接表第 6 节](CACHE_RUNTIME_HANDOFF.zh-CN.md#6-npu-小规模完整流程测试)。
- 本地没有 `torch_npu`，不能宣称 NPU 算子/捕图已验证。CPU trace 只检查固定形状下的数据流；其静态形状判断警告不表示支持动态 shape 或真实设备图。
- 当前本地回归：调用链测试 14 项、NPU 脚本参考计算的 CPU 检查 2 项，与元数据、LFU 和原框架测试合计 75 项 CPU 测试通过。本机没有 `torch_npu`，新增 NPU 完整流程脚本尚未实机验证。Ruff 检查及格式、isort 和空白检查通过。
