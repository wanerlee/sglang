# Decode 中 ScatterElements 的用途与耗时分析

代码基线：`codex/moe-offload-dev-20261027` / `531e228a`。

**这 7 次 ScatterElements，主要是在重复整理六份权重 component 的复制地址。
它们处理的是小型 INT64 元数据，但每次都是独立操作，并且串行挡在权重搬运前。**

这里的七次对应 W8A8、单个 wave 不超过 256 个专家的搬运路径。代码能解释
“为什么有七次、为什么总开销会累加”；单次为什么达到截图中的约 75.9 μs，
还需要结合实际执行类型和输入规模判断。

## 1. 第一次：检查目标 slot 是否被重复写入

[对应代码：transfer.py](../transfer.py#L442)：

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
不重复的序列，见[初始化代码](../../residency.py#L201)。因此，这个通用检查在
当前固定 decode 路径中，有改为绑定期校验的优化空间。需要通过专用路径保证
该约束，不能直接删除通用接口的保护。

## 2. 后六次：分别整理六个 component 的目标地址

W8A8 的完整专家 bundle 包含：

```text
w13_weight、w13_weight_scale、w13_weight_offset
w2_weight、 w2_weight_scale、 w2_weight_offset
```

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

见 [component 循环](../transfer.py#L489)。命中时搬运 hot cache → execution buffer
（D2D），未命中时搬运 Host master → execution buffer（H2D）；两者都需要准备地址。

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
  浮点权重。这个操作需要支持重复索引的累加语义，而不是简单连续写入，见
  [ScatterAdd 官方语义](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnScatterAdd.md)。
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
