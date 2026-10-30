# MoE 分层执行实验版

基线为 `wanerlee/sglang:codex/moe-offload-dev-20261027` 的
`ce4b5c8970cdbcd9fa5d49649c5894a5cee707fa`，开发分支为
`codex/moe-layer-execution`。未合入 BS1 专用实验分支。

本版本增加独立的整层执行路径。原有 wave 路径仍是默认值，便于回归和对照。
CPU 测试覆盖调度、权重来源、缓存策略、阶段切换与失败处理；尚未验证真实 NPU
上的算子精度、graph capture/replay、性能和显存峰值。

## 开启方式

在**当前实际使用的 offload JSON** 中增加以下字段，保留原来的缓存参数、
offline profile、内存预算和并行配置：

```json
{
  "prefill_mode": "layer",
  "decode_mode": "layer",
  "layer_profile": true
}
```

上面只是需要增加的字段，不能作为完整配置文件。`prefill_mode` 和 `decode_mode`
分别可设为 `layer` 或 `wave`，可逐项 A/B。`layer_profile` 开启 CPU 提交范围标记；
计时基准可关闭，采集 trace 时开启。

仓库示例 `scripts/moe_offload_a3_layer.json` 与 `scripts/moe_offload_a3.json`
仅相差这三个字段。它保留示例的 50% Hot 容量、初始化 profile、周期 LFU 参数，
不代表你的线上配置。使用现有启动脚本时将 `MOE_OFFLOAD_CONFIG` 指向新 JSON；
使用 `launch_unified.sh` 时更换 `--moe-offload-config` 的路径即可。

例如在已验证可启动的 16 die 环境，沿用其环境变量和模型路径：

```bash
MOE_OFFLOAD_CONFIG="$PWD/scripts/moe_offload_a3_layer.json" \
WORLD_SIZE=16 ATTN_DP_SIZE=2 \
bash scripts/launch_moe_offload_npu.sh --dry-run
```

核对后去掉 `--dry-run`。仍需设置 `MODEL_PATH`，并沿用已经验证的
`MEM_FRACTION_STATIC`、chunk、context 等值。没有新增启动参数或修改原脚本默认值。
保留 `--disable-overlap-schedule`、`--disable-shared-experts-fusion`；
prefill graph 仍关闭，decode 仍走原有 graph 后端。不能据此认定 eager decode 已受支持。

## Prefill：整层双缓冲

下文 L 表示本 rank 的第 L 个 offloaded 层；非 offloaded 层不会占用序列位置。

```text
计算流： attention L0 → 等 A → FFN L0 → combine L0 → attention L1 → 等 B → FFN L1 → ...
搬运流： Host L0 → A → Host L1 → B → 等 L0 最后一次读 A → Host L2 → A → ...
```

- 第一个 offloaded 层入口提交 L0 和 L1 的完整本地专家权重。
- 每层 dispatch 一次，按 expert 分组的接收结果直接进入一次整层 FFN，之后 combine 一次。
  FFN 继续调用原有 Ascend runner，内部仍包含矩阵乘、激活等多个设备任务。
- L0 的最后一次权重读取提交后，搬运流等待该 reader event，再覆盖 A 加载 L2。
  B 同理。没有对未记录 event 的等待，也没有每层全局 compute → copy 屏障。
- 新路径不创建 wave、不做 wave 选择，也不建立 Python 搬运工作线程。
  ACL 描述符在初始化时准备，运行时在主线程提交有序异步搬运。
  保留 native shim 的 torch_npu host queue flush，不能擅自删去。
- Prefill 预取整层全部本地专家，因此不需要提前预测下一层路由。
  Copy stream 仍是串行的；加载赶不上计算时，计算仍然需要等权重。

## Decode：整层按需准备，保留缓存

```text
dispatch / 设备 counts
        ↓
一次整层缓存查询：只为 counts > 0 的专家准备权重
        ├─ 命中：Hot → Execution[expert_id]
        └─ 缺失：Host → Execution[expert_id]
        ↓
等待本层准备完成 → 整层 FFN → combine
        ↓
原有模型尾部的统计与周期 LFU 更新
```

Decode 也没有 wave 循环，但不做下一层全专家盲预取：下一层路由尚未产生，
为 decode 每次搬一整层会失去按需加载的意义。Execution 固定保留本层所有专家的槽位，
只写实际使用的槽位，保留设备上的 valid mask 和 generation 校验。

Hot 命中统计、全 rank 内跨层 LFU 选择、替换门槛、衰减、维护周期以及 graph 更新门控均复用原逻辑。
Prefill 沿用原有 Hot 借用机制：先快照，切回 decode 时恢复原权重、映射及策略状态。
恢复本身仍有搬运开销，本版本没有消除该阶段切换成本。

## 显存约束

设每个 rank 每层有 E 个本地专家，每个完整专家（权重及量化参数）占 B 字节：

- `decode_mode=layer` 要求 `execution_slots >= E`。
- `prefill_mode=layer` 的 A 使用 Execution 前 E 槽；Hot 至少 E 槽时，B 借用 Hot 前 E 槽。
  Hot 不足 E 槽时，必须显式配置 `execution_slots >= 2E`，由 Execution 后 E 槽作为 B。
- 不自动增加池、不降低 Hot 容量，也不静默退回 wave。容量不够会报错。
- 原 Execution 为 S 槽时，增至 E 槽的权重增量为 `max(0, E-S) * B` / rank。
  两个 bank 是已有池的视图，不能再将它们的大小额外加到池的占用上。
- 整层 FFN 的激活和原生算子 workspace 峰值可能增加。启动日志的权重字节数
  不能替代实测的 prefill 峰值和 decode graph pool 峰值。

示例中的 16 槽只适用于 E 不超过 16 的场景。改变 EP 后要按实际本地专家数重新核对。

## NPU 验证要读什么

1. 先在同一代码提交、模型、TP/EP/attention DP、实际 `toks`、缓存配置和采集窗口下，
   对比 `wave/wave`、`layer/wave`、`layer/layer`。保存启动日志和显存峰值。
2. 做连续两次 prefill → decode、空接收 rank、不同 decode batch、至少跨越两个 LFU
   更新周期的正确性检查。核对输出/logits 误差、graph replay 与 cache hit/eviction 计数。
   CPU 替身测试不代表量化 GMM 改变分组后一定达到同样的数值误差。
3. 使用既有 `/start_profile` 流程采集，新增 CPU 标记为：
   `moe.prefill.layer_load_submit`、`moe.prefill.layer_wait_ready`、
   `moe.prefill.layer_compute_submit`、`moe.prefill.bank_reuse_wait`；
   decode 对应 `moe.decode.layer_*`。字段包含 forward/layer/bank/experts，完整层中没有 wave。
   Decode graph replay 不会重新执行 Python 标记，要结合 capture 关联和设备轨道分析。
4. 将 load 标记内 ACL 调用关联到设备 MEMCPY，将 FFN 标记关联到实际设备 GMM。
   核对 L+1 的加载是否与 L 的计算重叠；L+2 是否晚于 L 的最后一次 bank 读取；
   当前层首次读取是否晚于本层最后一笔权重搬运。CPU 标记横向重叠不能证明设备并行。
5. 比较 step 墙钟时间、关键路径上的计算等权重、CPU 提交间隙；不要累加不同流的等待
   时长。本实现移除了旧预取工作线程交接，但不保证消除所有气泡或带宽瓶颈。

## 代码入口与本地测试

- `layer_execution.py`：整层请求、绑定和物理 bank 容量选择。
- `layer_prefetch.py`：跨层预取、bank reader 依赖、异常 drain。
- `layer_programs.py`：单次 dispatch/FFN/combine。
- `residency.py`：复用设备缓存查询，新增整层固定槽位计划。
- `npu/backend.py`、`npu/experts.py`：配置装配、预制搬运描述符、整层输入及输出。

这些入口位于 `python/sglang/srt/offloading/moe/`。CPU 测试：

```bash
PYTHONPATH=python:test/registered/unit/offloading \
python -m unittest test_layer_execution test_npu_backend test_native_periodic_cache
```

测试覆盖：完整权重 bundle、命中/缺失和代际故障、counts 变化后的设备计划重放、
bank 覆盖依赖、连续 forward、单次 dispatch/FFN/combine、padding/NaN、
prefill/decode 缓存恢复以及原有周期 LFU 规则。
