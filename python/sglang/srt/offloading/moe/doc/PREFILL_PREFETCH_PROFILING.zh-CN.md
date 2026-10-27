# Prefill 下一组专家预取验证：NPU 执行 AI 指导

## 任务与背景

请在现有可用 NPU 环境验证：prefill 在计算第 i 组专家时，是否已提前搬运第 i+1 组专家权重。只做这项功能验证，不开展吞吐调优、LFU 对照实验或其他性能排查。

已知此前使用 DeepSeek V4 Flash ModelSlim W8A8、TP=EP=16、ATTN_DP_SIZE=2，MoE offload/cache 已能完成推理。此前环境还有 choices.py / storage.py 的本地兼容补丁。实际目录、设备映射、容器、模型路径、启动命令由你在机器上核实；保留能启动的配置和本地补丁。不要覆盖工作区、修改通信布局或关闭 offload。

本次新增的 Python profiling 标记不改变搬运顺序，不新增 synchronize，不需要重编译 ACL 扩展。代码必须在启动服务前加载；更新文件后要重启服务。

## 1. 确认代码与采集条件

调试代码位于 `https://github.com/wanerlee/sglang` 的
`codex/moe-prefill-prefetch-profile-20261027` 分支。在实际运行的源码目录中拉取：

```bash
git fetch https://github.com/wanerlee/sglang.git codex/moe-prefill-prefetch-profile-20261027
git switch -c codex/moe-prefill-prefetch-profile-20261027 FETCH_HEAD
```

若本地已有同名分支，切换并核对远程提交，不要强制重建。切换前记录并保存现有本地兼容补丁；如果 Git 因冲突拒绝切换，先处理补丁迁移，不要 reset/clean。此分支只提交了本次 profiling 标记、对应测试和这份指导；没有夹带本地尚未提交的其他优化。无需额外应用下面提到的独立 patch。

确认运行中的源码包含以下内容，而不是仅更新了另一个 checkout：

- `moe/runtime.py`：ManagedExecutionContext 分配进程内递增的 invocation_id。
- `moe/interfaces.py`：ExecutionContext 声明 invocation_id。
- `moe/prefill.py`：`moe.prefill.load_submit` 标记。
- `moe/programs.py`：`moe.prefill.compute_submit` 标记。

若收到单独的 `moe-prefill-profile-markers.patch`，在实际源码仓库先执行 `git apply --check <补丁路径>`，通过后应用。该补丁只包含上述四个生产文件。不要重置已有环境补丁；如有冲突，保留原业务实现，只迁移对应的标记和 invocation_id。

记录当前 commit、工作区 diff、实际启动命令、offload JSON、设备/rank 映射和 torch/torch_npu/CANN 版本。

继续使用已有 torch/torch_npu profiling 入口。采集必须包含 CPU/PyTorch 自定义范围、CANN API 和 NPU 设备任务。仅采设备任务会丢失本次标记。不要启用启动阶段 graph capture profiling 来替代本次采集；本次目标是预热后的 eager prefill。

## 2. 采集一个小的 prefill 窗口

1. 服务成功启动后，用 3 个短请求预热，等待它们全部完成。
2. 使用本机已经验证可用的 `/start_profile` 阶段采集方式：`profile_by_stage=true`，`num_steps=4`，输出到新的目录。核实该分支的参数和 profiler 实现；优先只选 prefill 阶段。若实现同时导出 decode，则单独保存并只分析 EXTEND/prefill 文件。
3. 发一个请求，客户端并发为 1。无需将服务的 max_running_requests 改成 1，也无需修改 decode graph 的 bucket。
4. 输入需要产生至少 4 个完整 prefill forward，并避免已有长前缀复用。例如 chunked_prefill_size=1024 时，可用约 6K 输入、16 个输出 token；检查实际 tokenizer 长度和 context_length。也可沿用现有 128K 请求，但不必为这项验证强制跑长请求。
5. 等待 prefill trace 完整导出。先分析一个实际处理该请求的 rank；其他 EP rank 继续正常参与运行，不能通过移除 rank 简化采集。

保留 server.log 和 trace_view.json，确认目标窗口为 4 个 prefill forward、没有 decode 或初始化/capture 混入。不要把多个 rank、layer 的日志条数当成 forward 步数。

## 3. 先确认标记可见

在原始 trace 或 MindStudio 中搜索以下前缀，标记位于 CPU/PyTorch 侧，不在设备算子 Summary 中：

```text
moe.prefill.load_submit
moe.prefill.compute_submit
```

示例：

```text
moe.prefill.load_submit forward=42 layer=10 kind=initial wave=0 microbatch=0 experts=[0,1) bank=0
moe.prefill.load_submit forward=42 layer=10 kind=initial wave=0 microbatch=1 experts=[1,2) bank=0
moe.prefill.load_submit forward=42 layer=10 kind=next current_wave=0 wave=1 experts=[2,10) bank=1
moe.prefill.compute_submit forward=42 layer=10 wave=0 experts=[0,2) bank=0
moe.prefill.compute_submit forward=42 layer=10 wave=1 experts=[2,10) bank=1
```

- 示例中的范围不是固定配置；以实际标记为准。
- `forward` 是进程内 invocation 编号，不是请求编号、token 步数或跨 rank 的全局编号。只在同一 rank/PID 内配对。
- `experts=[a,b)` 是本 rank 的层内专家范围，包含 a，不包含 b。
- `kind=initial` 的多次微批搬运共同组成最终 wave 0，不能把 microbatch 编号当成后续 wave 编号。
- `kind=next current_wave=i wave=j` 表示准备算 i 时提交 j 的权重搬运；此实现 j=i+1。
- `bank` 是缓冲区编号，复用时会重复。不能只靠 bank 配对。
- `compute_submit` 不包含 acquire 中下一组的搬运提交，但包含当前组就绪依赖、FFN 提交和 reader 完成事件提交；它不是纯 GMM 耗时。

如果原始 trace 中没有标记，先检查运行源码、CPU 采集是否开启，以及当前 torch_npu profiler 是否记录 record_function。不要将“未采到标记”判为“没有预取”。首批在后台 worker 中提交，某些 profiler 的线程采集行为可能不同；若仅 initial 标记缺失，可先用主线程的 kind=next 验证后续预取，并报告首批不可判定。

## 4. 将下一组搬运与当前组计算对应起来

选择同一 rank、同一 forward、同一 layer 中一对 `current_wave=i wave=i+1`，不要仅凭颜色或相邻位置配对。

1. 找到 `load_submit kind=next`。在这个 CPU 范围内找到 `aclrtMemcpyBatchAsync`；通过工具已有的关联线/connection id 等关联到设备 `MEMCPY_ASYNC`。一个专家组可有多个 batch、多块权重及 scale，必须收集该组的全部相关复制事件，不能只取第一个。
2. 找到同一层的 `compute_submit wave=i`。关联其内部提交的设备 FFN 算子，W8A8 的主要锚点是 W13 GroupedMatmul、ClippedSwiglu、DynamicQuant、W2 GroupedMatmul。
3. 找到 `compute_submit wave=i+1`，检查专家范围与 load_submit 一致，关联其设备 GMM。两个 GroupedMatmul 可能属于同一 wave 的 W13/W2，不要当成两组专家。
4. 使用设备事件时间判断实际搬运/计算关系。CPU 标记的结束仅说明提交返回，不能当成搬运完成或计算完成。
5. 若 CPU→设备关联缺失，报告“只能确认提交关系”，不要按时间邻近强行归属 memcpy。

对所选组记录：

| 字段 | 含义 |
|---|---|
| rank / PID / forward / layer | 配对身份 |
| current_wave / next_wave | 当前组与下一组 |
| local expert range / bank | 下一组的专家范围、目标缓冲区 |
| load CPU start/end | 下一组搬运提交范围 |
| compute i CPU start/end | 当前组计算提交范围 |
| next-copy device start/end | 下一组所有关联复制中最早开始、最晚结束 |
| current FFN device start/end | 当前组专家设备计算范围 |
| next GMM device start | 下一组首次读取该组权重的 GMM 开始 |
| overlap with current GMM | 复制与当前组真实 GMM 事件的相交时间，区间取并集后计量 |

## 5. 分开给出结论

分别报告以下三项，不能合并成一个“通过”：

1. **提前提交成立**：next load 的 CPU 范围在 current compute 的 CPU 范围之前，且它与下一组的专家范围、bank、forward、layer 一致。
2. **设备提前搬运成立**：关联的下一组复制在下一组 GMM 之前执行，并在首次读取权重前全部完成。若能找到就绪 event 的 record/wait，也保留该依赖；不要用提交时间代替设备时间。这验证时序，不单独证明所有权重字节内容正确。
3. **与当前组矩阵计算重叠成立**：下一组的设备 memcpy 与当前组的真实 GroupedMatmul 时间区间有交集。没有交集不等于没有预取，可能提前搬完，也可能有等待；应如实报告。

如果所有可见层都只有 compute wave=0、没有 kind=next，说明这次窗口没有覆盖“下一组”场景，不能判定失败。先核对 initial_submitted 是否等于本地专家总数。必要时增加一次独立的覆盖性采集：使用单独配置 cache_ratio=0，并在现有容量允许时令 execution_slots=2*prefill_group_size，且 group_size 小于本地专家数，使 A 只能容纳一部分专家。保留原配置结果；此附加实验只证明多 wave 流水可工作，不代替原 cache_ratio=0.05 下的结论。不要修改业务实现或模型来制造重叠。

## 6. 交付内容

- 一份简短报告：三项判断分别为已确认/未观察到/无法判定，以及上述时间表。尽量覆盖 3 个有后续 wave 的层；若可见样本不足，如实注明。
- 一张同一时间轴截图：CPU load/compute 标记、设备 GroupedMatmul、设备 memcpy，同时能看清层/wave 对应关系。
- 原始 trace 和配对事件信息（名称、时间、stream、可用的关联 id），以及采集配置与日志。

不要使用 CPU 提交范围的时长宣称 H2D 带宽、设备重叠率或预取加速比例；不要拿 AivKernel、Combine 或其他任意算子替代当前组 GMM 来证明“与专家矩阵计算重叠”。
