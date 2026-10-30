# 原生 NPU EP / FFN 接入

> 历史设计记录：本文中的 wave 调度、分组配置及相关接口已在分层分支删除。当前执行方式和配置迁移见[分层实现说明](../../../../../../../scripts/moe_layer/README.md)。缓存策略设计仍供参考。

`../experts.py` 将已有 `DispatchOnceProgram` / `PrefillDispatchOnceProgram`
连接到仓库内的 DeepEP dispatcher 与 `AscendRunnerCore`。模型的 shared expert、
routed scaling 仍由外层 DeepSeek MLP 执行；返回值包含原 `FusedMoE.forward_impl`
的 hidden-size 裁剪与 `reduce_results` 收尾。

## 已实现的执行链

1. Dispatch 前用 `context.num_token_non_padded` 设备标量将 padding 的 expert ID
   设为 `-1`、权重设为 `0`，保留其他 TopK 字段且不原地改写原 routing。
   这是必要的：当前 NPU V4 `fused_topk_npu` 并未实际应用传入的非 padding 数量。
2. 原生 dispatch 已在 `dispatch_b` 等待通信事件或 receive hook；在其后记录 compute
   事件，作为 prefill worker 的真实 dispatch completion。提前保留 native handle
   的值，避免 combine 清除 dispatcher 字段后过早释放通信元数据。
3. NPU DeepEP 的接收缓冲是二维、按专家紧排的张量。Decode 使用设备端 counts /
   prefix offsets 驱动固定容量 wave gather 与原位置回填。Eager NORMAL prefill 利用
   dispatcher 已返回的 Python-list counts 计算主机 offsets，直接切出本 wave 的连续
   activation / scale 行，FFN 只接收这些行，结果直接回填对应输出区间。全层输出仍保持
   combine 所需的形状与零 padding，不新增 `.item()` / `.tolist()` 设备回读。
   整个 local-demand 向量传给 residency，各 wave 只消费自身的专家区间。
4. Decode activation 与 MXFP8 scale 通过 byte views 搬运，保留量化位模式；prefill
   连续视图保留原 dtype 和位模式。固定 input、scale、
   output、counts、offsets、row-index allocation 显式计入预算。串行层可共享同一 layout，
   factory 只登记一次物理 allocation。
5. 每个 lease 的 `bindings.expert_bank` 包含 `AscendQuantInfo` 原生权重视图、
   `execution_valid` 与 canonical `error_status`。FFN 不重绑模型 Parameter，不转换
   FP4 精度。绑定时所有 quant 分量的专家维必须等于 validity 长度；runner 只接受
   相同宽度的 group counts，尾 wave 使用已构造好的精确视图。
6. 直接调用现有 `AscendRunnerCore`，保留 V4 `swiglu_limit` 所要求的非融合 GMM1 /
   clamp activation / GMM2 路径。任何有需求却无有效权重的 lane 都锁存错误，整组
   counts 归零；未写 padding 用 `where` 清零，避免 `NaN * 0`。
7. compute 与显式登记的附加 weight-reader stream 的事件组成 reader ticket。
   local program 在下一次 bank 写入前等待全部 readers；提交前即保留 bank / quant
   views / input，异常由 runtime drain 后释放。默认原生 core 不注入 overlap hooks，
   两个 GMM 与 activation 均使用当前 compute stream。Eager prefill 的零 token wave
   跳过 native FFN，仍返回正确 dtype 的空结果和 reader ticket，照常释放 lease；本地
   IDLE rank 是否计算由 dispatch 后的接收行数决定。
8. Combine 使用原 EP IDs / weights，wave 回填不重复应用 top-k 权重。错误状态还会
   清零此前成功 wave 的结果；masked token 的 combine 输出再次清零，避免读取未写 padding。

## 当前支持边界

- 原生构建入口要求 DeepEP dispatcher，默认 `DEEP_USE_MODE=default`。
- 必须设置 `MOE_ENABLE_TOPK_NEG_ONE=1`，使 native low-latency D/C 的 active mask
  忽略 `-1`；必须采用 `MOE_EXPERT_TOKEN_NUMS_TYPE=1` 的逐专家 count。
- Decode 必须提供设备非 padding 数量，接收 count 使用 low-latency 设备张量。
  Normal Python-list count 只允许 eager prefill。
- 拒绝两个 dispatcher 的 TBO；`alltoall` fallback 的 counts 包含 padding，`ops`
  strategy 的 active-mask / MXFP8 路径与当前绑定不兼容，因此不启用。
- Layout 的 AscendTP count 解析可独立测试，但生产 builder 暂不启用 AscendTP：其
  `-1` 路由与 finalize 端到端行为尚未在目标运行时验证。

## 目标机验收

CPU 契约测试覆盖字节打包、prefill 连续切片和回填、空组、padding、错误屏蔽及绑定宽度；
native/device 替身只存在
于测试目录，不能证明原生算子的数值或图行为。在目标机验证
全零 counts 的 grouped matmul 不读取无效 lane；byte gather / where / prefix-sum
的 torch_npu graph 支持；不同 padding 下 native `-1` D/C；MXFP8 scale / NZ
weight views；prefill 不同实际行数及非零 storage offset 的原生 GMM；多 rank 单层
参考一致性；整模型 graph replay、模式切换和故障 drain。
