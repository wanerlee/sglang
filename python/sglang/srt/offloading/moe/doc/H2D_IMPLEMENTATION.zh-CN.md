# H2D 实现与资源契约（2026-10-25）

> 历史设计记录：本文中的 wave 调度、分组配置及相关接口已在分层分支删除。当前执行方式和配置迁移见[分层实现说明](../../../../../../scripts/moe_layer/README.md)。缓存策略设计仍供参考。

当前实现由公共 `ExpertWeightAccess` 直接组合 `HostTransferOps` /
`DeviceTransferOps`，并复用有界 prefill pipeline。重构与验证状态见 [重构记录](REFACTOR_20261025.zh-CN.md)。

## 存储和传输

`HostMemoryCapability` 分开表示 host 可读区间、ACL 异步读取能力、device 映射地址
及 registration 保活。ACL 与 MemFabric 共用一份不可变 host master；物理保留容量
（包括对齐）由 ledger 计费，不通过额外 master 或未登记 staging 绕过预算。

`native_acl.AclHostTransferOps` 验证完整 bundle 字节范围、设备、registration 与目标重叠，
按条目上限拆 batch，提交前保活描述符。完整提交后产生 ready；部分提交异常保留
资源并尝试记录 fence。native ACL 必须使用目标 CANN 编译并验证 torch_npu 队列顺序。
`HostCopyRegion` 的容量来自 source/destination 能力对象，不再重复存储可失配的容量字段。

`MemFabricDeviceTransferOps` 从固定设备计划降低动态 HOST/HOT_SLOT 地址、目标
和 mask；校验 generation、计数、索引与重复目标，全部分量共享 sticky error。
完整 bundle 完成后才发布。生产路径不提供 CPU MemFabric 回退。

## 地址目录与独立 scratch

`MemFabricAddressCatalog` 每 arena 建立一次不可变 host/hot 源地址表；destination
表与别名验证按实际物理 view 缓存。目录同时保留 allocation owner，避免地址复用
造成缓存误命中；capture 前 `seal()`，之后不能新增物理 view。

各 `MemFabricTransferBindings` 只传入共享目录和 destination bundle，不再重复传入
或验证整套源分量；共享目录中的表，但保留各自会被覆写的 lowering
scratch、mask 和 packed metadata。目录的 `allocations()` 集中报告地址表，binding
报告自身 scratch；ledger 拒绝重复登记的物理存储。共享目录不允许并发覆盖同一 scratch。
PyTorch 临时表达式和 graph allocator 增长仍需纳入 workspace 与实机峰值检查。

## Prefill 与显式退休

一个 arena 的所有层共用一个 `BoundedPrefillWeightPipeline`，session 以 invocation
与 layer 标识。P/Q 限制初始提交，dispatch completion 封口 K；首批写 A 的不同偏移，
随后 B/a 交替。下一 wave、下一层和下一 invocation 覆写 bank 前等待旧 reader。
worker 显式进入设备/copy stream；停止并 join 后才记录最终 copy fence 和封闭 context。

`retain` 与完成依赖登记分离；`register_completion` 保留未知 transport/auxiliary
reader 的独立完成语义。`wait_for` 只跳过由 context 记录、且已处于目标流顺序内
的 completion；未知来源的完成事件仍建立真实等待。
普通提交不为资源退休新增 host synchronize；已有 done/wait 证据后显式释放保活。

异常时先阻止新提交、join worker，再 drain 包括部分 native 提交的全部流；graph
资源还须成功 reset 后才能退休。排空/reset 失败继续保留资源，不能依赖 Python GC。

## 验证边界

保留的 CPU 测试包含完整 bundle、稀疏计划、读者保护和部分提交保活；测试替身不进入生产实现。
2026-10-25 本轮在既有 `test_npu_backend.py` 中加入真实 metadata 存储测试，
核对共享地址表的 data_ptr、独立 scratch、seal 与物理计账；ACL 用例验证容量、
别名拒绝、部分提交和事件记录失败后的保活。格式用例位于既有 `test_w8a8_format.py`。
本轮未取得真实 NPU 实测。目标机还需验证 ACL ABI/队列顺序、MemFabric 动态地址、
跨流 full graph、FP4/W8A8 数值、多 rank EP、内存峰值及 H2D/F 时间线。
非 due 使用 masked tail，不声称设备 kernel 已真正早退。
