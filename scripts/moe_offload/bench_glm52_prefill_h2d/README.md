# GLM-5.2 W8A8 prefill：独立权重 H2D 带宽测试

不启动模型、不读取 safetensors、不创建 HCCL 进程组。只读 `config.json`，生成合成字节数据，复用当前生产分支的实际 Host 分配、权重格式和 copy 接口。

生产代码核对版本：`34f063d8885b4ecc1fd93e5329a5440bc645a593`。运行时会记录实际导入文件的路径和 SHA256；脚本不修改 SGLang 仓库。这里没有复制一套近似实现，**需要服务器上已经具备该 prefill offload 实现的 SGLang 源码及依赖**。

## 测的是什么

调用链与当前 prefill 一致：

```text
MemFabricHostPool：LOCAL Host 池 + ACL HostRegisterV2(PINNED)
NpuV4W8A8Format：探测真实 NZ 物理布局、分配两个整层 bank
_projection_copy_plans：W13/W2 各自的整层计划
AclHostTransferOps.materialize_into
_sglang_moe_acl.submit
torch_npu OpCommand::RunOpApi("SglangMoePrefillH2D")
aclrtMemcpyBatchAsync
```

这是 Host 提交的 ACL batch copy；不是 `Tensor.copy_`，也不是 decode 的 `group_pack_copy` / AIV MTE kernel。脚本使用生产代码的描述符合并、任务队列和 W13/W2 ready event。

- 每个设备一个进程；`--devices` 是当前可见设备的**逻辑编号**。
- `--ep-size 4` 决定每 rank 搬 256/4=64 个完整专家，不会再按 TP4 除一次权重。
- 默认选 `--pp-rank 3`，即分区 `18,20,24,16` 的最后一个 stage。想测 20 层 stage，使用 `--pp-rank 1`。
- 默认每个 routed layer 都有独立 Host 源数据；两块目标 bank 交替复用。
- 每层先 W13、再 W2，同一个 FIFO copy stream。没有计算消费者，因此不加入真实模型里的 combine 等待、bank reader 等待或 attention 延迟。
- 合成数据是字节校验模式，不保证代表有效的量化数值；不用于 FFN 精度验证。
- 格式固定模拟当前 W8A8 实现；config 只提供模型几何，不会自动切换成其中可能声明的其他量化格式。正式对照请提供实际 W8A8 模型的 config。
- 初始化、格式转换探测、Host 填充、描述符准备和字节验证均在计时外。

**它给出当前内存放置及 copy 路径下的纯搬运吞吐，不能直接给出 H2D 掩盖率或 TTFT。** 多设备同开始的固定工作量测试也不等于所有设备无限持续负载下的硬件峰值。

## 容量口径

若实际 config 为 H=6144、I=2048、256 experts、78 layers、前3层 dense：

| 数据 | 每 rank / EP4 |
|---|---:|
| 单专家三矩阵 INT8 权重 | 36 MiB |
| 单专家权重 + scale + offset 搬运量 | 36.05859375 MiB |
| 单层 64 专家搬运量 | 2.253662109375 GiB |
| 单层 Host 存储跨度 | 2.2548828125 GiB |
| 两个整层目标 bank | 4.50732421875 GiB HBM |

| PP rank | routed 层数 | 每遍 H2D GiB/rank | Host 源数据 GiB/rank |
|---|---:|---:|---:|
| 0 | 15 | 33.8049 | 33.8232 |
| 1 | 20 | 45.0732 | 45.0977 |
| 2 | 24 | 54.0879 | 54.1172 |
| 3 | 16 | 36.0586 | 36.0781 |

默认 MemFabric **物理预留为 56 GiB/进程**，与当前启动脚本一致。4 个进程合计预留 224 GiB Host，HBM 双 bank 合计约18.03 GiB，此外有格式探测、框架与分配器开销。源数据跨度和物理预留不是同一个数字。

scale 在 Host 按 FP32 大小留跨度、实际仅搬 BF16 字节，因此不能将整个 Host slab 一次性连续复制。典型 EP4 每 projection 的192个原始 regions，经生产合并后为66个 descriptors：1个 weight、64个 scale、1个 offset。实际物理 extent 以服务器格式探测结果为准。

## 环境

使用能运行当前 prefill offload 的原容器：torch、torch_npu、匹配的 CANN、MemFabric 和本分支 SGLang 都应已安装。需要同一分支的 native ACL 扩展。脚本不运行模型，但会导入这些库。

把本目录完整复制到服务器。进入目录后按实际位置设置：

```bash
export SGLANG_ROOT=/home/y00951466/pd_test/sglang
export MODEL_CONFIG=/home/weights/GLM-5.2-w8a8/config.json
export RUNTIME_DIR=/tmp/sglang-glm52-prefill-pp4-cp4-runtime
export PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

# 如果之前的启动脚本生成过隔离的 MemFabric 运行目录，使用同一份：
if [ -d "$RUNTIME_DIR/python/memfabric_hybrid" ]; then
  export PYTHONPATH="$RUNTIME_DIR/python:$PYTHONPATH"
  export LD_LIBRARY_PATH="$RUNTIME_DIR/python/memfabric_hybrid/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

unset ASCEND_LAUNCH_BLOCKING SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT
export TASK_QUEUE_ENABLE=1
```

先只检查容量，不加载 NPU：

```bash
python3 -B bench.py --model-config "$MODEL_CONFIG" \
  --sglang-root "$SGLANG_ROOT" --pp-rank 3 --devices 0,1,2,3 --dry-run
```

已有对应 native 扩展时无需重新编译。若没有，或 CANN / torch_npu / native 源码发生了变化，显式构建：

```bash
MAX_JOBS=4 python3 -B bench.py --model-config "$MODEL_CONFIG" \
  --sglang-root "$SGLANG_ROOT" \
  --native-extension-dir "$RUNTIME_DIR/native_acl" --build-only
```

SDK 在非标准位置时加 `--cann-home /usr/local/Ascend/cann-9.1.0`。这只编译生产 ACL shim，不编译模型。没有扩展或 ABI 不兼容会直接报错，不会悄悄改用其他拷贝接口。

## 建议先跑小内存检查

```bash
python3 -u -B bench.py --model-config "$MODEL_CONFIG" \
  --sglang-root "$SGLANG_ROOT" \
  --native-extension-dir "$RUNTIME_DIR/native_acl" \
  --pp-rank 3 --ep-size 4 --devices 0 \
  --source-layers 1 --host-pool-gib 3 \
  --scenarios all --warmup 1 --iterations 2 \
  --output ./results/smoke
```

仍然是每层64专家、完整16层遍历，**但反复读取同一层 Host 源**。它用来确认接口/布局/校验能工作，不能当完整 stage Host 工作集的带宽结果。报告会标记 `source_reused=true`。

## 正式测一个 PP stage 的 EP4

默认每层独立 Host 源，2轮预热、5轮测量，每轮一遍完整 stage。依次测4个设备独跑和4设备同时搬运：

```bash
TASK_QUEUE_ENABLE=1 python3 -u -B bench.py --model-config "$MODEL_CONFIG" \
  --sglang-root "$SGLANG_ROOT" \
  --native-extension-dir "$RUNTIME_DIR/native_acl" \
  --pp-rank 3 --ep-size 4 --devices 0,1,2,3 \
  --output ./results/pp3_ep4_queue1
```

只测并发可加 `--scenarios all`。测试20层 stage 改为 `--pp-rank 1`；不要仅因为旧 trace 有20层就假定实际启动分区还是18,20,24,16，应按当次分区设置 `--pp-partition`。

对照任务队列2，除环境变量和输出目录外保持完全一致：

```bash
TASK_QUEUE_ENABLE=2 python3 -u -B bench.py --model-config "$MODEL_CONFIG" \
  --sglang-root "$SGLANG_ROOT" \
  --native-extension-dir "$RUNTIME_DIR/native_acl" \
  --pp-rank 3 --ep-size 4 --devices 0,1,2,3 \
  --output ./results/pp3_ep4_queue2
```

每个测试必须独立运行，不与模型或另一份带宽测试同时占用这些设备。脚本不会终止其他进程，也不会自动选择设备。

若设备0–3不是目标空闲设备，可按实际可见逻辑编号修改 `--devices`。只填一个设备时，仍按 EP4 的64专家分片大小测，不会自动变成256专家。

## 结果

- `summary.csv`：每个场景一行。重点看 `all` 的 `median_total_gbps`、`median_window_ms`、`window_cv_pct`、`max_start_skew_ms`。
- `ranks.csv`：每设备自己的 Host 窗口、event 时长、Host 提交时间和提交后等待时间。
- `samples.csv`：逐设备逐轮原始时间戳和真实字节数。
- `iterations.csv`：每轮共同窗口和合计带宽。
- `devices.json`：实际格式、Host/HBM容量、版本、copy源码指纹、任务队列环境、CPU亲和性和可读到的Host内存 NUMA 信息。
- `results.json`：完整结果，只有所有进程校验和清理成功才写 `complete=true`。
- `events.jsonl`：初始化、原始样本、校验和失败信息；失败另存 `failure.json`，不会打印成功结果。

**总带宽 = 各活跃设备实际搬运字节数之和 ÷（最晚 Host 完成 − 最早 Host 开始）。** 同一机器进程使用共同的 monotonic clock。GB/s 是十进制，GiB 是二进制容量。不能直接把各设备自己的速率相加当总吞吐。`median_event_estimate_gbps` 仅是总字节数除最长 event 时长的辅助估算，并非跨设备对齐的窗口。

Host 提交和设备执行可以重叠；`median_submit_ms` 不是“纯 CPU 额外损失”。event 区间也可能包含设备等 Host 提交的空白，因此这里是实际提交方式的有效带宽，不是隔离出的 DMA engine 峰值。

加 `--projection-timing` 可额外生成 `projections.csv`，逐层列 W13/W2 的 event 耗时和带宽。该选项添加额外事件，会改变测量开销；应与默认测试分开跑、分开比较。默认只增加每轮开始/结束两个 timing event，同时保留生产接口本身的就绪事件。

默认 `--verify sample`：计时外逐层复制，检查**每个专家每个 component 的首/中/尾**原始物理字节；全部场景后再验证最终两个 bank。它不是完整缓冲区校验。需要完整字节检查可使用 `--verify full`，会增加大量计时外 D2H 和 CPU 验证时间。`none` 明确关闭验证，不建议首次使用。

## 与 prefill profile 如何比较

1. 先看单设备和EP4并发的有效带宽，确认真实路径能达到多少。
2. 用 profile 中一层实际 H2D 字节除以这次带宽，估算纯搬运时间尺度；不是必然能达到的模型内耗时。
3. 模型同时存在 GMM 的 HBM 访问、CP/EP/PP 通信、reader依赖和Host调度，本测试均未包含。模型内更慢不一定是 copy 接口本身异常。
4. chunk 4096→16384 **不改变一次全层权重搬运的字节数**，只改变整请求需要遍历 stage 的次数以及可用于掩盖H2D的计算窗口。`--passes` 表示连续完整stage遍历次数，不模拟chunk的计算。
5. 默认不绑定NUMA，不自动设置服务器的CPU亲和性。请保留 `devices.json`；若多卡差异明显，结合实际源内存落点再做绑定对照。

## 本地检查与验证范围

```bash
python3 -B -m unittest discover -s . -p 'test_*.py' -v
```

CPU检查覆盖容量、分区、共同窗口统计以及模拟异步流中的copy顺序/生命周期/验证边界。Mac上没有NPU：尚未做真实CANN编译、NPU搬运、驱动兼容性或带宽验证。进程超时仅负责终止本次测试的子进程，不能保证解除驱动内部卡死。

## `allow_internal_format` 属性错误

若旧版脚本在生成环境信息时出现 `_npuConfig object has no attribute allow_internal_format`，更新 `bench.py` 即可，无需重编 ACL 扩展。

torch_npu 2.10 的 `_npuConfig` 只实现该属性的 setter：赋值会向运行时设置选项，但不会保存可读的 Python 属性。[上游实现](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/npu/npu_config.py)

脚本仍执行 `torch.npu.config.allow_internal_format = True`，但记录的是 `allow_internal_format_requested: true`，不再读回该属性。真实 NZ 布局仍由 `NpuV4W8A8Format` 探测和检查。CPU初始化回归使用同样的可写不可读配置对象，覆盖从设置选项到生成环境信息的完整初始化路径。

重新运行时换一个 `--output` 目录；脚本会保留已有失败报告，不覆盖旧目录。
