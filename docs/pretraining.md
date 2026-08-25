# π0.5 大规模 RLDS 预训练

本仓库的预训练入口用于在多个机器人 RLDS/TFDS 数据源上随机初始化、从 PaliGemma 初始化，或从已有 π0.5 checkpoint
继续训练 flow-matching VLA。它与 `scripts/train.py` 的微调配置和数据管线相互独立，不需要修改
`src/openpi/training/config.py`。

> 这里的“预训练”指多机器人、多任务的 action-conditioned VLA 训练框架。公开代码和公开数据无法复现
> Physical Intelligence 内部 10k+ 小时数据配方；当前实现也不额外加入语言模型/网页数据 objective。

## 配置与数据契约

从 [`configs/pretraining/pi05/template.yaml`](../configs/pretraining/pi05/template.yaml) 复制一份完整配置。建议按项目分目录：

```text
configs/pretraining/pi05/
├── template.yaml
├── embodiment_mix/
│   ├── baseline.yaml
│   └── temperature_ablation.yaml
└── continuation/
    └── stage2.yaml
```

每个 source 必须是 RLDS：外层 element 是 trajectory，trajectory 内所有 frame 字段的第 0 维都是 `T`。
内置 `field_map` adapter 将原始字段映射为以下 canonical 数据：

- `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`；缺失的腕部相机可设为 `null`。
- `[T, state_dim]` state 和 `[T, action_dim]` action。
- trajectory scalar 或 `[T]` 的语言 prompt。

真实 state/action 维度不足 `model.action_dim` 时会 padding。loss 使用显式 `[T, horizon, action_dim]`
mask，因此 padding 维度以及 trajectory 尾部不完整的 future action 都不参与 loss。

多个 source 在 sample/frame 层混合，实际概率为：

```text
p_i = weight_i^(1 / temperature) / sum_j(weight_j^(1 / temperature))
```

如果声明式字段拼接不足以表示某个数据集，应在 `openpi.training.rlds_adapters` 注册一个经过代码审查的 adapter，
然后只在 YAML 中引用注册名。YAML 不允许 Python import 或任意 callable。

AgiBotWorld Beta 的 tar 流式转换、断点续传和生产运行方式见
[`agibotworld_beta_conversion.md`](agibotworld_beta_conversion.md)。转换器生成的是 external folder dataset，loader 会在
`<data_dir>/<tfds_name>/<version>` 检测 `dataset_info.json` 并通过 `builder_from_directory` 加载，不需要注册 Python TFDS builder。
ABC-130K 的 MCAP 流式转换见 [`abc130k_conversion.md`](abc130k_conversion.md)，输出遵循同一个 external folder dataset
约定，并保留原始 train/val 为 TFDS train/validation split。

## 环境与归一化统计

RLDS 依赖单独放在可选 dependency group 中，并要求 Python 3.11：

```bash
uv venv --python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv sync --group rlds
```

训练前先计算 q01/q99、mean 和 std：

```bash
uv run --group rlds scripts/compute_pretrain_norm_stats.py \
  configs/pretraining/pi05/embodiment_mix/baseline.yaml \
  --max-frames-per-normalization 1000000 \
  --batch-size 1024
```

统计发生在 chunk/padding 之前。共享同一 `normalization_id` 的 source 必须有相同 state/action 维度；其采样预算按训练时的
温度概率分配。每份统计资产都有 source 配置 fingerprint，数据路径、字段映射、stride 或维度改变后，loader 会要求重新计算。

## 训练、验证和恢复

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run --group rlds scripts/pretrain.py \
  configs/pretraining/pi05/embodiment_mix/baseline.yaml
```

在新 GPU 架构或空 CUDA driver cache 上，第一次 NCCL collective 可能触发 PTX JIT，耗时几十秒。建议把缓存放到
持久且可写的目录，并在正式训练前运行数值校验脚本：

```bash
mkdir -p /mnt/pfs/rhos-vla/chenyuan/.cache/openpi-cuda
CUDA_CACHE_PATH=/mnt/pfs/rhos-vla/chenyuan/.cache/openpi-cuda \
CUDA_CACHE_MAXSIZE=4294967296 \
uv run scripts/check_gpu_collectives.py \
  --visible-devices 4,5,6,7 \
  --expected-device-count 4
```

脚本会连续执行两次小型 `AllReduce`、`AllGather` 和 `ReduceScatter` 并核对结果：第一次包含冷编译，第二次反映热缓存速度。
这些操作覆盖 FSDP 的关键通信路径；例如 AllReduce 正常并不能证明 AllGather 正常。训练入口也会在加载模型
和 checkpoint 前执行同样的轻量预热，由 `distributed.warmup_collectives` 控制。推荐保持 `true`；只有一个本地 device
时会自动跳过。独立诊断脚本默认设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`，不会为这个小探针预占训练规模的显存。
若冷启动期间 XLA 在 10 秒后输出 `rendezvous ... may be stuck`，先等待该数值校验完成；只有校验最终
报错或长期不返回时，才按实际 JAX/NCCL 通信故障排查。不要设置 `CUDA_CACHE_DISABLE=1`，并确保
`CUDA_CACHE_PATH` 可写。

本项目通过 `uv` 将 `nvidia-nccl-cu12` 固定为 2.28.9。不要降回 PyTorch 2.7.1 元数据中的 2.26.2：该版本在本机
sm_120 GPU 上可通过 AllReduce，却会在 AllGather 触发 `CUDA_ERROR_ILLEGAL_ADDRESS`。依赖安装应使用 `uv sync`，
保证 `uv.lock` 中的 NCCL override 生效。

初始化方式由 YAML 决定：

- `random`：使用 `training.seed` 随机初始化全部参数，不访问 GCS，适合从零预训练和离线 smoke test。
- `paligemma`：加载 PaliGemma 参数，其余 π0.5 参数随机初始化。
- `pi05_checkpoint`：加载完整的 π0.5 params，可做二阶段或继续预训练。

`runtime.compilation_cache` 控制 JAX 持久编译缓存。代码会在 distributed 初始化和任何 JAX 运算前应用这些配置，并分别记录
`performance/train_compile_seconds` 和训练 step 耗时。同一拓扑、模型 shape、JAX/jaxlib 版本及 XLA flags 不变时，后续运行应复用
缓存；`4×2`、`2×4`、`1×8` 的 executable 不同，首次切换拓扑仍会发生一次冷编译。首次从 checkpoint 恢复时也可能因
输入 state 的 committed sharding 与随机初始化路径不同而生成一个额外 cache key，之后相同恢复拓扑会复用该 key。真实多节点上
JAX 只让 global rank 0 写缓存，因此若所有节点都要在后续作业中命中，`runtime.compilation_cache.directory` 必须指向共享 PFS。
调试 miss 时可临时设置：

```bash
JAX_EXPLAIN_CACHE_MISSES=true JAX_LOG_COMPILES=true \
uv run --group rlds scripts/pretrain.py <config.yaml>
```

验证按 source 独立读取 `validation_split`，记录每个 source 的 loss、source macro loss 和按训练概率加权的 mixture loss。
checkpoint 精确保存和恢复模型参数、optimizer、EMA 与 step，同时保存完整 YAML manifest、CLI override、Git revision、
所有 normalization assets 和各 source 已消费样本数。`data_resume_mode: exact` 会在同一个 Orbax step 中为每个 rank 保存
tf.data iterator 与 source 健康状态，恢复时直接回到下一个 batch，不再按 step 回放；它要求相同 topology，并会关闭无法原子保存的
Python/device 预取。sidecar 缺失或 topology 改变时，由 `on_missing_iterator_state`/`on_topology_change` 决定拒绝恢复还是显式降级到
`statistical`。统计恢复会使用 step 派生的新 seed，不保证逐条复现数据顺序。

输入侧按 rank 记录 TFDS fetch、tokenize、host queue wait、device submit、抽样 host-to-device block、queue depth、坏 batch、
重试以及每个 source 的实际比例、饥饿 step、quota shortfall 和重复率。混合 iterator 的底层 TFDS 读取错误会指数退避重试；由于
错误发生在 sample 产出 source ID 之前，重试耗尽后会 fail-safe 终止，而不会错误地降级无关 source。已进入 adapter/tokenize
阶段的异常携带 source ID，可按 `source_failure_policy: degrade` 独立计数和降级，并且未完成 `min_samples` 的 source 不允许降级。

## 全周期日志、W&B 与告警

预训练、RLDS 转换和 normalization stats 使用同一套 lineage。转换 manifest 中的 `lineage_id` 会被统计任务和训练任务继承；
W&B 中分别显示为 `data_conversion`、`normalization` 和 `training` job。大体积 RLDS 与 checkpoint 始终保留在 BOS/PFS，
W&B Artifact 只保存配置、manifest、统计摘要以及对应 URI 和 SHA-256，不会重复上传模型或数据。

训练默认写入 W&B Cloud，同时将完整记录写入：

```text
<logging.local_root>/<project>/<experiment>/<run_id>/
├── run_manifest.json       # 配置、代码版本、host/process 信息
├── lineage.json            # dataset/normalization revision
├── metrics.jsonl           # loss、吞吐、数据源和 checkpoint 指标
├── events.jsonl            # 初始化、编译、训练、验证和退出状态
├── alerts.jsonl
├── logs/process-*.log
└── system/process-*.jsonl  # 每日或 512 MiB 轮转
```

`logging.local_root: null` 时目录位于 `<checkpoint_dir>/observability`。W&B 初始化或网络发送失败只会降级到 PFS 日志，
不会终止健康训练；`wandb_mode: offline` 可主动只落本地，之后运行 `wandb sync <run目录>/wandb`。
resume 会沿用 checkpoint 根目录中的 `wandb_id.txt`，因此 W&B run、PFS run 目录和 global step 都保持连续。

可选通用 Webhook 通过环境变量提供，URL 不会进入 YAML snapshot：

```bash
export OPENPI_TRAIN_ALERT_WEBHOOK_URL='https://example.internal/openpi-alert'
```

NaN/Inf、checkpoint 失败、未捕获异常、终止信号、PFS 空间不足，以及 training phase 600 秒没有 optimizer step
都会写入 PFS 并发送 W&B/Webhook 告警。JAX 首次编译、验证和 checkpoint 有独立 phase，不会误报 stall。致命异常会在当前
安全边界尝试 emergency checkpoint；超过配置的 grace period 后以非零状态退出。

高频系统日志可以交给定时任务归档；最近 5 分钟仍有写入的文件会自动跳过，`archive` 只压缩，`prune` 仅在 `.zst`
校验成功后删除超过保留期的原始 JSONL：

```bash
uv run scripts/manage_observability.py \
  --root /mnt/pfs/path/to/observability --action archive
uv run scripts/manage_observability.py \
  --root /mnt/pfs/path/to/observability --action prune --raw-retention-days 90
```

训练指标、事件、告警、manifest 和压缩后的系统日志不会由训练进程自动删除。

## 单机与原生 JAX 多机

单机多卡只需设置全局 `batch_size` 和 `distributed.fsdp_devices`。开发时可以在一台 8 卡机器上用两个 JAX rank
模拟两个节点；每个 rank 只看到自己的四张卡，但共同建立一个 8-device global mesh：

```bash
uv run --group rlds scripts/launch_pretrain.py local \
  configs/pretraining/pi05/<项目>/<实验>.yaml \
  --device-group 0,1,2,3 \
  --device-group 4,5,6,7
```

先加 `--probe-only` 可以只运行本地和跨 rank 的 AllReduce、AllGather、ReduceScatter 数值检查，不加载 RLDS 或模型。
launcher 自动选择 loopback coordinator 端口、为每个 rank 保存独立日志，并在任一 rank 失败时回收自己启动的其他 rank。
训练 rank 遇到未捕获异常时会先落盘 traceback 和 observer 状态，再尝试关闭日志与 JAX distributed runtime；超过
`runtime.fatal_cleanup_timeout_seconds` 后硬退出，避免一个已失败 rank 卡住其余进程。
训练参数仍可放在 `--` 后临时覆盖；`--distributed.*` 参数由 launcher 独占，不能通过 passthrough 重复设置：

```bash
uv run --group rlds scripts/launch_pretrain.py local <config.yaml> \
  --device-group 0,1,2,3 --device-group 4,5,6,7 -- \
  --batch-size 8 --num-train-steps 2
```

真实多机不由本脚本执行 SSH。Slurm 可由仓库内置提交器启动一节点一 JAX process；YAML 设置 `cluster.platform: slurm`，
checkpoint 必须位于所有节点可见的共享文件系统：

```bash
uv run --group rlds scripts/submit_pretrain_slurm.py <config.yaml> \
  --nodes 4 --gpus-per-node 8 --cpus-per-task 64 --partition gpu
```

提交器配置 `USR1` 提前通知、checkpoint grace period 和 whole-job requeue；任一 rank 失败时 `srun --kill-on-bad-exit`
终止整个 step，重排队次数受 `cluster.max_restarts` 限制。重排队后训练从最新 checkpoint 恢复；节点身份可以改变，设备总数改变时
exact 数据恢复按 `cluster.allow_topology_change` 与 `checkpoint.on_topology_change` 的策略处理。先加 `--dry-run` 可检查完整 `sbatch`
命令而不提交任务。

Kubernetes 或人工 SSH 仍可在每个节点分别运行 `rank`，所有节点使用完全相同的 YAML 和共享 checkpoint 路径，只改变运行时 rank：

```bash
# node 0
uv run --group rlds scripts/launch_pretrain.py rank <config.yaml> \
  --coordinator-address 10.0.0.1:12345 --coordinator-bind-address '[::]:12345' \
  --num-processes 2 --process-id 0 --local-device-ids 0,1,2,3,4,5,6,7

# node 1
uv run --group rlds scripts/launch_pretrain.py rank <config.yaml> \
  --coordinator-address 10.0.0.1:12345 \
  --num-processes 2 --process-id 1 --local-device-ids 0,1,2,3,4,5,6,7
```

上述 launcher 用法要求 YAML 中 `distributed.initialize: false`，coordinator/rank/device 字段保持 `null`；launcher 会在进入
训练入口前注入完整参数。也可以不使用 launcher，在每个 host 的 YAML/CLI 中显式设置：

```yaml
distributed:
  fsdp_devices: 8
  warmup_collectives: true
  initialize: true
  coordinator_address: host0.example:12345
  coordinator_bind_address: null
  num_processes: 4
  process_id: 0  # 每个 host 不同
  local_device_ids: [0, 1, 2, 3, 4, 5, 6, 7]
  cluster_detection_method: null
  initialization_timeout: 300
```

在 Slurm/MPI 等 JAX 可自动检测的环境，可以把 coordinator/进程/device 字段保持 `null` 并设置相应
`cluster_detection_method`。全局 batch size 必须能被全局 device 数整除。TFDS 会先按 process shard，之后各 host
共同构造全局 JAX array；Orbax checkpoint 和 W&B 写入仅由 process 0 负责。

多网卡真实节点可通过 `coordinator_bind_address` 控制 process 0 的监听接口，并按集群实际网络设置
`NCCL_SOCKET_IFNAME`、`NCCL_IB_HCA`。本机双 rank 测试只能验证 JAX 多进程、跨 rank NCCL 和训练语义，不能替代真实
IB/RoCE 链路验收。

`check_gpu_collectives.py --write-baseline <file.json>` 可在健康集群上记录各 payload 的 AllReduce/AllGather/ReduceScatter
带宽；后续用 `--baseline` 检查，或在 YAML 配置 `distributed.diagnostics.collective_baseline_path` 作为训练启动门禁。拓扑日志同时记录
GPU 拓扑、网卡状态/MTU/link speed/NUMA、RDMA device 和关键 NCCL 绑定。真实训练阶段的 JAX profiler trace 由
`profile_start_step/profile_num_steps` 控制，可用于在 TensorBoard/XProf 中区分计算、collective 与 overlap。

仓库提供可重复的完整模型 smoke test。它会在一个全新的 PFS 目录生成两个 mock RLDS source，依次执行跨 rank 通信探针、
完整 `gemma_2b + gemma_300m` step 1 保存和 resume 到 step 2，并核对数据消费计数、有限 loss/grad 和 rank 日志：

```bash
uv run --group rlds scripts/multinode_pretrain_smoke.py \
  --work-dir /mnt/pfs/path/to/new-smoke-directory \
  --wait-for-memory-seconds 3600
```

只验证 launcher、数据、FSDP 和 checkpoint 的多进程拓扑时可加 `--dummy-model`，它缩小语言与 action expert，保留真实视觉
输入和相同训练控制流；正式模型显存/主存验收不要使用该选项。

单机 8 卡也可以用 8 个单卡 rank 模拟 `8 节点 × 1 GPU`，为命令增加八组
`--device-group 0 --device-group 1 ... --device-group 7`。这种方式会执行真实的 8-rank JAX/NCCL 协调，但通信仍走本机
SHM/P2P，不能用于推断真实节点间 IB/RoCE 性能。

默认要求 cgroup 至少有 140 GiB 可用主存，并在使用率达到 95% 时只停止本次启动的 rank。内存保护默认使用
`memory.current - memory.stat:inactive_file` 得到 working set，不会把可回收 file cache 当成不可用内存；需要保守复现原始
口径时可给 launcher 传 `--cgroup-memory-accounting current`。smoke test 不会删除或复用已有目录，也不会终止机器上的其他进程。
