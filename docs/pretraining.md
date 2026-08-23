# π0.5 大规模 RLDS 预训练

本仓库的预训练入口用于在多个机器人 RLDS/TFDS 数据源上从 PaliGemma 初始化或已有 π0.5 checkpoint
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

- `paligemma`：加载 PaliGemma 参数，其余 π0.5 参数随机初始化。
- `pi05_checkpoint`：加载完整的 π0.5 params，可做二阶段或继续预训练。

验证按 source 独立读取 `validation_split`，记录每个 source 的 loss、source macro loss 和按训练概率加权的 mixture loss。
checkpoint 精确保存和恢复模型参数、optimizer、EMA 与 step，同时保存完整 YAML manifest、CLI override、Git revision、
所有 normalization assets 和各 source 已消费样本数。RLDS shuffle/prefetch 流采用统计恢复：resume 后使用由 step 派生的新 seed，
不会逐条重放中断前的数据顺序。

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
训练参数仍可放在 `--` 后临时覆盖；`--distributed.*` 参数由 launcher 独占，不能通过 passthrough 重复设置：

```bash
uv run --group rlds scripts/launch_pretrain.py local <config.yaml> \
  --device-group 0,1,2,3 --device-group 4,5,6,7 -- \
  --batch-size 8 --num-train-steps 2
```

真实多机不由本脚本执行 SSH。Slurm、Kubernetes 或人工 SSH 在每个节点分别运行 `rank`，所有节点使用完全相同的 YAML
和共享 checkpoint 路径，只改变运行时 rank：

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

仓库提供可重复的完整模型 smoke test。它会在一个全新的 PFS 目录生成两个 mock RLDS source，依次执行跨 rank 通信探针、
完整 `gemma_2b + gemma_300m` step 1 保存和 resume 到 step 2，并核对数据消费计数、有限 loss/grad 和 rank 日志：

```bash
uv run --group rlds scripts/multinode_pretrain_smoke.py \
  --work-dir /mnt/pfs/path/to/new-smoke-directory \
  --wait-for-memory-seconds 3600
```

默认要求 cgroup 至少有 140 GiB 可用主存，并在使用率达到 95% 时只停止本次启动的 rank。smoke test 不会删除或复用已有目录，
也不会终止机器上的其他进程。
