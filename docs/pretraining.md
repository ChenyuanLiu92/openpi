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

初始化方式由 YAML 决定：

- `paligemma`：加载 PaliGemma 参数，其余 π0.5 参数随机初始化。
- `pi05_checkpoint`：加载完整的 π0.5 params，可做二阶段或继续预训练。

验证按 source 独立读取 `validation_split`，记录每个 source 的 loss、source macro loss 和按训练概率加权的 mixture loss。
checkpoint 精确保存和恢复模型参数、optimizer、EMA 与 step，同时保存完整 YAML manifest、CLI override、Git revision、
所有 normalization assets 和各 source 已消费样本数。RLDS shuffle/prefetch 流采用统计恢复：resume 后使用由 step 派生的新 seed，
不会逐条重放中断前的数据顺序。

## 单机与原生 JAX 多机

单机多卡只需设置全局 `batch_size` 和 `distributed.fsdp_devices`。多机应在每个 host 上使用同一共享 checkpoint 目录和配置，
并在首次访问 JAX device 前初始化集群。可以在 YAML 中显式设置：

```yaml
distributed:
  fsdp_devices: 8
  initialize: true
  coordinator_address: host0.example:12345
  num_processes: 4
  process_id: 0  # 每个 host 不同
  local_device_ids: null
  cluster_detection_method: null
  initialization_timeout: 300
```

在 Slurm/MPI 等 JAX 可自动检测的环境，可以把 coordinator/进程字段保持 `null` 并设置相应
`cluster_detection_method`。全局 batch size 必须能被全局 device 数整除。TFDS 会先按 process shard，之后各 host
共同构造全局 JAX array；Orbax checkpoint 和 W&B 写入仅由 process 0 负责。
