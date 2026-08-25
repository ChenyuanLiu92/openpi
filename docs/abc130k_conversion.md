# ABC-130K MCAP → TFDS/RLDS 生产转换

转换入口是 [`scripts/stream_abc130k_to_rlds.py`](../scripts/stream_abc130k_to_rlds.py)：

```text
BOS episode.mcap 顺序复制（可并发多个 reader、支持 .partial 续传）
                       ↓
PFS 有界 spool（默认 512 GiB）
                       ↓
多进程 MCAP/Protobuf 解析 + H.264/H.265 解码 + 30 Hz 因果对齐
                       ↓
sharded TFDS/RLDS + SHA-256 + 回读验证
                       ↓
BOS 隐藏目录上传 + 回读验证 + READY + 原子发布
```

原始 MCAP 始终只读。每个 shard 成功落盘并记录校验值后才删除对应 PFS spool。SQLite 状态位于
`<pfs-work-root>/state/pipeline.sqlite3`，中断后用完全相同的命令恢复。

每个 PFS spool MCAP 都带有源文件 `size + mtime_ns` sidecar。恢复时只有 sidecar 与当前 BOS 对象一致，且 MCAP
首尾 magic/summary 可读，才会复用已有副本；旧版无 sidecar 的 spool 会自动重新复制。这可以防止 BOS 对象保持最终
文件大小、但内容仍在后台更新时产生同尺寸的截断副本。

## Smoke test

先用一个真实 episode 验证完整链路，只写 PFS：

```bash
uv run --group rlds scripts/stream_abc130k_to_rlds.py run \
  --episode-path /mnt/bos/dataset/abc-130k/data/train/fold_and_stack_the_t_shirts/episode_001005fe-c6ed-4e3c-b6ce-6beb4e8ce0cf \
  --pfs-work-root /mnt/pfs/rhos-vla/chenyuan/abc-130k-rlds-smoke \
  --dataset-name abc_130k_smoke \
  --episodes-per-shard 1 \
  --stream-readers 1 \
  --convert-workers 1 \
  --no-publish \
  --no-wandb
```

## 全量转换

```bash
uv run --group rlds scripts/stream_abc130k_to_rlds.py run \
  --input-root /mnt/bos/dataset/abc-130k \
  --pfs-work-root /mnt/pfs/rhos-vla/chenyuan/abc-130k-rlds-work \
  --output-root /mnt/bos/dataset/RLDS \
  --dataset-name abc_130k \
  --version 1.0.0 \
  --discovery-workers 32 \
  --stream-readers 4 \
  --spool-limit-gib 512 \
  --episodes-per-shard 8
```

`--convert-workers 0` 自动使用物理 CPU 核数；每个 worker 的 FFmpeg、OpenCV、BLAS 和 TensorFlow 内部线程均受限，避免
多进程过度订阅。终端直接显示 BOS→PFS 字节、转换 shard 和 PFS→BOS 发布进度。默认同时写入 W&B project
`openpi-data` 和 `<pfs-work-root>/observability`；可用 `--no-wandb` 只保留 PFS 日志。

发现阶段默认使用 32 个线程并行扫描 BOS 任务目录，每个 MCAP 只执行一次 `stat`，终端会显示已发现的 episode 数和
已完成的任务目录数。`--discovery-workers` 只影响扫描速度，不改变数据集内容或断点续跑身份。

生产模式默认遇到坏 episode 就停止，避免静默缺数据。只有明确接受隔离坏样本时才使用 `--skip-bad-episodes`；该选项
同时覆盖源文件复制/MCAP 完整性校验和解码阶段，并把错误写入 SQLite/manifest。已有 strict 工作目录允许原地升级为
skip-bad 策略，且会复用已经校验成功的 shard。可按
`--split`、`--task-name`、`--episode-id`、`--episode-path` 或 `--max-episodes` 选择数据；不同选择必须使用不同的
`--pfs-work-root`。

查看状态或验证输出：

```bash
uv run --group rlds scripts/stream_abc130k_to_rlds.py status \
  --pfs-work-root /mnt/pfs/rhos-vla/chenyuan/abc-130k-rlds-work

uv run --group rlds scripts/stream_abc130k_to_rlds.py validate \
  --location bos --dataset-name abc_130k --version 1.0.0
```

## 输出契约

- `observation/base_0_rgb`：单目 top camera，或按 episode ID 稳定选择 stereo top 的一只眼
- `observation/left_wrist_0_rgb`、`right_wrist_0_rgb`：左右腕部相机
- `observation/state`：左臂 6 + 左夹爪 1 + 右臂 6 + 右夹爪 1，共 14 维
- `action`：同顺序的 14 维控制量
- `language_instruction`：episode task/instruction
- `subtask_instruction`：逐帧因果对齐 annotation；未标注时回退到主指令
- train MCAP 写入 TFDS `train`，val MCAP 写入 TFDS `validation`

相机、状态、action 都使用“时间点之前最近一条消息”的因果 floor 对齐，默认固定 30 Hz；图片为保持宽高比并补黑边的
224×224 JPEG。最终 `conversion_manifest.json` 记录 split 数量、frame 数、异常、每个 shard 的 SHA-256 和训练 lineage。

预训练 YAML 中对应 source 的关键配置为：

```yaml
- id: abc_130k
  tfds_name: abc_130k
  version: 1.0.0
  data_dir: /mnt/bos/dataset/RLDS
  train_split: train
  validation_split: validation
  weight: 1.0
  normalization_id: abc_130k_v1
  action_stride: 1
  state_dim: 14
  action_dim: 14
  adapter:
    type: field_map
    options:
      images:
        base_0_rgb: observation/base_0_rgb
        left_wrist_0_rgb: observation/left_wrist_0_rgb
        right_wrist_0_rgb: observation/right_wrist_0_rgb
      state: [observation/state]
      actions: [action]
      prompt: language_instruction
      image_range: uint8
```
