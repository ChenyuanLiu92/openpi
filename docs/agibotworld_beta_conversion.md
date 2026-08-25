# AgiBotWorld Beta → TFDS/RLDS 生产转换

大规模转换优先使用
[`scripts/stream_agibotworld_beta_to_rlds.py`](../scripts/stream_agibotworld_beta_to_rlds.py)。它针对 BOS 的顺序吞吐和
PFS 的高并发本地访问设计：

```text
BOS observation tar 顺序读取（16 MiB buffer，1–2 readers）
             ↓ 只保留三路 RGB MP4
PFS 有界 spool（默认上限 1 TiB） + 本地 proprio/task_info
             ↓ 多进程 AV1 解码和 TFRecord fragment
fragment 落盘 + SHA-256 + 状态库提交
             ↓ 才删除对应 PFS MP4 spool
PFS 标准 sharded TFDS/RLDS + 回读验证
             ↓ TFRecord 优先、metadata 最后
BOS 隐藏上传目录 + 回读验证 + READY + 原子发布
```

BOS 原始 tar 始终只读且永不删除。状态库位于
`<pfs-work-root>/state/pipeline.sqlite3`；tar 扫描从上次完整 member 边界恢复，文件复制从 `.partial` 长度恢复。
同一 task 的 tar 保持顺序，较早归档中成功转换的 episode 优先；失败时后续重复归档可以接替。任何未恢复的坏 archive、
缺相机 episode、缺 proprio/annotation episode 默认都会阻止正式发布。

## 流式 smoke test

先验证一个真实 episode，只写 PFS、不发布到 BOS：

```bash
uv run --group rlds scripts/stream_agibotworld_beta_to_rlds.py run \
  --observation-archive /mnt/bos/dataset/agibotworld-beta/observations/389/653277-674627.tar \
  --task-id 389 \
  --episode-id 655660 \
  --pfs-work-root /mnt/pfs/rhos-vla/chenyuan/agibotworld-beta-rlds-stream-smoke \
  --dataset-name agibotworld_beta_stream_smoke \
  --decoder cpu \
  --convert-workers 1 \
  --no-publish
```

指定 episode 的 smoke test 在三路视频全部取得后提前结束当前 tar；正式全量转换仍逐字节顺序读完每个 tar。查看持久状态：

```bash
uv run --group rlds scripts/stream_agibotworld_beta_to_rlds.py status \
  --pfs-work-root /mnt/pfs/rhos-vla/chenyuan/agibotworld-beta-rlds-stream-smoke \
  --dataset-name agibotworld_beta_stream_smoke
```

## 流式全量转换

当前服务器在停止旧随机索引任务后，以 16 MiB block、每路 1 GiB、`O_DIRECT` 实测顺序读取：1/2/4 readers 的聚合吞吐
分别为 582/934/1106 MiB/s。4 路比 2 路高约 18%，所以生产任务使用 4 个 reader。`convert-workers=0` 自动使用
90 个物理 CPU 核；归档扫描和转码并行运行，
但同一 task 内保持确定顺序：

```bash
uv run --group rlds scripts/stream_agibotworld_beta_to_rlds.py run \
  --input-root /mnt/bos/dataset/agibotworld-beta \
  --pfs-work-root /mnt/pfs/rhos-vla/chenyuan/agibotworld-beta-rlds-work \
  --output-root /mnt/bos/dataset/RLDS \
  --dataset-name agibotworld_beta \
  --version 1.0.0 \
  --stream-readers 4 \
  --spool-limit-gib 1024 \
  --episodes-per-shard 8
```

该命令默认将转换进度写入 W&B project `openpi-data`，并在
`<pfs-work-root>/observability` 永久保存结构化指标、系统指标和 lineage。可用 `--wandb-project`、`--wandb-entity`
调整归属，用 `--wandb-mode offline` 延后同步，或用 `--no-wandb` 仅保留 PFS 日志。转换 manifest 会携带稳定
`lineage_id`；后续 normalization 和训练会自动引用它。若需要跨多个独立数据转换显式指定业务 lineage，可传
`--lineage-id <ID>`。

运行期间同一个 terminal 会显示 `tqdm` 的 BOS→PFS 字节进度、归档转换进度和 PFS→BOS 发布进度；不必另开
`tail -f`。中断后执行完全相同的命令即可恢复。`--pfs-work-root` 与数据选择和转换 schema 绑定，切换 episode/task
过滤条件时必须使用新的 work root。完整 PFS 输出默认保留，即使 BOS 已完成并验证。

若只需要检查输出，不开始转换：

```bash
uv run --group rlds scripts/stream_agibotworld_beta_to_rlds.py validate \
  --location bos --expected-episodes <N>
```

`--allow-incomplete` 是显式的数据质量豁免，会允许存在隔离错误时生成/发布不完整数据集；生产运行不应默认使用。

## 直接 seek 转换器（小规模/兼容路径）

[`scripts/convert_agibotworld_beta_to_rlds.py`](../scripts/convert_agibotworld_beta_to_rlds.py)
把 Beta WebDataset tar 直接转换为 OpenPI 预训练可读的 sharded TFDS/RLDS。转换过程不会展开整个 tar：

```text
扫描 tar header 并持久化 member offset
                 ↓
按 episode 直接 seek 三路 RGB MP4 + proprio_stats.h5
                 ↓
并行 AV1 解码、时间对齐、224×224 JPEG 编码
                 ↓
每个 worker 原子写入独立 TFRecord shard
                 ↓
生成 TFDS metadata、回读验证、完成清单
```

索引位于 `<output-root>/.agibotworld_beta_conversion/agibotworld_beta_index.sqlite3`，按 tar 的绝对路径、
文件大小和 mtime 复用。每个 shard 有独立 sidecar；进程中断后运行同一命令会跳过已完成且 fingerprint 匹配的 shard。
如果转换参数或 episode 集合发生变化，工具会拒绝混写，需改用新的 `dataset-name` 或版本。

### 受控试跑

先用一个真实 episode 验证安装和输出：

```bash
uv run --group rlds scripts/convert_agibotworld_beta_to_rlds.py \
  --task-id 389 \
  --episode-id 655660 \
  --dataset-name agibotworld_beta_pilot
```

再转换一个完整 observation tar：

```bash
uv run --group rlds scripts/convert_agibotworld_beta_to_rlds.py \
  --observation-archive /mnt/bos/dataset/agibotworld-beta/observations/389/653277-674627.tar \
  --dataset-name agibotworld_beta_task389_pilot
```

仅建立索引或只生成计划、不解码视频：

```bash
uv run --group rlds scripts/convert_agibotworld_beta_to_rlds.py --index-only
uv run --group rlds scripts/convert_agibotworld_beta_to_rlds.py --plan-only
```

### 全量转换与性能

默认命令选择所有 observation tar：

```bash
uv run --group rlds scripts/convert_agibotworld_beta_to_rlds.py
```

`--workers 0` 使用物理 CPU 核数，避免把超线程误当成独立 AV1 解码核；`--index-workers 0` 会限制并发 tar
扫描，避免索引阶段冲击 BOS。`--decoder auto` 在第一个真实视频上比较 `libdav1d` 与 `av1_cuvid`，选择本机实测更快的后端。
可用 `--decoder cpu` 或 `--decoder nvidia --gpu-workers-per-device 4` 强制指定。每个 worker 的 OpenCV、BLAS 和
TensorFlow 线程被限制为 1，防止多进程内部再次扩线程导致过度订阅。

当前服务器为 90 个物理 CPU 核（180 线程）、1.8 TiB 内存和 8 张约 96 GiB GPU。2026-08-20 的 16-way
真实试跑转换了 16 个 episode、17,777 个 trajectory step（53,331 张图），shard 阶段约 32 秒，输出 940 MB；
聚合吞吐约为 548 step/s 或 1,645 image/s。`av1_cuvid` 可用，但单流只比单线程 `libdav1d` 快约 3%，计入可并行的
CPU/GPU worker 总数后 CPU 整机吞吐更高，因此 `auto` 选择 CPU。GPU 驱动或后续数据特征变化时会重新以首个真实视频
benchmark；也可以显式强制后端。

默认写入与 OpenPI 入模一致的 224×224 letterbox JPEG（质量 90），避免先保存 480×640、训练时再缩放造成数倍空间和
读取带宽浪费。需要保留更高分辨率时可设置 `--image-height`、`--image-width` 和 `--jpeg-quality`。

异常 episode 默认使对应 shard 失败，从而不会悄悄丢数据。已知少量坏样本且接受隔离时可显式添加
`--skip-bad-episodes`，错误会写入 shard sidecar 和最终 manifest。

## 输出契约

- `observation/base_0_rgb`：`head_color.mp4`
- `observation/left_wrist_0_rgb`：`hand_left_color.mp4`
- `observation/right_wrist_0_rgb`：`hand_right_color.mp4`
- `observation/state`：joint 14 + effector 2 + head 2 + waist 2，共 20 维
- `action`：joint 14 + effector 2 + head 2 + waist 2 + robot velocity 2，共 22 维
- `language_instruction`：逐帧使用 `action_config.action_text`，无动作分段处回退到 `task_name`

训练配置中的 source 可写为：

```yaml
- id: agibotworld_beta
  tfds_name: agibotworld_beta
  version: 1.0.0
  data_dir: /mnt/bos/dataset/RLDS
  train_split: train[:99%]
  validation_split: train[99%:]
  weight: 1.0
  normalization_id: agibotworld_beta_v1
  action_stride: 1
  state_dim: 20
  action_dim: 22
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

原始 tar 默认永久保留。成功生成 TFDS metadata 并回读后，工具会写
`verified_deletable_source_archives.txt`。只有显式传入 `--delete-source-archives-after-success` 才会删除其中“所有已索引 episode
均出现在已验证输出中”的 tar；共享但未完全覆盖的 proprio tar 不会删除。
