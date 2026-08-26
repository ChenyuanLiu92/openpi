# 预训练生产前 GPU 验收

统一入口 `scripts/validate_pretrain_gpu.py` 将 GPU 验收分为被动健康检查、计算/显存压力、collective 和真实模型集成四个阶段。
主动阶段不会终止机器上的其他任务；资源不足时等待或失败，并将原因写入报告。

## 标准验收

先用 dry-run 检查计划，不创建目录或初始化 JAX：

```bash
uv run scripts/validate_pretrain_gpu.py \
  --work-dir /mnt/pfs/rhos-vla/chenyuan/gpu-validation/<run-id> \
  --dry-run
```

标准约 30 分钟的主动验收命令为：

```bash
uv run scripts/validate_pretrain_gpu.py \
  --work-dir /mnt/pfs/rhos-vla/chenyuan/gpu-validation/<run-id> \
  --phase all \
  --burn-seconds 600 \
  --memory-gib-per-gpu 75 \
  --min-memory-headroom-gib 140 \
  --max-load-one 45 \
  --max-cpu-psi-full-avg60 1 \
  --min-pfs-free-gib 100
```

只需在繁忙机器上获取无分配健康快照时使用：

```bash
uv run scripts/validate_pretrain_gpu.py \
  --work-dir /mnt/pfs/rhos-vla/chenyuan/gpu-validation/<run-id> \
  --phase passive
```

主动阶段要求 GPU 独占，并按 cgroup working set（`memory.current - inactive_file`）计算内存余量。可用
`--wait-for-resources-seconds` 等待现有任务结束；超时后只退出本次验收。TMPDIR、CUDA cache、JAX cache、mock RLDS 和 checkpoint
均位于 `--work-dir`，不会继续占用容器根盘。

## 测试内容与产物

- burn：8 卡并发 BF16 GEMM；每卡约 75 GiB device-side pattern/checksum；默认要求 GPU 吞吐差异不超过 10%。
- collective：以 `1×8`、NUMA 对齐的 `2×4`、`4×2`、`8×1` 测试 AllReduce、AllGather、ReduceScatter；payload 为
  1/16/64/256/1024 MiB，每种拓扑独立启动两轮。
- model：完整 π0.5 在 `1×8` 和 `2×4` 纯 DP 下完成训练/checkpoint/resume；dummy 模型验证 FSDP 2/4 mesh。
- health：前后对比 ECC、row remap、Xid、PCIe width、温度和 recovery action。

成功后保留 `report.json`、`summary.md`、telemetry、命令日志和 `baselines/<topology>.json`，删除大体积 mock 数据、checkpoint
和编译缓存。失败时保留全部现场。若要调试成功运行，可加 `--retain-success-artifacts`。

生产实验应选择与实际 launcher topology 一致的 baseline：

```yaml
distributed:
  diagnostics:
    collective_baseline_path: /mnt/pfs/rhos-vla/chenyuan/gpu-validation/<run-id>/baselines/2x4.json
    minimum_baseline_fraction: 0.8
    bandwidth_regression_policy: fail
```

该测试不代表真实物理多节点/RDMA 验收，也不覆盖 tokenizer、RLDS 输入吞吐、host 内存和 PFS checkpoint 性能。
