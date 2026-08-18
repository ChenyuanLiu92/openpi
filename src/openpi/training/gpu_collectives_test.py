import logging
import pathlib

import numpy as np
import pytest

from openpi.training import gpu_collectives


class _FakeDevice:
    device_kind = "fake"
    compute_capability = "12.0"

    def __str__(self) -> str:
        return "fake:0"


def test_probe_skips_one_device():
    result = gpu_collectives.run_local_fsdp_collective_probe(devices=(_FakeDevice(),))

    assert result.device_count == 1
    assert result.elapsed_seconds == ()


def test_probe_requires_multiple_devices():
    with pytest.raises(RuntimeError, match="at least two visible JAX devices"):
        gpu_collectives.run_local_fsdp_collective_probe(devices=(_FakeDevice(),), require_multiple_devices=True)


def test_validate_collective_results_rejects_wrong_all_reduce():
    with pytest.raises(RuntimeError, match="AllReduce produced"):
        gpu_collectives._validate_collective_results(  # noqa: SLF001
            np.array([0.0, 0.0], dtype=np.float32),
            np.empty((2, 4, 1), dtype=np.float32),
            np.empty((2, 1, 1), dtype=np.float32),
            expected_sum=1.0,
            device_count=2,
            payload_rows=2,
            payload_columns=1,
        )


def test_cuda_cache_warnings(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: pathlib.Path):
    missing_cache = tmp_path / "missing-parent" / "cache"
    monkeypatch.setenv("CUDA_CACHE_DISABLE", "1")
    monkeypatch.setenv("CUDA_CACHE_PATH", str(missing_cache))
    monkeypatch.setattr(gpu_collectives, "_path_is_writable", lambda _: False)

    with caplog.at_level(logging.INFO):
        gpu_collectives.log_cuda_cache_configuration((_FakeDevice(),))

    assert "disables the driver JIT cache" in caplog.text
    assert "is not writable" in caplog.text
    assert "message after 10 seconds is a progress warning" in caplog.text
