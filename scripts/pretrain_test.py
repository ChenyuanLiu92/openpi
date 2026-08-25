from __future__ import annotations

import threading
import types

import numpy as np
import pytest

from . import pretrain


def _shard(start: int, values: list[int]):
    data = np.asarray(values, dtype=np.int32)[:, None]
    return types.SimpleNamespace(index=(slice(start, start + len(values)), slice(None)), data=data)


def test_addressable_shard_prefix_sorts_and_limits_local_examples():
    result = pretrain._addressable_shard_prefix(  # noqa: SLF001
        [_shard(4, [4, 5]), _shard(0, [0, 1, 2, 3])],
        limit=5,
    )

    assert result[:, 0].tolist() == [0, 1, 2, 3, 4]


def test_process_local_prefix_handles_fully_addressable_array():
    values = np.arange(10)[:, None]

    result = pretrain._process_local_prefix(values, limit=3)  # noqa: SLF001

    assert result[:, 0].tolist() == [0, 1, 2]


def test_process_local_prefix_rejects_invalid_limit():
    with pytest.raises(ValueError, match="positive"):
        pretrain._process_local_prefix(np.arange(3), limit=0)  # noqa: SLF001


def test_validation_batch_availability_is_synchronized_across_processes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pretrain.jax, "process_count", lambda: 2)
    monkeypatch.setattr(
        pretrain.multihost_utils,
        "process_allgather",
        lambda _: np.asarray([[1], [0]], dtype=np.int32),
    )

    with pytest.raises(RuntimeError, match="1/2 ranks have data"):
        pretrain._next_synchronized_validation_batch(  # noqa: SLF001
            iter([object()]), source_id="mock", split="validation", batch_index=0
        )


def test_fatal_watchdog_forces_nonzero_exit(monkeypatch: pytest.MonkeyPatch):
    called = threading.Event()
    exit_codes: list[int] = []

    def fake_exit(code: int) -> None:
        exit_codes.append(code)
        called.set()

    monkeypatch.setattr(pretrain.os, "_exit", fake_exit)
    pretrain._start_fatal_watchdog(0.01, 7)  # noqa: SLF001

    assert called.wait(timeout=1)
    assert exit_codes == [7]


@pytest.mark.parametrize("code", [None, 0])
def test_clean_system_exit_skips_fatal_cleanup(code: int | None):
    assert not pretrain._requires_fatal_cleanup(SystemExit(code))  # noqa: SLF001


@pytest.mark.parametrize("exc", [SystemExit(2), KeyboardInterrupt(), RuntimeError("boom")])
def test_failure_requires_fatal_cleanup(exc: BaseException):
    assert pretrain._requires_fatal_cleanup(exc)  # noqa: SLF001
