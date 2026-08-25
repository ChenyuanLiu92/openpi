import dataclasses
import pathlib

import jax
import numpy as np
import pytest
import torch

from openpi.training import config as _config
from openpi.training import optimizer as _optimizer

from . import train_pytorch


@pytest.mark.parametrize(
    "schedule_config",
    [
        _optimizer.CosineDecaySchedule(warmup_steps=3, peak_lr=1e-3, decay_steps=10, decay_lr=1e-4),
        _optimizer.RsqrtDecaySchedule(warmup_steps=3, peak_lr=1e-3, timescale=10),
    ],
)
def test_lr_schedule_matches_jax(schedule_config):
    pytorch_schedule = train_pytorch._create_lr_schedule(schedule_config)  # noqa: SLF001
    jax_schedule = schedule_config.create()

    for step in (0, 1, 2, 3, 4, 9, 10, 12):
        np.testing.assert_allclose(pytorch_schedule(step), jax.device_get(jax_schedule(step)), rtol=1e-6)


@pytest.mark.parametrize(
    ("optimizer_config", "expected_type"),
    [
        (_optimizer.AdamW(), torch.optim.AdamW),
        (_optimizer.SGD(momentum=0.8, nesterov=True), torch.optim.SGD),
    ],
)
def test_create_optimizer_supports_all_registered_types(optimizer_config, expected_type):
    model = torch.nn.Linear(2, 1)

    result = train_pytorch._create_optimizer(model, optimizer_config, initial_lr=1e-3)  # noqa: SLF001

    assert isinstance(result, expected_type)
    assert result.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_sgd_gradient_norm_does_not_clip():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.tensor([[3.0]])

    norm = train_pytorch._gradient_norm(model, _optimizer.SGD())  # noqa: SLF001

    assert float(norm) == pytest.approx(3.0)
    assert model.weight.grad.item() == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("step", "save_interval", "num_steps", "expected"),
    [
        (0, 1000, 1, False),
        (1, 1000, 1, True),
        (2, 2, 5, True),
        (4, 3, 5, False),
        (5, 3, 5, True),
        (6, 3, 10, True),
    ],
)
def test_should_save_checkpoint(step: int, save_interval: int, num_steps: int, expected):
    assert (
        train_pytorch._should_save_checkpoint(  # noqa: SLF001
            step,
            save_interval=save_interval,
            num_train_steps=num_steps,
        )
        is expected
    )


def test_save_checkpoint_writes_the_final_step(tmp_path: pathlib.Path):
    config = dataclasses.replace(
        _config.get_config("debug"),
        checkpoint_base_dir=str(tmp_path),
        exp_name="final-step",
        num_train_steps=1,
        save_interval=1000,
        wandb_enabled=False,
    )
    config.checkpoint_dir.mkdir(parents=True)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    train_pytorch.save_checkpoint(
        model=model,
        optimizer=optimizer,
        global_step=1,
        config=config,
        is_main=True,
        data_config=_config.DataConfig(),
        config_snapshot={"schema_version": 1},
    )

    checkpoint_dir = config.checkpoint_dir / "1"
    metadata = torch.load(checkpoint_dir / "metadata.pt", weights_only=False)
    assert (checkpoint_dir / "model.safetensors").is_file()
    assert (checkpoint_dir / "metadata" / "train_config.yaml").is_file()
    assert metadata["global_step"] == 1


def test_validate_pytorch_config_accepts_sgd_rsqrt():
    config = dataclasses.replace(
        _config.get_config("debug"),
        optimizer=_optimizer.SGD(),
        lr_schedule=_optimizer.RsqrtDecaySchedule(),
    )

    train_pytorch._validate_pytorch_config(config)  # noqa: SLF001


@pytest.mark.parametrize(
    ("lr_schedule", "message"),
    [
        (_optimizer.CosineDecaySchedule(warmup_steps=3, decay_steps=3), "Invalid cosine"),
        (_optimizer.RsqrtDecaySchedule(peak_lr=float("nan")), "must be finite"),
    ],
)
def test_validate_pytorch_config_rejects_invalid_schedule(lr_schedule, message: str):
    config = dataclasses.replace(_config.get_config("debug"), lr_schedule=lr_schedule)

    with pytest.raises(ValueError, match=message):
        train_pytorch._validate_pytorch_config(config)  # noqa: SLF001
