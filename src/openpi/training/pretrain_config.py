"""Declarative configuration types for large-scale pi0.5 pre-training."""

from __future__ import annotations

import dataclasses
import math
import pathlib
from typing import Any, Literal, TypeAlias

import flax.nnx as nnx
import tyro

from openpi.models import pi0_config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders

Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AdapterConfig:
    """Selects a safe RLDS adapter and provides adapter-specific declarative options."""

    type: str
    options: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class RldsSourceConfig:
    """One TFDS/RLDS source participating in a pre-training mixture."""

    id: str
    tfds_name: str
    version: str
    data_dir: str
    train_split: str
    validation_split: str
    weight: float
    normalization_id: str
    action_stride: int
    state_dim: int
    action_dim: int
    adapter: AdapterConfig


@dataclasses.dataclass(frozen=True)
class RldsMixtureConfig:
    """Streaming and sampling settings shared by all RLDS sources."""

    type: Literal["rlds_mixture"]
    temperature: float
    shuffle_buffer_size: int
    num_parallel_reads: int
    num_parallel_calls: int
    prefetch_batches: int
    sources: tuple[RldsSourceConfig, ...]

    def effective_probabilities(self) -> tuple[float, ...]:
        logits = [math.log(source.weight) / self.temperature for source in self.sources]
        maximum = max(logits)
        scaled = [math.exp(logit - maximum) for logit in logits]
        total = sum(scaled)
        return tuple(value / total for value in scaled)


@dataclasses.dataclass(frozen=True)
class InitializationConfig:
    """How the pi0.5 parameter tree is initialized."""

    type: Literal["paligemma", "pi05_checkpoint"]
    params_path: str | None

    def create_weight_loader(self) -> weight_loaders.WeightLoader:
        if self.type == "paligemma":
            return weight_loaders.PaliGemmaWeightLoader()
        assert self.params_path is not None
        return weight_loaders.CheckpointWeightLoader(self.params_path)


@dataclasses.dataclass(frozen=True)
class ValidationConfig:
    interval_steps: int
    batches_per_source: int


@dataclasses.dataclass(frozen=True)
class DistributedConfig:
    """Single-host FSDP and optional native JAX multi-host initialization."""

    fsdp_devices: int
    warmup_collectives: bool
    initialize: bool
    coordinator_address: str | None
    num_processes: int | None
    process_id: int | None
    local_device_ids: tuple[int, ...] | None
    cluster_detection_method: str | None
    initialization_timeout: int


@dataclasses.dataclass(frozen=True)
class PretrainConfig:
    """Complete pi0.5 pre-training configuration."""

    name: str
    project_name: str
    exp_name: str
    model: pi0_config.Pi0Config
    initialization: InitializationConfig
    data: RldsMixtureConfig
    lr_schedule: _optimizer.LRScheduleConfig
    optimizer: _optimizer.OptimizerConfig
    ema_decay: float | None
    seed: int
    batch_size: int
    num_train_steps: int
    assets_base_dir: str
    checkpoint_base_dir: str
    log_interval: int
    save_interval: int
    keep_period: int | None
    overwrite: bool
    resume: bool
    wandb_enabled: bool
    validation: ValidationConfig
    distributed: DistributedConfig
    policy_metadata: dict[str, Any] | None = None
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.exp_name:
            raise ValueError("exp_name must be non-empty")
        if not self.project_name or not self.assets_base_dir or not self.checkpoint_base_dir:
            raise ValueError("project_name and base directories must be non-empty")
        if not self.model.pi05:
            raise ValueError("Pre-training currently supports pi0.5 only; model.pi05 must be true")
        if "lora" in self.model.paligemma_variant or "lora" in self.model.action_expert_variant:
            raise ValueError("LoRA variants are not supported by the pi0.5 pre-training entrypoint")
        if self.initialization.type == "paligemma" and self.initialization.params_path is not None:
            raise ValueError("initialization.params_path must be null for paligemma initialization")
        if self.initialization.type == "pi05_checkpoint" and not self.initialization.params_path:
            raise ValueError("initialization.params_path is required for pi05_checkpoint initialization")
        if not math.isfinite(self.data.temperature) or self.data.temperature <= 0:
            raise ValueError("data.temperature must be greater than zero")
        if not self.data.sources:
            raise ValueError("data.sources must not be empty")
        source_ids = [source.id for source in self.data.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("data.sources IDs must be unique")
        for source in self.data.sources:
            if not source.id or not source.tfds_name or not source.version or not source.data_dir:
                raise ValueError("Every data source requires non-empty id, tfds_name, version, and data_dir")
            if not source.train_split or not source.validation_split:
                raise ValueError(f"Source {source.id!r} requires train_split and validation_split")
            if not source.normalization_id or not source.adapter.type:
                raise ValueError(f"Source {source.id!r} requires normalization_id and adapter.type")
            if not math.isfinite(source.weight) or source.weight <= 0:
                raise ValueError(f"Source {source.id!r} weight must be greater than zero")
            if source.action_stride <= 0:
                raise ValueError(f"Source {source.id!r} action_stride must be positive")
            if not 0 < source.state_dim <= self.model.action_dim:
                raise ValueError(
                    f"Source {source.id!r} state_dim must be in [1, model.action_dim={self.model.action_dim}]"
                )
            if not 0 < source.action_dim <= self.model.action_dim:
                raise ValueError(
                    f"Source {source.id!r} action_dim must be in [1, model.action_dim={self.model.action_dim}]"
                )
        if self.data.shuffle_buffer_size <= 0 or self.data.prefetch_batches <= 0:
            raise ValueError("RLDS shuffle and prefetch sizes must be positive")
        for field_name, value in {
            "num_parallel_reads": self.data.num_parallel_reads,
            "num_parallel_calls": self.data.num_parallel_calls,
        }.items():
            if value != -1 and value <= 0:
                raise ValueError(f"data.{field_name} must be -1 (AUTOTUNE) or positive")
        normalization_shapes: dict[str, tuple[int, int]] = {}
        for source in self.data.sources:
            shape = (source.state_dim, source.action_dim)
            previous = normalization_shapes.setdefault(source.normalization_id, shape)
            if previous != shape:
                raise ValueError(
                    f"Sources sharing normalization_id {source.normalization_id!r} must have identical "
                    f"state_dim/action_dim; got {previous} and {shape}"
                )
        if self.seed < 0 or self.batch_size <= 0 or self.num_train_steps <= 0:
            raise ValueError("seed must be non-negative and training sizes must be positive")
        if self.ema_decay is not None and (not math.isfinite(self.ema_decay) or not 0.0 < self.ema_decay < 1.0):
            raise ValueError("ema_decay must be between zero and one, or null")
        for component_name, component in (("lr_schedule", self.lr_schedule), ("optimizer", self.optimizer)):
            for field in dataclasses.fields(component):
                value = getattr(component, field.name)
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"{component_name}.{field.name} must be finite")
        if isinstance(self.lr_schedule, _optimizer.CosineDecaySchedule):
            if (
                self.lr_schedule.warmup_steps < 0
                or self.lr_schedule.decay_steps <= 0
                or self.lr_schedule.peak_lr <= 0
                or self.lr_schedule.decay_lr < 0
            ):
                raise ValueError("Invalid cosine learning-rate schedule")
        elif isinstance(self.lr_schedule, _optimizer.RsqrtDecaySchedule) and (
            self.lr_schedule.warmup_steps < 0 or self.lr_schedule.peak_lr <= 0 or self.lr_schedule.timescale <= 0
        ):
            raise ValueError("Invalid reciprocal-square-root learning-rate schedule")
        if isinstance(self.optimizer, _optimizer.AdamW):
            if (
                not 0 <= self.optimizer.b1 < 1
                or not 0 <= self.optimizer.b2 < 1
                or self.optimizer.eps <= 0
                or self.optimizer.weight_decay < 0
                or self.optimizer.clip_gradient_norm <= 0
            ):
                raise ValueError("Invalid AdamW optimizer parameters")
        elif isinstance(self.optimizer, _optimizer.SGD) and (
            self.optimizer.lr <= 0 or not 0 <= self.optimizer.momentum < 1
        ):
            raise ValueError("Invalid SGD optimizer parameters")
        if self.log_interval <= 0 or self.save_interval <= 0:
            raise ValueError("log_interval and save_interval must be positive")
        if self.keep_period is not None and self.keep_period <= 0:
            raise ValueError("keep_period must be positive or null")
        if self.validation.interval_steps <= 0 or self.validation.batches_per_source <= 0:
            raise ValueError("validation intervals and batch counts must be positive")
        if self.distributed.fsdp_devices <= 0 or self.distributed.initialization_timeout <= 0:
            raise ValueError("distributed device counts and timeout must be positive")
        if self.distributed.num_processes is not None and self.distributed.num_processes <= 0:
            raise ValueError("distributed.num_processes must be positive or null")
        if self.distributed.process_id is not None and self.distributed.process_id < 0:
            raise ValueError("distributed.process_id must be non-negative or null")
        if (
            self.distributed.num_processes is not None
            and self.distributed.process_id is not None
            and self.distributed.process_id >= self.distributed.num_processes
        ):
            raise ValueError("distributed.process_id must be smaller than distributed.num_processes")
        if self.overwrite and self.resume:
            raise ValueError("overwrite and resume cannot both be true")

    @property
    def weight_loader(self) -> weight_loaders.WeightLoader:
        return self.initialization.create_weight_loader()

    @property
    def fsdp_devices(self) -> int:
        return self.distributed.fsdp_devices

    @property
    def assets_dirs(self) -> pathlib.Path:
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    @property
    def source_indices(self) -> dict[str, int]:
        return {source.id: index for index, source in enumerate(self.data.sources)}
