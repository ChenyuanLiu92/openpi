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
class PipelineConfig:
    """Host preprocessing, prefetching, and bad-sample policy."""

    tokenizer_threads: int
    host_prefetch_batches: int
    device_prefetch_batches: int
    bad_sample_policy: Literal["fail", "skip", "quarantine"]
    bad_sample_window: int
    max_bad_samples: int
    max_bad_fraction: float
    quarantine_dir: str | None


@dataclasses.dataclass(frozen=True)
class MixingSchedulePoint:
    """Mixture controls taking effect at one optimizer step."""

    step: int
    temperature: float
    weights: dict[str, float]


@dataclasses.dataclass(frozen=True)
class SourceLimit:
    """Probability and lifetime sampling bounds for one source."""

    min_probability: float
    max_probability: float
    min_samples: int | None
    max_samples: int | None


@dataclasses.dataclass(frozen=True)
class MixingConfig:
    """Runtime mixture schedule, bounds, and source-failure behavior."""

    interpolation: Literal["step", "linear"]
    schedule: tuple[MixingSchedulePoint, ...]
    source_limits: dict[str, SourceLimit]
    source_failure_policy: Literal["fail", "degrade"]
    consecutive_failure_threshold: int


@dataclasses.dataclass(frozen=True)
class RldsMixtureConfig:
    """Streaming and sampling settings shared by all RLDS sources."""

    type: Literal["rlds_mixture"]
    temperature: float
    shuffle_buffer_size: int
    num_parallel_reads: int
    num_parallel_calls: int
    prefetch_batches: int
    pipeline: PipelineConfig
    mixing: MixingConfig
    sources: tuple[RldsSourceConfig, ...]

    def effective_probabilities(self, step: int = 0) -> tuple[float, ...]:
        temperature, weights = self.mixture_at_step(step)
        logits = [math.log(weights[source.id]) / temperature for source in self.sources]
        maximum = max(logits)
        scaled = [math.exp(logit - maximum) for logit in logits]
        total = sum(scaled)
        probabilities = [value / total for value in scaled]
        minimums = []
        maximums = []
        for source in self.sources:
            limit = self.mixing.source_limits.get(source.id)
            minimums.append(0.0 if limit is None else limit.min_probability)
            maximums.append(1.0 if limit is None else limit.max_probability)
        return _project_bounded_probabilities(probabilities, minimums, maximums)

    def mixture_at_step(self, step: int) -> tuple[float, dict[str, float]]:
        """Return scheduled temperature and weights at ``step``."""
        base_weights = {source.id: source.weight for source in self.sources}
        points = self.mixing.schedule
        if not points or step < points[0].step:
            return self.temperature, base_weights
        left = max((point for point in points if point.step <= step), key=lambda point: point.step)
        if self.mixing.interpolation == "step":
            return left.temperature, {**base_weights, **left.weights}
        right = next((point for point in points if point.step > step), None)
        if right is None:
            return left.temperature, {**base_weights, **left.weights}
        ratio = (step - left.step) / (right.step - left.step)
        left_weights = {**base_weights, **left.weights}
        right_weights = {**base_weights, **right.weights}
        return (
            left.temperature + ratio * (right.temperature - left.temperature),
            {key: left_weights[key] + ratio * (right_weights[key] - left_weights[key]) for key in base_weights},
        )


def _project_bounded_probabilities(
    probabilities: list[float], minimums: list[float], maximums: list[float]
) -> tuple[float, ...]:
    """Project positive weights onto a probability simplex with per-source bounds."""
    result = [0.0] * len(probabilities)
    free = set(range(len(probabilities)))
    remaining = 1.0
    while free:
        weight_total = sum(probabilities[index] for index in free)
        proposed = {
            index: remaining * probabilities[index] / weight_total if weight_total else remaining / len(free)
            for index in free
        }
        violated = False
        for index in tuple(free):
            if proposed[index] < minimums[index]:
                result[index] = minimums[index]
            elif proposed[index] > maximums[index]:
                result[index] = maximums[index]
            else:
                continue
            remaining -= result[index]
            free.remove(index)
            violated = True
        if not violated:
            for index in free:
                result[index] = proposed[index]
            break
    return tuple(result)


@dataclasses.dataclass(frozen=True)
class InitializationConfig:
    """How the pi0.5 parameter tree is initialized."""

    type: Literal["random", "paligemma", "pi05_checkpoint"]
    params_path: str | None

    def create_weight_loader(self) -> weight_loaders.WeightLoader:
        if self.type == "random":
            return weight_loaders.NoOpWeightLoader()
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
    coordinator_bind_address: str | None
    num_processes: int | None
    process_id: int | None
    local_device_ids: tuple[int, ...] | None
    cluster_detection_method: str | None
    initialization_timeout: int
    diagnostics: DistributedDiagnosticsConfig


@dataclasses.dataclass(frozen=True)
class DistributedDiagnosticsConfig:
    topology_check: bool
    tensor_sizes_mib: tuple[float, ...]
    warmup_iterations: int
    measure_iterations: int
    straggler_ratio_threshold: float
    profile_start_step: int | None
    profile_num_steps: int


@dataclasses.dataclass(frozen=True)
class CompilationCacheConfig:
    """Persistent JAX compilation-cache settings."""

    enabled: bool
    directory: str
    minimum_compile_time_seconds: float
    explain_misses: bool


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    """Process runtime and fatal-cleanup behavior."""

    compilation_cache: CompilationCacheConfig
    fatal_cleanup_timeout_seconds: float


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
    micro_batch_size: int | None
    gradient_accumulation_steps: int
    data_resume_mode: Literal["statistical", "exact"]
    on_topology_change: Literal["error", "statistical"]
    num_train_steps: int
    assets_base_dir: str
    checkpoint_base_dir: str
    log_interval: int
    save_interval: int
    keep_period: int | None
    overwrite: bool
    resume: bool
    wandb_enabled: bool
    wandb_mode: Literal["online", "offline", "disabled"]
    wandb_entity: str | None
    observability_local_root: str | None
    wandb_tags: tuple[str, ...]
    system_interval_seconds: int
    heartbeat_interval_seconds: int
    stall_timeout_seconds: int
    emergency_checkpoint_timeout_seconds: int
    webhook_url_env: str
    min_free_space_gib: int
    raw_retention_days: int
    validation: ValidationConfig
    distributed: DistributedConfig
    runtime: RuntimeConfig
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
        if self.initialization.type in {"random", "paligemma"} and self.initialization.params_path is not None:
            raise ValueError(f"initialization.params_path must be null for {self.initialization.type} initialization")
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
        pipeline = self.data.pipeline
        if (
            pipeline.tokenizer_threads <= 0
            or pipeline.host_prefetch_batches < 0
            or pipeline.device_prefetch_batches < 0
        ):
            raise ValueError("pipeline tokenizer_threads must be positive and prefetch sizes must be non-negative")
        if pipeline.bad_sample_window <= 0 or pipeline.max_bad_samples < 0:
            raise ValueError("pipeline bad-sample window must be positive and max_bad_samples non-negative")
        if not 0.0 <= pipeline.max_bad_fraction <= 1.0:
            raise ValueError("pipeline.max_bad_fraction must be in [0, 1]")
        if pipeline.bad_sample_policy == "quarantine" and not pipeline.quarantine_dir:
            raise ValueError("pipeline.quarantine_dir is required when bad_sample_policy is quarantine")
        schedule_steps = [point.step for point in self.data.mixing.schedule]
        if schedule_steps != sorted(set(schedule_steps)) or any(step < 0 for step in schedule_steps):
            raise ValueError("mixing schedule steps must be unique, non-negative, and increasing")
        for point in self.data.mixing.schedule:
            if not math.isfinite(point.temperature) or point.temperature <= 0:
                raise ValueError("mixing schedule temperatures must be finite and positive")
            unknown = set(point.weights) - set(source_ids)
            if unknown or any(not math.isfinite(weight) or weight <= 0 for weight in point.weights.values()):
                raise ValueError(f"Invalid mixing schedule weights; unknown sources: {sorted(unknown)}")
        if self.data.mixing.consecutive_failure_threshold <= 0:
            raise ValueError("mixing.consecutive_failure_threshold must be positive")
        if set(self.data.mixing.source_limits) - set(source_ids):
            raise ValueError("mixing.source_limits contains unknown source IDs")
        for source_id, limit in self.data.mixing.source_limits.items():
            if not 0 <= limit.min_probability <= limit.max_probability <= 1:
                raise ValueError(f"Invalid probability limits for source {source_id!r}")
            if limit.min_samples is not None and limit.min_samples < 0:
                raise ValueError(f"min_samples for source {source_id!r} must be non-negative or null")
            if limit.max_samples is not None and limit.max_samples <= 0:
                raise ValueError(f"max_samples for source {source_id!r} must be positive or null")
            if (
                limit.min_samples is not None
                and limit.max_samples is not None
                and limit.min_samples > limit.max_samples
            ):
                raise ValueError(f"min_samples must not exceed max_samples for source {source_id!r}")
        probability_minimum = sum(
            self.data.mixing.source_limits.get(source_id, SourceLimit(0, 1, None, None)).min_probability
            for source_id in source_ids
        )
        probability_maximum = sum(
            self.data.mixing.source_limits.get(source_id, SourceLimit(0, 1, None, None)).max_probability
            for source_id in source_ids
        )
        if probability_minimum > 1 + 1e-9 or probability_maximum < 1 - 1e-9:
            raise ValueError(
                "mixing.source_limits probability bounds are infeasible: "
                f"sum(min)={probability_minimum}, sum(max)={probability_maximum}"
            )
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
        if self.micro_batch_size is not None and self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive or null")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        minimum_samples = sum(limit.min_samples or 0 for limit in self.data.mixing.source_limits.values())
        if minimum_samples > self.num_train_steps * self.batch_size:
            raise ValueError(
                "Sum of mixing.source_limits min_samples exceeds the total global examples in this run: "
                f"{minimum_samples} > {self.num_train_steps * self.batch_size}"
            )
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
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be online, offline, or disabled")
        for field_name, value in {
            "system_interval_seconds": self.system_interval_seconds,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "stall_timeout_seconds": self.stall_timeout_seconds,
            "emergency_checkpoint_timeout_seconds": self.emergency_checkpoint_timeout_seconds,
            "raw_retention_days": self.raw_retention_days,
        }.items():
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.min_free_space_gib < 0:
            raise ValueError("min_free_space_gib must be non-negative")
        if not self.webhook_url_env:
            raise ValueError("webhook_url_env must be non-empty")
        if self.keep_period is not None and self.keep_period <= 0:
            raise ValueError("keep_period must be positive or null")
        if self.validation.interval_steps <= 0 or self.validation.batches_per_source <= 0:
            raise ValueError("validation intervals and batch counts must be positive")
        if self.distributed.fsdp_devices <= 0 or self.distributed.initialization_timeout <= 0:
            raise ValueError("distributed device counts and timeout must be positive")
        diagnostics = self.distributed.diagnostics
        if not diagnostics.tensor_sizes_mib or any(
            size <= 0 or not math.isfinite(size) for size in diagnostics.tensor_sizes_mib
        ):
            raise ValueError("distributed.diagnostics.tensor_sizes_mib must contain positive finite values")
        if diagnostics.warmup_iterations < 0 or diagnostics.measure_iterations <= 0:
            raise ValueError("diagnostic warmup must be non-negative and measure_iterations positive")
        if diagnostics.straggler_ratio_threshold <= 1 or not math.isfinite(diagnostics.straggler_ratio_threshold):
            raise ValueError("diagnostic straggler_ratio_threshold must be finite and greater than one")
        if diagnostics.profile_start_step is not None and diagnostics.profile_start_step < 0:
            raise ValueError("diagnostic profile_start_step must be non-negative or null")
        if diagnostics.profile_num_steps <= 0:
            raise ValueError("diagnostic profile_num_steps must be positive")
        if not self.runtime.compilation_cache.directory:
            raise ValueError("runtime.compilation_cache.directory must be non-empty")
        if (
            not math.isfinite(self.runtime.compilation_cache.minimum_compile_time_seconds)
            or self.runtime.compilation_cache.minimum_compile_time_seconds < 0
        ):
            raise ValueError("runtime.compilation_cache.minimum_compile_time_seconds must be finite and non-negative")
        if (
            not math.isfinite(self.runtime.fatal_cleanup_timeout_seconds)
            or self.runtime.fatal_cleanup_timeout_seconds <= 0
        ):
            raise ValueError("runtime.fatal_cleanup_timeout_seconds must be finite and positive")
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
        local_device_ids = self.distributed.local_device_ids
        if local_device_ids is not None:
            if not local_device_ids:
                raise ValueError("distributed.local_device_ids must not be empty")
            if any(device_id < 0 for device_id in local_device_ids):
                raise ValueError("distributed.local_device_ids must be non-negative")
            if len(local_device_ids) != len(set(local_device_ids)):
                raise ValueError("distributed.local_device_ids must be unique")
        explicit_values = (
            self.distributed.coordinator_address,
            self.distributed.num_processes,
            self.distributed.process_id,
            self.distributed.local_device_ids,
        )
        has_explicit_values = any(value is not None for value in explicit_values)
        if not self.distributed.initialize and (
            has_explicit_values
            or self.distributed.coordinator_bind_address is not None
            or self.distributed.cluster_detection_method is not None
        ):
            raise ValueError("distributed.initialize must be true when distributed initialization fields are set")
        if self.distributed.initialize:
            if self.distributed.cluster_detection_method is not None and has_explicit_values:
                raise ValueError(
                    "Use either distributed.cluster_detection_method or explicit coordinator/rank/device fields, not both"
                )
            if self.distributed.cluster_detection_method is None and not all(
                value is not None for value in explicit_values
            ):
                raise ValueError(
                    "Explicit distributed initialization requires coordinator_address, num_processes, process_id, "
                    "and local_device_ids"
                )
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
    def observability_root(self) -> pathlib.Path:
        if self.observability_local_root is not None:
            return pathlib.Path(self.observability_local_root).expanduser().resolve()
        return self.checkpoint_dir / "observability"

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    @property
    def source_indices(self) -> dict[str, int]:
        return {source.id: index for index, source in enumerate(self.data.sources)}
