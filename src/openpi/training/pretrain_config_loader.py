"""Safe YAML loading and provenance snapshots for pi0.5 pre-training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import datetime
import enum
import pathlib
import subprocess
import sys
import types
import typing
from typing import Any, Literal, get_args, get_origin, get_type_hints

import tyro
import yaml

from openpi.models import pi0_config
from openpi.training import optimizer
from openpi.training import pretrain_config as _config

SCHEMA_VERSION = 1
KIND = "pi05_pretrain"


class ConfigError(ValueError):
    """Raised when a pre-training YAML manifest is invalid."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict:  # noqa: FBT001, FBT002
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigError(f"Duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


@dataclasses.dataclass(frozen=True)
class ResolvedPretrainConfig:
    config: _config.PretrainConfig
    source: str
    manifest: dict[str, Any]
    cli_args: tuple[str, ...] = ()

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "name": self.config.name,
            "source": self.source,
            "manifest": self.manifest,
            "cli_args": list(self.cli_args),
            "resolved_config": _to_serializable(self.config),
            "resolved_mixture_probabilities": {
                source.id: probability
                for source, probability in zip(
                    self.config.data.sources, self.config.data.effective_probabilities(), strict=True
                )
            },
            "git": _git_provenance(),
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }


def load(path: str | pathlib.Path) -> ResolvedPretrainConfig:
    config_path = pathlib.Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Pre-training config does not exist: {config_path}")
    try:
        contents = yaml.load(config_path.read_text(), Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(contents, dict):
        raise ConfigError("Pre-training config must contain a YAML mapping")
    try:
        config = _build_config(contents)
        _validate_adapters(config)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(str(exc)) from exc
    return ResolvedPretrainConfig(config, str(config_path.resolve()), contents)


def parse_cli(args: Sequence[str] | None = None) -> ResolvedPretrainConfig:
    cli_args = list(sys.argv[1:] if args is None else args)
    if not cli_args:
        raise ConfigError("Usage: pretrain.py <config.yaml> [overrides]")
    resolved = load(cli_args[0])
    config = tyro.cli(_config.PretrainConfig, default=resolved.config, args=cli_args[1:])
    _validate_adapters(config)
    return dataclasses.replace(resolved, config=config, cli_args=tuple(cli_args[1:]))


def write_snapshot(path: str | pathlib.Path, snapshot: Mapping[str, Any]) -> None:
    output_path = pathlib.Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(dict(snapshot), sort_keys=False, allow_unicode=True))


def _build_config(contents: dict[str, Any]) -> _config.PretrainConfig:
    top_fields = {
        "schema_version",
        "kind",
        "name",
        "model",
        "initialization",
        "data",
        "lr_schedule",
        "optimizer",
        "training",
        "paths",
        "checkpoint",
        "logging",
        "distributed",
        "validation",
        "policy_metadata",
    }
    _check_keys(contents, top_fields, top_fields, "config")
    if contents["schema_version"] != SCHEMA_VERSION:
        raise ConfigError(f"Unsupported schema_version {contents['schema_version']!r}; expected {SCHEMA_VERSION}")
    if contents["kind"] != KIND:
        raise ConfigError(f"Unsupported kind {contents['kind']!r}; expected {KIND!r}")

    model_payload = dict(_mapping(contents["model"], "model"))
    if model_payload.pop("type", None) != "pi0":
        raise ConfigError("model.type must be 'pi0' for pi0.5 pre-training")
    model = _strict_dataclass(pi0_config.Pi0Config(), model_payload, "model")

    initialization = _section(
        contents["initialization"], "initialization", {"type", "params_path"}, nullable={"params_path"}
    )
    if initialization["type"] not in {"paligemma", "pi05_checkpoint"}:
        raise ConfigError("initialization.type must be 'paligemma' or 'pi05_checkpoint'")

    data_payload = dict(_mapping(contents["data"], "data"))
    data_fields = {
        "type",
        "temperature",
        "shuffle_buffer_size",
        "num_parallel_reads",
        "num_parallel_calls",
        "prefetch_batches",
        "sources",
    }
    _check_keys(data_payload, data_fields, data_fields, "data")
    if data_payload["type"] != "rlds_mixture":
        raise ConfigError("data.type must be 'rlds_mixture'")
    source_values = data_payload["sources"]
    if not isinstance(source_values, list) or not source_values:
        raise ConfigError("data.sources must be a non-empty list")
    sources = tuple(_build_source(value, index) for index, value in enumerate(source_values))
    data = _config.RldsMixtureConfig(
        type="rlds_mixture",
        temperature=_number(data_payload["temperature"], "data.temperature"),
        shuffle_buffer_size=_integer(data_payload["shuffle_buffer_size"], "data.shuffle_buffer_size"),
        num_parallel_reads=_integer(data_payload["num_parallel_reads"], "data.num_parallel_reads"),
        num_parallel_calls=_integer(data_payload["num_parallel_calls"], "data.num_parallel_calls"),
        prefetch_batches=_integer(data_payload["prefetch_batches"], "data.prefetch_batches"),
        sources=sources,
    )

    lr_schedule = _build_component(
        contents["lr_schedule"],
        "lr_schedule",
        {"cosine_decay": optimizer.CosineDecaySchedule(), "rsqrt_decay": optimizer.RsqrtDecaySchedule()},
    )
    optimizer_config = _build_component(
        contents["optimizer"], "optimizer", {"adamw": optimizer.AdamW(), "sgd": optimizer.SGD()}
    )
    training = _section(
        contents["training"],
        "training",
        {"seed", "batch_size", "num_train_steps", "ema_decay"},
        nullable={"ema_decay"},
    )
    paths = _section(contents["paths"], "paths", {"assets_base_dir", "checkpoint_base_dir"})
    checkpoint = _section(
        contents["checkpoint"],
        "checkpoint",
        {"exp_name", "save_interval", "keep_period", "overwrite", "resume"},
        nullable={"keep_period"},
    )
    logging_config = _section(contents["logging"], "logging", {"project_name", "wandb_enabled", "log_interval"})
    distributed = _section(
        contents["distributed"],
        "distributed",
        {
            "fsdp_devices",
            "warmup_collectives",
            "initialize",
            "coordinator_address",
            "num_processes",
            "process_id",
            "local_device_ids",
            "cluster_detection_method",
            "initialization_timeout",
        },
        nullable={
            "coordinator_address",
            "num_processes",
            "process_id",
            "local_device_ids",
            "cluster_detection_method",
        },
    )
    validation = _section(contents["validation"], "validation", {"interval_steps", "batches_per_source"})
    local_device_ids = distributed["local_device_ids"]
    if local_device_ids is not None:
        if not isinstance(local_device_ids, list) or not all(isinstance(item, int) for item in local_device_ids):
            raise ConfigError("distributed.local_device_ids must be a list of integers or null")
        local_device_ids = tuple(local_device_ids)
    policy_metadata = contents["policy_metadata"]
    if policy_metadata is not None and not isinstance(policy_metadata, dict):
        raise ConfigError("policy_metadata must be a mapping or null")

    return _config.PretrainConfig(
        name=_string(contents["name"], "name"),
        project_name=_string(logging_config["project_name"], "logging.project_name"),
        exp_name=_string(checkpoint["exp_name"], "checkpoint.exp_name"),
        model=model,
        initialization=_config.InitializationConfig(
            initialization["type"],
            _nullable_string(initialization["params_path"], "initialization.params_path"),
        ),
        data=data,
        lr_schedule=lr_schedule,
        optimizer=optimizer_config,
        ema_decay=_nullable_number(training["ema_decay"], "training.ema_decay"),
        seed=_integer(training["seed"], "training.seed"),
        batch_size=_integer(training["batch_size"], "training.batch_size"),
        num_train_steps=_integer(training["num_train_steps"], "training.num_train_steps"),
        assets_base_dir=_string(paths["assets_base_dir"], "paths.assets_base_dir"),
        checkpoint_base_dir=_string(paths["checkpoint_base_dir"], "paths.checkpoint_base_dir"),
        log_interval=_integer(logging_config["log_interval"], "logging.log_interval"),
        save_interval=_integer(checkpoint["save_interval"], "checkpoint.save_interval"),
        keep_period=_nullable_integer(checkpoint["keep_period"], "checkpoint.keep_period"),
        overwrite=_boolean(checkpoint["overwrite"], "checkpoint.overwrite"),
        resume=_boolean(checkpoint["resume"], "checkpoint.resume"),
        wandb_enabled=_boolean(logging_config["wandb_enabled"], "logging.wandb_enabled"),
        validation=_config.ValidationConfig(
            _integer(validation["interval_steps"], "validation.interval_steps"),
            _integer(validation["batches_per_source"], "validation.batches_per_source"),
        ),
        distributed=_config.DistributedConfig(
            fsdp_devices=_integer(distributed["fsdp_devices"], "distributed.fsdp_devices"),
            warmup_collectives=_boolean(distributed["warmup_collectives"], "distributed.warmup_collectives"),
            initialize=_boolean(distributed["initialize"], "distributed.initialize"),
            coordinator_address=_nullable_string(distributed["coordinator_address"], "distributed.coordinator_address"),
            num_processes=_nullable_integer(distributed["num_processes"], "distributed.num_processes"),
            process_id=_nullable_integer(distributed["process_id"], "distributed.process_id"),
            local_device_ids=local_device_ids,
            cluster_detection_method=_nullable_string(
                distributed["cluster_detection_method"], "distributed.cluster_detection_method"
            ),
            initialization_timeout=_integer(
                distributed["initialization_timeout"], "distributed.initialization_timeout"
            ),
        ),
        policy_metadata=policy_metadata,
    )


def _build_source(value: Any, index: int) -> _config.RldsSourceConfig:
    path = f"data.sources[{index}]"
    fields = {
        "id",
        "tfds_name",
        "version",
        "data_dir",
        "train_split",
        "validation_split",
        "weight",
        "normalization_id",
        "action_stride",
        "state_dim",
        "action_dim",
        "adapter",
    }
    payload = _section(value, path, fields)
    adapter_payload = _section(payload["adapter"], f"{path}.adapter", {"type", "options"})
    options = adapter_payload["options"]
    if not isinstance(options, dict):
        raise ConfigError(f"{path}.adapter.options must be a mapping")
    return _config.RldsSourceConfig(
        id=_string(payload["id"], f"{path}.id"),
        tfds_name=_string(payload["tfds_name"], f"{path}.tfds_name"),
        version=_string(payload["version"], f"{path}.version"),
        data_dir=_string(payload["data_dir"], f"{path}.data_dir"),
        train_split=_string(payload["train_split"], f"{path}.train_split"),
        validation_split=_string(payload["validation_split"], f"{path}.validation_split"),
        weight=_number(payload["weight"], f"{path}.weight"),
        normalization_id=_string(payload["normalization_id"], f"{path}.normalization_id"),
        action_stride=_integer(payload["action_stride"], f"{path}.action_stride"),
        state_dim=_integer(payload["state_dim"], f"{path}.state_dim"),
        action_dim=_integer(payload["action_dim"], f"{path}.action_dim"),
        adapter=_config.AdapterConfig(_string(adapter_payload["type"], f"{path}.adapter.type"), options),
    )


def _validate_adapters(config: _config.PretrainConfig) -> None:
    # Import lazily to keep the config dataclasses independent from the runtime registry.
    from openpi.training import rlds_adapters

    for source in config.data.sources:
        rlds_adapters.create_adapter(source)


def _build_component(section: Any, path: str, registry: Mapping[str, Any]) -> Any:
    payload = dict(_mapping(section, path))
    type_name = payload.pop("type", None)
    if type_name not in registry:
        raise ConfigError(f"Unknown {path}.type {type_name!r}; expected one of {sorted(registry)}")
    return _strict_dataclass(registry[type_name], payload, path)


def _strict_dataclass(prototype: Any, payload: Mapping[str, Any], path: str) -> Any:
    fields = {field.name for field in dataclasses.fields(prototype) if field.init}
    _check_keys(payload, fields, fields, path)
    annotations = get_type_hints(type(prototype))
    for name, value in payload.items():
        annotation = annotations.get(name, Any)
        if not _matches_annotation(value, annotation):
            raise ConfigError(f"Invalid value at {path}.{name}: expected {annotation}, got {value!r}")
    try:
        return dataclasses.replace(prototype, **payload)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid value at {path}: {exc}") from exc


def _matches_annotation(value: Any, annotation: Any) -> bool:
    if annotation is Any:
        return True
    origin = get_origin(annotation)
    if origin is Literal:
        return value in get_args(annotation)
    if origin in {types.UnionType, typing.Union}:
        return any(_matches_annotation(value, item) for item in get_args(annotation))
    if annotation is bool:
        return isinstance(value, bool)
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation is float:
        return isinstance(value, int | float) and not isinstance(value, bool)
    if annotation is str:
        return isinstance(value, str)
    if annotation is type(None):
        return value is None
    return True


def _section(section: Any, path: str, fields: set[str], *, nullable: set[str] | None = None) -> dict[str, Any]:
    payload = dict(_mapping(section, path))
    _check_keys(payload, fields, fields, path)
    nullable = nullable or set()
    for field in fields - nullable:
        if payload[field] is None:
            raise ConfigError(f"{path}.{field} cannot be null")
    return payload


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{path} must be a mapping with string keys")
    return value


def _check_keys(payload: Mapping[str, Any], allowed: set[str], required: set[str], path: str) -> None:
    if unknown := set(payload) - allowed:
        raise ConfigError(f"Unknown fields at {path}: {sorted(unknown)}")
    if missing := required - set(payload):
        raise ConfigError(f"Missing required fields at {path}: {sorted(missing)}")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def _nullable_string(value: Any, path: str) -> str | None:
    return None if value is None else _string(value, path)


def _integer(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{path} must be an integer")
    return value


def _nullable_integer(value: Any, path: str) -> int | None:
    return None if value is None else _integer(value, path)


def _number(value: Any, path: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigError(f"{path} must be a number")
    return float(value)


def _nullable_number(value: Any, path: str) -> float | None:
    return None if value is None else _number(value, path)


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be a boolean")
    return value


def _to_serializable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "_type": f"{type(value).__module__}.{type(value).__qualname__}",
            **{field.name: _to_serializable(getattr(value, field.name)) for field in dataclasses.fields(value)},
        }
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_serializable(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return {"_type": f"{type(value).__module__}.{type(value).__qualname__}", "repr": repr(value)}


def _git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}
