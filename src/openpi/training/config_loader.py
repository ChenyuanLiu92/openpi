"""Load versioned experiment configs from safe, declarative YAML files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import datetime
import enum
import pathlib
import subprocess
import sys
from typing import Any

import tyro
import yaml

SCHEMA_VERSION = 1
_YAML_SUFFIXES = {".yaml", ".yml"}
_RUNTIME_ONLY_PATHS = {
    ("assets_base_dir",),
    ("checkpoint_base_dir",),
    ("exp_name",),
    ("overwrite",),
    ("resume",),
    ("data", "rlds_data_dir"),
}


class ConfigError(ValueError):
    """Raised when an experiment config is invalid."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,  # noqa: FBT001, FBT002
) -> dict:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigError(f"Duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


@dataclasses.dataclass(frozen=True)
class ResolvedTrainConfig:
    """A TrainConfig together with the information needed to reproduce how it was built."""

    config: Any
    base: str | None
    source: str
    overrides: dict[str, Any] = dataclasses.field(default_factory=dict)
    manifest: dict[str, Any] | None = None
    cli_args: tuple[str, ...] = ()

    def snapshot(self) -> dict[str, Any]:
        snapshot = {
            "schema_version": SCHEMA_VERSION,
            "name": self.config.name,
            "mode": "standalone" if self.base is None else "inherited",
            "source": self.source,
            "cli_args": list(self.cli_args),
            "git": _git_provenance(),
            "resolved_config": _to_serializable(self.config),
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        if self.base is None:
            snapshot["manifest"] = self.manifest
        else:
            snapshot["base"] = self.base
            snapshot["overrides"] = self.overrides
        return snapshot


def is_yaml_ref(config_ref: str | pathlib.Path) -> bool:
    return pathlib.Path(config_ref).suffix.lower() in _YAML_SUFFIXES


def load_yaml_config(path: str | pathlib.Path) -> ResolvedTrainConfig:
    """Load either a standalone YAML config or a legacy base/overrides config."""
    config_path = pathlib.Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config does not exist: {config_path}")

    try:
        contents = yaml.load(config_path.read_text(), Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(contents, dict):
        raise ConfigError(f"Experiment config must contain a YAML mapping: {config_path}")
    if contents.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(f"Unsupported schema_version {contents.get('schema_version')!r}; expected {SCHEMA_VERSION}")
    if "base" in contents or "overrides" in contents:
        return _load_inherited_config(contents, config_path)
    return _load_standalone_config(contents, config_path)


def _load_inherited_config(contents: dict[str, Any], config_path: pathlib.Path) -> ResolvedTrainConfig:
    """Load the original base + overrides YAML format."""
    from openpi.training import config as _config

    allowed_keys = {"schema_version", "name", "base", "overrides"}
    _check_keys(contents, allowed_keys, allowed_keys - {"overrides"}, "config")
    name = _nonempty_string(contents["name"], "name")
    base = _nonempty_string(contents["base"], "base")
    overrides = _mapping(contents.get("overrides", {}), "overrides")

    base_config = _config.get_builtin_config(base)
    merged = _merge_dataclass(base_config, overrides)
    merged = dataclasses.replace(merged, name=name)
    return ResolvedTrainConfig(
        config=merged,
        base=base,
        source=str(config_path.resolve()),
        overrides=dict(overrides),
    )


def _load_standalone_config(contents: dict[str, Any], config_path: pathlib.Path) -> ResolvedTrainConfig:
    """Construct a complete TrainConfig from registered component types."""
    import flax.nnx as nnx

    import openpi.models.pi0_config as pi0_config
    import openpi.models.pi0_fast as pi0_fast
    import openpi.training.config as _config
    import openpi.training.droid_rlds_dataset as droid_rlds_dataset
    import openpi.training.optimizer as optimizer
    import openpi.training.weight_loaders as weight_loaders

    top_level_fields = {
        "schema_version",
        "name",
        "model",
        "data",
        "weight_loader",
        "lr_schedule",
        "optimizer",
        "training",
        "paths",
        "checkpoint",
        "logging",
        "distributed",
        "pytorch",
        "policy_metadata",
    }
    _check_keys(contents, top_level_fields, top_level_fields, "config")
    name = _nonempty_string(contents["name"], "name")

    model = _build_registered_dataclass(
        contents["model"],
        "model",
        {
            "pi0": pi0_config.Pi0Config(),
            "pi0_fast": pi0_fast.Pi0FASTConfig(),
        },
    )
    data = _build_data_config(contents["data"], _config, droid_rlds_dataset)
    weight_loader = _build_registered_dataclass(
        contents["weight_loader"],
        "weight_loader",
        {
            "none": weight_loaders.NoOpWeightLoader(),
            "checkpoint": weight_loaders.CheckpointWeightLoader(params_path=""),
            "paligemma": weight_loaders.PaliGemmaWeightLoader(),
        },
    )
    lr_schedule = _build_registered_dataclass(
        contents["lr_schedule"],
        "lr_schedule",
        {
            "cosine_decay": optimizer.CosineDecaySchedule(),
            "rsqrt_decay": optimizer.RsqrtDecaySchedule(),
        },
    )
    optimizer_config = _build_registered_dataclass(
        contents["optimizer"],
        "optimizer",
        {
            "adamw": optimizer.AdamW(),
            "sgd": optimizer.SGD(),
        },
    )

    training = _validated_section(
        contents["training"],
        "training",
        {
            "seed": 42,
            "batch_size": 32,
            "num_workers": 2,
            "num_train_steps": 30_000,
            "ema_decay": 0.99,
            "freeze_mode": "full",
        },
        nullable={"ema_decay"},
    )
    paths = _validated_section(
        contents["paths"],
        "paths",
        {"assets_base_dir": "./assets", "checkpoint_base_dir": "./checkpoints"},
    )
    checkpoint = _validated_section(
        contents["checkpoint"],
        "checkpoint",
        {
            "exp_name": "",
            "save_interval": 1000,
            "keep_period": 5000,
            "overwrite": False,
            "resume": False,
        },
        nullable={"keep_period"},
    )
    logging_config = _validated_section(
        contents["logging"],
        "logging",
        {"project_name": "openpi", "wandb_enabled": True, "log_interval": 100},
    )
    distributed = _validated_section(
        contents["distributed"],
        "distributed",
        {"fsdp_devices": 1},
    )
    pytorch = _validated_section(
        contents["pytorch"],
        "pytorch",
        {"weight_path": "", "training_precision": "bfloat16"},
        nullable={"weight_path"},
    )

    freeze_mode = training.pop("freeze_mode")
    if freeze_mode == "full":
        freeze_filter = nnx.Nothing()
    elif freeze_mode == "lora":
        variants = (
            getattr(model, "paligemma_variant", ""),
            getattr(model, "action_expert_variant", ""),
        )
        if not any("lora" in variant for variant in variants):
            raise ConfigError("training.freeze_mode=lora requires a model variant containing 'lora'")
        freeze_filter = model.get_freeze_filter()
    else:
        raise ConfigError("training.freeze_mode must be 'full' or 'lora'")

    if pytorch["training_precision"] not in {"bfloat16", "float32"}:
        raise ConfigError("pytorch.training_precision must be 'bfloat16' or 'float32'")
    if checkpoint["overwrite"] and checkpoint["resume"]:
        raise ConfigError("checkpoint.overwrite and checkpoint.resume cannot both be true")

    policy_metadata = contents["policy_metadata"]
    if policy_metadata is not None and not isinstance(policy_metadata, dict):
        raise ConfigError("policy_metadata must be a mapping or null")

    train_config = _config.TrainConfig(
        name=name,
        project_name=logging_config["project_name"],
        exp_name=checkpoint["exp_name"],
        model=model,
        weight_loader=weight_loader,
        pytorch_weight_path=pytorch["weight_path"],
        pytorch_training_precision=pytorch["training_precision"],
        lr_schedule=lr_schedule,
        optimizer=optimizer_config,
        ema_decay=training["ema_decay"],
        freeze_filter=freeze_filter,
        data=data,
        assets_base_dir=paths["assets_base_dir"],
        checkpoint_base_dir=paths["checkpoint_base_dir"],
        seed=training["seed"],
        batch_size=training["batch_size"],
        num_workers=training["num_workers"],
        num_train_steps=training["num_train_steps"],
        log_interval=logging_config["log_interval"],
        save_interval=checkpoint["save_interval"],
        keep_period=checkpoint["keep_period"],
        overwrite=checkpoint["overwrite"],
        resume=checkpoint["resume"],
        wandb_enabled=logging_config["wandb_enabled"],
        policy_metadata=policy_metadata,
        fsdp_devices=distributed["fsdp_devices"],
    )
    return ResolvedTrainConfig(
        config=train_config,
        base=None,
        source=str(config_path.resolve()),
        manifest=contents,
    )


def _build_data_config(section: Any, config_module: Any, droid_module: Any) -> Any:
    payload = dict(_mapping(section, "data"))
    type_name = payload.pop("type", None)
    if not isinstance(type_name, str):
        raise ConfigError("data.type must be a string")

    required_fields = {
        "fake": {"repo_id", "assets", "base_config"},
        "lerobot_libero": {"repo_id", "assets", "base_config", "extra_delta_transform"},
        "lerobot_aloha": {
            "repo_id",
            "assets",
            "base_config",
            "use_delta_joint_actions",
            "default_prompt",
            "adapt_to_pi",
            "action_sequence_keys",
        },
        "lerobot_droid": {"repo_id", "assets", "base_config"},
        "rlds_droid": {"repo_id", "assets", "base_config", "rlds_data_dir", "action_space", "datasets"},
    }
    if type_name not in required_fields:
        raise ConfigError(f"Unknown data.type {type_name!r}; expected one of {sorted(required_fields)}")
    if missing := required_fields[type_name] - set(payload):
        raise ConfigError(f"Missing required fields at data: {sorted(missing)}")

    assets_payload = _mapping(payload["assets"], "data.assets")
    _check_keys(assets_payload, {"assets_dir", "asset_id"}, {"assets_dir", "asset_id"}, "data.assets")

    base_config_value = payload.get("base_config")
    if base_config_value is not None:
        base_payload = _mapping(base_config_value, "data.base_config")
        allowed_base_fields = {"prompt_from_task", "action_sequence_keys"}
        _check_keys(base_payload, allowed_base_fields, allowed_base_fields, "data.base_config")
        action_keys = base_payload.get("action_sequence_keys", ["actions"])
        if not isinstance(action_keys, list) or not all(isinstance(key, str) for key in action_keys):
            raise ConfigError("data.base_config.action_sequence_keys must be a list of strings")
        payload["base_config"] = {
            "prompt_from_task": base_payload["prompt_from_task"],
            "action_sequence_keys": action_keys,
        }

    registry = {
        "fake": config_module.FakeDataConfig(base_config=config_module.DataConfig()),
        "lerobot_libero": config_module.LeRobotLiberoDataConfig(repo_id="", base_config=config_module.DataConfig()),
        "lerobot_aloha": config_module.LeRobotAlohaDataConfig(repo_id="", base_config=config_module.DataConfig()),
        "lerobot_droid": config_module.LeRobotDROIDDataConfig(repo_id="", base_config=config_module.DataConfig()),
        "rlds_droid": config_module.RLDSDroidDataConfig(repo_id="", base_config=config_module.DataConfig()),
    }
    if type_name == "rlds_droid":
        action_space = payload.get("action_space")
        action_spaces = {
            "joint_position": droid_module.DroidActionSpace.JOINT_POSITION,
            "joint_velocity": droid_module.DroidActionSpace.JOINT_VELOCITY,
        }
        if action_space not in action_spaces:
            raise ConfigError(f"data.action_space must be one of {sorted(action_spaces)}")
        payload["action_space"] = action_spaces[action_space]
        datasets = payload.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            raise ConfigError("data.datasets must be a non-empty list")
        parsed_datasets = []
        dataset_fields = {"name", "version", "weight", "filter_dict_path"}
        for index, dataset in enumerate(datasets):
            dataset_payload = _mapping(dataset, f"data.datasets[{index}]")
            _check_keys(
                dataset_payload, dataset_fields, dataset_fields - {"filter_dict_path"}, f"data.datasets[{index}]"
            )
            parsed_datasets.append(droid_module.RLDSDataset(**dataset_payload))
        payload["datasets"] = tuple(parsed_datasets)

    return _merge_dataclass(registry[type_name], payload, ("data",), enforce_runtime_only=False)


def _build_registered_dataclass(section: Any, path: str, registry: Mapping[str, Any]) -> Any:
    payload = dict(_mapping(section, path))
    type_name = payload.pop("type", None)
    if not isinstance(type_name, str):
        raise ConfigError(f"{path}.type must be a string")
    if type_name not in registry:
        raise ConfigError(f"Unknown {path}.type {type_name!r}; expected one of {sorted(registry)}")
    prototype = registry[type_name]
    required_fields = {field.name for field in dataclasses.fields(prototype) if field.init}
    if missing := required_fields - set(payload):
        raise ConfigError(f"Missing required fields at {path}: {sorted(missing)}")
    return _merge_dataclass(prototype, payload, (path,), enforce_runtime_only=False)


def _validated_section(
    section: Any,
    path: str,
    defaults: Mapping[str, Any],
    *,
    nullable: set[str] | None = None,
) -> dict[str, Any]:
    payload = _mapping(section, path)
    _check_keys(payload, set(defaults), set(defaults), path)
    nullable = nullable or set()
    result = {}
    for name, default in defaults.items():
        value = payload[name]
        if value is None and name in nullable:
            result[name] = None
        else:
            result[name] = _coerce_value(default, value, (path, name))
    return result


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{path} keys must be strings")
    return value


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def _check_keys(payload: Mapping[str, Any], allowed: set[str], required: set[str], path: str) -> None:
    if unknown := set(payload) - allowed:
        raise ConfigError(f"Unknown fields at {path}: {sorted(unknown)}")
    if missing := required - set(payload):
        raise ConfigError(f"Missing required fields at {path}: {sorted(missing)}")


def resolve_config(config_ref: str | pathlib.Path) -> ResolvedTrainConfig:
    """Resolve either a built-in config name or a YAML config path."""
    from openpi.training import config as _config

    ref = str(config_ref)
    if is_yaml_ref(ref):
        return load_yaml_config(ref)
    config = _config.get_builtin_config(ref)
    return ResolvedTrainConfig(config=config, base=ref, source=f"builtin:{ref}")


def parse_cli(args: Sequence[str] | None = None) -> ResolvedTrainConfig:
    """Parse a built-in config subcommand or a YAML path, preserving Tyro overrides."""
    from openpi.training import config as _config

    cli_args = list(sys.argv[1:] if args is None else args)
    if not cli_args:
        # Preserve Tyro's normal help/error behavior for the existing CLI.
        config = tyro.extras.overridable_config_cli(
            {k: (k, v) for k, v in _config.builtin_configs().items()}, args=cli_args
        )
        return ResolvedTrainConfig(config=config, base=config.name, source=f"builtin:{config.name}")

    config_ref = cli_args[0]
    if is_yaml_ref(config_ref):
        resolved = load_yaml_config(config_ref)
        config = tyro.cli(
            _config.TrainConfig,
            default=resolved.config,
            args=cli_args[1:],
        )
        return dataclasses.replace(resolved, config=config, cli_args=tuple(cli_args[1:]))

    config = tyro.extras.overridable_config_cli(
        {k: (k, v) for k, v in _config.builtin_configs().items()}, args=cli_args
    )
    return ResolvedTrainConfig(
        config=config,
        base=config_ref,
        source=f"builtin:{config_ref}",
        cli_args=tuple(cli_args[1:]),
    )


def snapshot_for_config(config: Any) -> dict[str, Any]:
    """Create provenance for configs constructed programmatically (primarily tests)."""
    return ResolvedTrainConfig(
        config=config,
        base=config.name,
        source=f"programmatic:{config.name}",
    ).snapshot()


def write_snapshot(path: str | pathlib.Path, snapshot: Mapping[str, Any]) -> None:
    output_path = pathlib.Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(dict(snapshot), sort_keys=False, allow_unicode=True))


def _merge_dataclass(
    current: Any,
    overrides: Mapping[str, Any],
    path: tuple[str, ...] = (),
    *,
    enforce_runtime_only: bool = True,
) -> Any:
    if not dataclasses.is_dataclass(current) or isinstance(current, type):
        raise ConfigError(f"Cannot apply nested overrides to {_format_path(path)}")
    if not isinstance(overrides, Mapping):
        raise ConfigError(f"Expected a mapping at {_format_path(path)}")

    fields = {field.name: field for field in dataclasses.fields(current) if field.init}
    if unknown_fields := set(overrides) - set(fields):
        raise ConfigError(f"Unknown fields at {_format_path(path)}: {sorted(unknown_fields)}")

    updates = {}
    for name, override in overrides.items():
        field_path = (*path, name)
        if enforce_runtime_only and field_path in _RUNTIME_ONLY_PATHS:
            raise ConfigError(f"{_format_path(field_path)} is runtime-only; provide it as a command-line override")
        current_value = getattr(current, name)
        if dataclasses.is_dataclass(current_value) and not isinstance(current_value, type):
            if not isinstance(override, Mapping):
                raise ConfigError(f"Expected a mapping at {_format_path(field_path)}")
            updates[name] = _merge_dataclass(
                current_value,
                override,
                field_path,
                enforce_runtime_only=enforce_runtime_only,
            )
        else:
            updates[name] = _coerce_value(current_value, override, field_path)
    try:
        return dataclasses.replace(current, **updates)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid value at {_format_path(path)}: {exc}") from exc


def _coerce_value(current: Any, value: Any, path: tuple[str, ...]) -> Any:
    if isinstance(current, enum.Enum):
        try:
            return type(current)(value)
        except ValueError as exc:
            raise ConfigError(f"Invalid enum value at {_format_path(path)}: {value!r}") from exc
    if isinstance(current, tuple):
        if not isinstance(value, list | tuple):
            raise ConfigError(f"Expected a sequence at {_format_path(path)}")
        return tuple(value)
    if isinstance(current, list):
        if not isinstance(value, list):
            raise ConfigError(f"Expected a list at {_format_path(path)}")
        return value
    if isinstance(current, dict):
        if not isinstance(value, dict):
            raise ConfigError(f"Expected a mapping at {_format_path(path)}")
        return value
    if current is None:
        return value
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"Expected a boolean at {_format_path(path)}")
        return value
    if isinstance(current, int) and not isinstance(current, bool):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"Expected an integer at {_format_path(path)}")
        return value
    if isinstance(current, float):
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ConfigError(f"Expected a number at {_format_path(path)}")
        return float(value)
    if isinstance(current, str):
        if not isinstance(value, str):
            raise ConfigError(f"Expected a string at {_format_path(path)}")
        return value
    raise ConfigError(
        f"Field {_format_path(path)} has non-declarative type {type(current).__name__}; choose a compatible base config"
    )


def _format_path(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "overrides"


def _qualified_name(value: Any) -> str:
    target = value if isinstance(value, type) else type(value)
    return f"{target.__module__}.{target.__qualname__}"


def _to_serializable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result = {"_type": _qualified_name(value)}
        for field in dataclasses.fields(value):
            result[field.name] = _to_serializable(getattr(value, field.name))
        return result
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
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        qualname = getattr(value, "__qualname__", type(value).__qualname__)
        return {"_callable": f"{module}.{qualname}"}
    return {"_type": _qualified_name(value), "repr": repr(value)}


def _git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    tracked_status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit,
        "dirty": bool(tracked_status) if tracked_status is not None else None,
    }
