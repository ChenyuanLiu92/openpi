"""Load versioned experiment configs from YAML files.

YAML configs intentionally extend an existing Python config instead of constructing arbitrary
Python objects. This keeps model and transform registration in Python while allowing experiments
to be reviewed and versioned as data.
"""

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
    base: str
    source: str
    overrides: dict[str, Any] = dataclasses.field(default_factory=dict)
    cli_args: tuple[str, ...] = ()

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.config.name,
            "base": self.base,
            "source": self.source,
            "overrides": self.overrides,
            "cli_args": list(self.cli_args),
            "git": _git_provenance(),
            "resolved_config": _to_serializable(self.config),
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }


def is_yaml_ref(config_ref: str | pathlib.Path) -> bool:
    return pathlib.Path(config_ref).suffix.lower() in _YAML_SUFFIXES


def load_yaml_config(path: str | pathlib.Path) -> ResolvedTrainConfig:
    """Load a YAML experiment config and merge it into its built-in base config."""
    from openpi.training import config as _config

    config_path = pathlib.Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config does not exist: {config_path}")

    try:
        contents = yaml.load(config_path.read_text(), Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(contents, dict):
        raise ConfigError(f"Experiment config must contain a YAML mapping: {config_path}")
    allowed_keys = {"schema_version", "name", "base", "overrides"}
    if unknown_keys := set(contents) - allowed_keys:
        raise ConfigError(f"Unknown top-level fields in {config_path}: {sorted(unknown_keys)}")

    if contents.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(
            f"Unsupported schema_version {contents.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    name = contents.get("name")
    base = contents.get("base")
    overrides = contents.get("overrides", {})
    if not isinstance(name, str) or not name:
        raise ConfigError("'name' must be a non-empty string")
    if not isinstance(base, str) or not base:
        raise ConfigError("'base' must be a non-empty string")
    if not isinstance(overrides, dict):
        raise ConfigError("'overrides' must be a mapping")

    base_config = _config.get_builtin_config(base)
    merged = _merge_dataclass(base_config, overrides)
    merged = dataclasses.replace(merged, name=name)
    return ResolvedTrainConfig(
        config=merged,
        base=base,
        source=str(config_path.resolve()),
        overrides=overrides,
    )


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


def _merge_dataclass(current: Any, overrides: Mapping[str, Any], path: tuple[str, ...] = ()) -> Any:
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
        if field_path in _RUNTIME_ONLY_PATHS:
            raise ConfigError(
                f"{_format_path(field_path)} is runtime-only; provide it as a command-line override"
            )
        current_value = getattr(current, name)
        if dataclasses.is_dataclass(current_value) and not isinstance(current_value, type):
            if not isinstance(override, Mapping):
                raise ConfigError(f"Expected a mapping at {_format_path(field_path)}")
            updates[name] = _merge_dataclass(current_value, override, field_path)
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
