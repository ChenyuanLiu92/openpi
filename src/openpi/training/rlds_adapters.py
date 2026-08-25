"""Runtime adapters from heterogeneous RLDS trajectories to OpenPI's canonical schema."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
from typing import Any, Protocol

from openpi.training import pretrain_config

CANONICAL_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


class AdapterError(ValueError):
    """Raised when an RLDS adapter configuration or output is invalid."""


class RldsAdapter(Protocol):
    """TensorFlow-graph-compatible trajectory adapter."""

    def adapt_trajectory(self, trajectory: Mapping[str, Any]) -> dict[str, Any]: ...


AdapterFactory = Callable[[pretrain_config.RldsSourceConfig], RldsAdapter]
_REGISTRY: dict[str, AdapterFactory] = {}


def register_adapter(name: str) -> Callable[[AdapterFactory], AdapterFactory]:
    """Register a trusted adapter factory under a stable YAML name."""

    if not name:
        raise ValueError("Adapter name must be non-empty")

    def decorator(factory: AdapterFactory) -> AdapterFactory:
        if name in _REGISTRY:
            raise ValueError(f"RLDS adapter {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory

    return decorator


def create_adapter(source: pretrain_config.RldsSourceConfig) -> RldsAdapter:
    try:
        factory = _REGISTRY[source.adapter.type]
    except KeyError as exc:
        raise AdapterError(
            f"Unknown adapter {source.adapter.type!r} for source {source.id!r}; expected one of {sorted(_REGISTRY)}"
        ) from exc
    return factory(source)


def registered_adapters() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def _get_path(tree: Mapping[str, Any], path: str) -> Any:
    """Resolve slash-separated paths while preserving literal keys that contain slashes."""
    if path in tree:
        return tree[path]
    value: Any = tree
    traversed: list[str] = []
    for component in path.split("/"):
        traversed.append(component)
        if not isinstance(value, Mapping) or component not in value:
            raise AdapterError(f"RLDS field {'/'.join(traversed)!r} was not found while resolving {path!r}")
        value = value[component]
    return value


def _string_mapping(value: Any, path: str) -> dict[str, str | None]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise AdapterError(f"{path} must be a mapping")
    result: dict[str, str | None] = {}
    for key, item in value.items():
        if item is not None and not isinstance(item, str):
            raise AdapterError(f"{path}.{key} must be a string or null")
        result[key] = item
    return result


def _string_sequence(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str) or not value:
        raise AdapterError(f"{path} must be a non-empty list of field paths")
    if not all(isinstance(item, str) and item for item in value):
        raise AdapterError(f"{path} must contain non-empty strings")
    return tuple(value)


@dataclasses.dataclass(frozen=True)
class FieldMapAdapter:
    """Declarative field selection/concatenation for time-major RLDS trajectories."""

    image_paths: Mapping[str, str | None]
    state_paths: tuple[str, ...]
    action_paths: tuple[str, ...]
    prompt_path: str
    image_range: str

    @classmethod
    def from_source(cls, source: pretrain_config.RldsSourceConfig) -> FieldMapAdapter:
        options = source.adapter.options
        allowed = {"images", "state", "actions", "prompt", "image_range"}
        if unknown := set(options) - allowed:
            raise AdapterError(f"Unknown field_map options for source {source.id!r}: {sorted(unknown)}")
        if missing := allowed - set(options):
            raise AdapterError(f"Missing field_map options for source {source.id!r}: {sorted(missing)}")
        images = _string_mapping(options["images"], f"source {source.id}.adapter.options.images")
        if set(images) != set(CANONICAL_IMAGE_KEYS):
            raise AdapterError(f"source {source.id}.adapter.options.images must define exactly {CANONICAL_IMAGE_KEYS}")
        if images["base_0_rgb"] is None:
            raise AdapterError(f"source {source.id} requires a base_0_rgb image")
        prompt = options["prompt"]
        if not isinstance(prompt, str) or not prompt:
            raise AdapterError(f"source {source.id}.adapter.options.prompt must be a non-empty field path")
        image_range = options["image_range"]
        if image_range not in {"uint8", "zero_one", "minus_one_one"}:
            raise AdapterError(
                f"source {source.id}.adapter.options.image_range must be uint8, zero_one, or minus_one_one"
            )
        return cls(
            image_paths=images,
            state_paths=_string_sequence(options["state"], f"source {source.id}.adapter.options.state"),
            action_paths=_string_sequence(options["actions"], f"source {source.id}.adapter.options.actions"),
            prompt_path=prompt,
            image_range=image_range,
        )

    def adapt_trajectory(self, trajectory: Mapping[str, Any]) -> dict[str, Any]:
        # TensorFlow remains optional for fine-tuning users and is imported only on this code path.
        import tensorflow as tf

        def concatenate(paths: tuple[str, ...]) -> Any:
            values = []
            for path in paths:
                value = tf.convert_to_tensor(_get_path(trajectory, path))
                if value.shape.rank == 1:
                    value = value[..., None]
                if value.shape.rank != 2:
                    raise AdapterError(f"Field {path!r} must have time-major shape [T, D], got {value.shape}")
                values.append(tf.cast(value, tf.float32))
            return tf.concat(values, axis=-1)

        state = concatenate(self.state_paths)
        actions = concatenate(self.action_paths)
        length = tf.shape(state)[0]
        tf.debugging.assert_equal(tf.shape(actions)[0], length, message="state/action trajectory length mismatch")

        base_path = self.image_paths["base_0_rgb"]
        assert base_path is not None
        base_image = self._convert_images(_get_path(trajectory, base_path))
        tf.debugging.assert_equal(tf.shape(base_image)[0], length, message="image/state trajectory length mismatch")
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": tf.ones([length], dtype=tf.bool)}
        for key in CANONICAL_IMAGE_KEYS[1:]:
            source_path = self.image_paths[key]
            if source_path is None:
                images[key] = tf.zeros_like(base_image)
                image_masks[key] = tf.zeros([length], dtype=tf.bool)
            else:
                image = self._convert_images(_get_path(trajectory, source_path))
                tf.debugging.assert_equal(tf.shape(image)[0], length, message=f"{key}/state length mismatch")
                images[key] = image
                image_masks[key] = tf.ones([length], dtype=tf.bool)

        prompt = tf.convert_to_tensor(_get_path(trajectory, self.prompt_path))
        if prompt.dtype != tf.string:
            prompt = tf.strings.as_string(prompt)
        if prompt.shape.rank == 0:
            prompt = tf.repeat(prompt[None], length)
        elif prompt.shape.rank != 1:
            raise AdapterError(f"Prompt field {self.prompt_path!r} must be scalar or [T], got {prompt.shape}")
        tf.debugging.assert_equal(tf.shape(prompt)[0], length, message="prompt/state trajectory length mismatch")

        return {
            "image": images,
            "image_mask": image_masks,
            "state": state,
            "actions": actions,
            "prompt": prompt,
        }

    def _convert_images(self, value: Any) -> Any:
        import tensorflow as tf

        image = tf.convert_to_tensor(value)
        if image.dtype == tf.string:
            image = tf.map_fn(
                lambda encoded: tf.io.decode_image(encoded, channels=3, expand_animations=False),
                image,
                fn_output_signature=tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
            )
        if image.shape.rank != 4:
            raise AdapterError(f"Image field must have time-major shape [T,H,W,C] or [T,C,H,W], got {image.shape}")
        if image.shape[-1] == 3:
            pass
        elif image.shape[1] == 3:
            image = tf.transpose(image, [0, 2, 3, 1])
        else:
            raise AdapterError(f"Image field must have exactly three channels, got {image.shape}")
        image = tf.cast(image, tf.float32)
        if self.image_range == "uint8":
            image = image / 127.5 - 1.0
        elif self.image_range == "zero_one":
            image = image * 2.0 - 1.0
        image = tf.image.resize_with_pad(image, 224, 224, antialias=True)
        return tf.clip_by_value(image, -1.0, 1.0)


@register_adapter("field_map")
def _create_field_map(source: pretrain_config.RldsSourceConfig) -> RldsAdapter:
    return FieldMapAdapter.from_source(source)
