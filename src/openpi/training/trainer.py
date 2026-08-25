"""Shared JAX train-state and step functions for fine-tuning and pre-training."""

from __future__ import annotations

import dataclasses
from typing import Any

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import optax

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.training import optimizer as _optimizer
from openpi.training import rlds_mixture
from openpi.training import sharding
from openpi.training import utils as training_utils
from openpi.training import weight_loaders


def _load_weights_and_validate(loader: weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {
            key: value
            for key, value in traverse_util.flatten_dict(loaded_params).items()
            if not isinstance(value, jax.ShapeDtypeStruct)
        }
    )


def init_train_state(config: Any, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool):
    """Initialize a shared TrainState from any config implementing the TrainConfig surface."""
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)
    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)
    return train_state, state_sharding


@at.typecheck
def train_step(
    config: Any,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """The original OpenPI fine-tuning step."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(
        model: _model.BaseModel, key: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        return jnp.mean(model.compute_loss(key, observation, actions, train=True))

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)
    new_state, model = _apply_gradients(config, state, model, grads)
    return new_state, _base_metrics(model, loss, grads)


def pretrain_step(
    config: Any,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: rlds_mixture.PretrainBatch,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Masked pi0.5 step with one optimizer update over accumulated microbatches."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model: _model.BaseModel, key: at.KeyArrayLike):
        loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
        valid_count = jnp.asarray(0.0, dtype=jnp.float32)
        valid_actions = jnp.asarray(0.0, dtype=jnp.float32)
        action_elements = jnp.asarray(0.0, dtype=jnp.float32)
        source_loss_sums = jnp.zeros(len(config.data.sources), dtype=jnp.float32)
        source_valid_counts = jnp.zeros(len(config.data.sources), dtype=jnp.float32)
        source_examples = jnp.zeros(len(config.data.sources), dtype=jnp.float32)
        for micro_index in range(config.gradient_accumulation_steps):
            micro = jax.tree.map(lambda value, index=micro_index: value[index], batch)
            chunked_loss = model.compute_loss(
                jax.random.fold_in(key, micro_index),
                micro.observation,
                micro.actions,
                train=True,
                action_mask=micro.action_mask,
            )
            time_mask = jnp.any(micro.action_mask, axis=-1)
            numeric_mask = time_mask.astype(jnp.float32)
            loss_sum += jnp.sum(chunked_loss * numeric_mask)
            valid_count += jnp.sum(numeric_mask)
            valid_actions += jnp.sum(micro.action_mask)
            action_elements += micro.action_mask.size
            for source_index in range(len(config.data.sources)):
                example_mask = micro.source_id == source_index
                source_mask = numeric_mask * example_mask[:, None]
                source_loss_sums = source_loss_sums.at[source_index].add(jnp.sum(chunked_loss * source_mask))
                source_valid_counts = source_valid_counts.at[source_index].add(jnp.sum(source_mask))
                source_examples = source_examples.at[source_index].add(jnp.sum(example_mask))
        loss = loss_sum / jnp.clip(valid_count, 1)
        aux = (valid_actions / jnp.clip(action_elements, 1), source_loss_sums, source_valid_counts, source_examples)
        return loss, aux

    train_rng = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model, train_rng)
    new_state, model = _apply_gradients(config, state, model, grads)
    metrics = _base_metrics(model, loss, grads)
    valid_action_fraction, source_loss_sums, source_valid_counts, source_examples = aux
    metrics["valid_action_fraction"] = valid_action_fraction
    total_examples = jnp.sum(source_examples)
    for index, source in enumerate(config.data.sources):
        metrics[f"source/{source.id}/loss_sum"] = source_loss_sums[index]
        metrics[f"source/{source.id}/valid_count"] = source_valid_counts[index]
        metrics[f"source/{source.id}/fraction"] = source_examples[index] / jnp.clip(total_examples, 1)
    return new_state, metrics


def validation_step(
    config: Any,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: rlds_mixture.PretrainBatch,
) -> dict[str, at.Array]:
    """No-gradient, augmentation-free validation on current parameters."""
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
    valid_count = jnp.asarray(0.0, dtype=jnp.float32)
    source_loss_sums = jnp.zeros(len(config.data.sources), dtype=jnp.float32)
    source_valid_counts = jnp.zeros(len(config.data.sources), dtype=jnp.float32)
    source_examples = jnp.zeros(len(config.data.sources), dtype=jnp.float32)
    for micro_index in range(config.gradient_accumulation_steps):
        micro = jax.tree.map(lambda value, index=micro_index: value[index], batch)
        chunked_loss = model.compute_loss(
            jax.random.fold_in(rng, micro_index),
            micro.observation,
            micro.actions,
            train=False,
            action_mask=micro.action_mask,
        )
        time_mask = jnp.any(micro.action_mask, axis=-1)
        numeric_mask = time_mask.astype(jnp.float32)
        loss_sum += jnp.sum(chunked_loss * numeric_mask)
        valid_count += jnp.sum(numeric_mask)
        for source_index in range(len(config.data.sources)):
            example_mask = micro.source_id == source_index
            source_mask = numeric_mask * example_mask[:, None]
            source_loss_sums = source_loss_sums.at[source_index].add(jnp.sum(chunked_loss * source_mask))
            source_valid_counts = source_valid_counts.at[source_index].add(jnp.sum(source_mask))
            source_examples = source_examples.at[source_index].add(jnp.sum(example_mask))
    metrics = {"loss": loss_sum / jnp.clip(valid_count, 1)}
    total_examples = jnp.sum(source_examples)
    for index, source in enumerate(config.data.sources):
        metrics[f"source/{source.id}/loss_sum"] = source_loss_sums[index]
        metrics[f"source/{source.id}/valid_count"] = source_valid_counts[index]
        metrics[f"source/{source.id}/fraction"] = source_examples[index] / jnp.clip(total_examples, 1)
    return metrics


def _apply_gradients(config: Any, state: training_utils.TrainState, model: _model.BaseModel, grads: Any):
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )
    return new_state, model


def _base_metrics(model: _model.BaseModel, loss: Any, grads: Any) -> dict[str, at.Array]:
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, value: value.value.ndim > 1,
        ),
    )
    return {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }


def _source_metrics(config: Any, loss: Any, time_mask: Any, source_ids: Any) -> dict[str, at.Array]:
    metrics = {}
    for index, source in enumerate(config.data.sources):
        example_mask = source_ids == index
        source_time_mask = time_mask & example_mask[:, None]
        numeric_mask = source_time_mask.astype(loss.dtype)
        metrics[f"source/{source.id}/loss_sum"] = jnp.sum(loss * numeric_mask)
        metrics[f"source/{source.id}/valid_count"] = jnp.sum(numeric_mask)
        metrics[f"source/{source.id}/fraction"] = jnp.mean(example_mask)
    return metrics


def _masked_mean(value: Any, mask: Any) -> Any:
    mask = jnp.asarray(mask, dtype=value.dtype)
    return jnp.sum(value * mask) / jnp.clip(jnp.sum(mask), 1)
