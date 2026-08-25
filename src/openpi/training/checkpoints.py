from __future__ import annotations

import asyncio
from collections.abc import Mapping
import concurrent.futures as futures
import dataclasses
import json
import logging
import shutil
from typing import Any, Protocol

from etils import epath
import jax
from jax.experimental import multihost_utils
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.config_loader as config_loader
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    existed = checkpoint_dir.exists()
    if existed and not overwrite and not resume:
        raise FileExistsError(
            f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
            "to indicate how to handle it."
        )
    resuming = existed and resume and not overwrite
    if jax.process_index() == 0:
        if existed and overwrite:
            checkpoint_dir.rmtree()
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if jax.process_count() > 1:
        multihost_utils.sync_global_devices("checkpoint_directory_initialized")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "metadata": CallbackHandler(),
            "data_iterator": CallbackHandler(all_processes=True),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader | None,
    step: int,
    *,
    config_snapshot: dict | None = None,
    extra_assets: Mapping[str, dict[str, _normalize.NormStats]] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    iterator_snapshot_dir: epath.Path | str | None = None,
):
    def save_assets(directory: epath.Path):
        if data_loader is not None:
            # Save the normalization stats used by the fine-tuning loader.
            data_config = data_loader.data_config()
            norm_stats = data_config.norm_stats
            if norm_stats is not None and data_config.asset_id is not None:
                _normalize.save(directory / data_config.asset_id, norm_stats)
        for asset_id, norm_stats in (extra_assets or {}).items():
            _normalize.save(directory / asset_id, norm_stats)

    def save_metadata(directory: epath.Path):
        if config_snapshot is not None:
            config_loader.write_snapshot(directory / "train_config.yaml", config_snapshot)
        if extra_metadata is not None:
            (directory / "data_state.json").write_text(
                json.dumps(dict(extra_metadata), indent=2, sort_keys=True) + "\n"
            )

    def save_iterator(directory: epath.Path):
        assert iterator_snapshot_dir is not None
        source = epath.Path(iterator_snapshot_dir)
        target = directory / f"rank-{jax.process_index():05d}"
        target.mkdir(parents=True, exist_ok=False)
        for path in source.iterdir():
            if path.is_dir():
                shutil.copytree(str(path), str(target / path.name))
            else:
                shutil.copy2(str(path), str(target / path.name))

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "metadata": save_metadata,
        "train_state": train_state,
        "params": {"params": params},
    }
    if iterator_snapshot_dir is not None:
        items["data_iterator"] = save_iterator
    checkpoint_manager.save(step, items)


def load_extra_metadata(checkpoint_dir: epath.Path | str, step: int | None = None) -> dict[str, Any] | None:
    """Loads optional data-stream metadata saved alongside a checkpoint."""
    checkpoint_dir = epath.Path(checkpoint_dir)
    if step is None:
        steps = sorted(int(path.name) for path in checkpoint_dir.iterdir() if path.name.isdigit())
        if not steps:
            return None
        step = steps[-1]
    path = checkpoint_dir / str(step) / "metadata" / "data_state.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader | None,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def __init__(self, *, all_processes: bool = False):
        self._all_processes = all_processes

    def save(self, directory: epath.Path, args: CallbackSave):
        if self._all_processes or jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
