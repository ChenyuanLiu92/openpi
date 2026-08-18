"""Compute mixture-aware pi0.5 pretraining normalization statistics."""

from __future__ import annotations

import logging
import math

import jax
import numpy as np
import tyro

from openpi.shared import normalize
from openpi.training import pretrain_config
from openpi.training import pretrain_config_loader
from openpi.training import rlds_mixture


def _allocate_samples(probabilities: list[float], total: int) -> list[int]:
    """Largest-remainder allocation with enough examples to compute variance."""
    if total < 2 * len(probabilities):
        raise ValueError(f"Sample budget {total} must be at least {2 * len(probabilities)}")
    normalized = np.asarray(probabilities, dtype=np.float64)
    normalized /= normalized.sum()
    remaining = total - 2 * len(probabilities)
    raw = normalized * remaining
    counts = np.floor(raw).astype(np.int64) + 2
    for index in np.argsort(-(raw - np.floor(raw)))[: total - int(counts.sum())]:
        counts[index] += 1
    return counts.tolist()


def _group_sources(
    config: pretrain_config.PretrainConfig,
) -> dict[str, list[tuple[pretrain_config.RldsSourceConfig, float]]]:
    groups: dict[str, list[tuple[pretrain_config.RldsSourceConfig, float]]] = {}
    for source, probability in zip(config.data.sources, config.data.effective_probabilities(), strict=True):
        groups.setdefault(source.normalization_id, []).append((source, probability))
    return groups


def main(
    config_path: str,
    *,
    max_frames_per_normalization: int = 1_000_000,
    batch_size: int = 1024,
) -> None:
    """Compute q01/q99, mean, and std from each normalization group.

    Source sample counts within a shared normalization group follow the same
    temperature-adjusted probabilities as training.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if jax.process_count() != 1:
        raise RuntimeError("Normalization statistics must be computed by one JAX process")
    if max_frames_per_normalization <= 0 or batch_size <= 0:
        raise ValueError("Frame budget and batch size must be positive")

    config = pretrain_config_loader.load(config_path).config
    for normalization_id, weighted_sources in _group_sources(config).items():
        sources = [source for source, _ in weighted_sources]
        probabilities = [probability for _, probability in weighted_sources]
        targets = _allocate_samples(probabilities, max_frames_per_normalization)
        state_stats = normalize.RunningStats()
        action_stats = normalize.RunningStats()
        total_frames = 0

        logging.info(
            "Computing %s from %s",
            normalization_id,
            ", ".join(f"{source.id}={target}" for source, target in zip(sources, targets, strict=True)),
        )
        for source, target in zip(sources, targets, strict=True):
            dataset = rlds_mixture.create_source_frames(
                config,
                source,
                split=source.train_split,
            )
            dataset = dataset.take(target).batch(batch_size, drop_remainder=False)
            source_frames = 0
            for batch in dataset.as_numpy_iterator():
                states = np.asarray(batch["state"], dtype=np.float32)
                actions = np.asarray(batch["actions"], dtype=np.float32)
                if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
                    raise ValueError(f"Source {source.id!r} contains NaN or infinite state/action values")
                state_stats.update(states)
                action_stats.update(actions)
                source_frames += len(states)
            if source_frames < target:
                logging.warning(
                    "Source %s provided %d frames, below its requested budget of %d",
                    source.id,
                    source_frames,
                    target,
                )
            if source_frames < 2 and len(sources) == 1:
                raise ValueError(f"Source {source.id!r} must contain at least two training frames")
            total_frames += source_frames

        if total_frames < 2 or not math.isfinite(total_frames):
            raise ValueError(f"Normalization group {normalization_id!r} has insufficient data")
        output = rlds_mixture.save_stats(
            config,
            normalization_id,
            {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()},
            sample_count=total_frames,
        )
        logging.info("Saved %d-frame normalization statistics to %s", total_frames, output)


if __name__ == "__main__":
    tyro.cli(main)
