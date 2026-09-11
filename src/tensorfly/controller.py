"""Deterministic neural-to-runtime control primitives.

This module deliberately does not create a connectome or load Qwen.  A caller
must provide the selected MaleCNS *internal* indices and a benchmark callable.
The metric-to-neuron mapping and the dynamics-to-configuration mapping below
are engineered experimental mappings, not reconstructions of fly physiology.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np


METRIC_ORDER = (
    "ttft_ms",
    "decode_ms_per_token",
    "throughput_tps",
    "peak_allocated_vram_mb",
    "peak_reserved_vram_mb",
)


def _metric(metrics: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if name in metrics and metrics[name] is not None:
            try:
                value = float(metrics[name])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return max(0.0, value)
    return 0.0


@dataclass(frozen=True)
class SensoryEncoding:
    """One deterministic encoding and its provenance."""

    drive: np.ndarray
    normalized_metrics: dict[str, float]
    selected_indices: tuple[int, ...]
    metric_order: tuple[str, ...] = METRIC_ORDER

    # Array-like conveniences keep the public result pleasant for callers
    # that only need the drive while preserving normalization provenance for
    # trial logs.
    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.drive, dtype=dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.drive.shape

    def __len__(self) -> int:
        return len(self.drive)

    def __getitem__(self, key: Any) -> Any:
        return self.drive[key]


class SensoryEncoder:
    """Encode measured inference metrics onto selected real sensory neurons.

    Each selected neuron receives one normalized metric, cycling through the
    fixed metric order.  This is intentionally simple, deterministic, and
    explicit: it is an engineered input encoding, not a biological claim.
    """

    # Stable scales keep the encoding comparable across trials without fitting
    # on a trial (which would make replay and cross-strategy comparisons drift).
    SCALES = {
        "ttft_ms": 1000.0,
        "decode_ms_per_token": 100.0,
        "throughput_tps": 100.0,
        "peak_allocated_vram_mb": 24_576.0,
        "peak_reserved_vram_mb": 24_576.0,
    }

    def __init__(
        self,
        sensory_indices: Sequence[int] | None,
        *,
        num_neurons: int | None = None,
        metric_order: Sequence[str] = METRIC_ORDER,
        synthetic_dev: bool = False,
    ) -> None:
        if not sensory_indices:
            if not synthetic_dev:
                raise ValueError(
                    "real MaleCNS sensory body/internal indices are required; "
                    "synthetic-dev must be explicit"
                )
            # Even developer mode does not invent a population.  An empty
            # vector is useful for isolated plumbing tests only.
            sensory_indices = ()
        selected = tuple(int(i) for i in sensory_indices)
        if any(i < 0 for i in selected):
            raise ValueError("sensory internal indices must be non-negative")
        if len(set(selected)) != len(selected):
            raise ValueError("sensory internal indices must be unique")
        if num_neurons is None:
            num_neurons = (max(selected) + 1) if selected else 0
        if int(num_neurons) < (max(selected) + 1 if selected else 0):
            raise ValueError("num_neurons is smaller than a sensory index")
        order = tuple(str(name) for name in metric_order)
        if not order:
            raise ValueError("metric_order must not be empty")
        self.selected_indices = selected
        self.num_neurons = int(num_neurons)
        self.metric_order = order
        self.synthetic_dev = bool(synthetic_dev)

    @classmethod
    def normalize_metrics(cls, metrics: Mapping[str, Any] | None) -> dict[str, float]:
        values = metrics or {}
        aliases = {
            "ttft_ms": ("ttft_ms", "time_to_first_token_ms", "prefill_latency_ms"),
            "decode_ms_per_token": ("decode_ms_per_token", "tpot_ms", "decode_latency_ms_per_token"),
            "throughput_tps": ("throughput_tps", "tokens_per_second", "throughput"),
            "peak_allocated_vram_mb": ("peak_allocated_vram_mb", "peak_allocated_mb", "peak_allocated_bytes"),
            "peak_reserved_vram_mb": ("peak_reserved_vram_mb", "peak_reserved_mb", "peak_reserved_bytes"),
        }
        out: dict[str, float] = {}
        for name in METRIC_ORDER:
            raw = _metric(values, *aliases[name])
            byte_key = {
                "peak_allocated_vram_mb": "peak_allocated_bytes",
                "peak_reserved_vram_mb": "peak_reserved_bytes",
            }.get(name)
            if byte_key and byte_key in values:
                # QwenInference records CUDA memory in bytes; the fixed
                # encoding scale is expressed in MB for readability.
                raw = _metric(values, byte_key) / (1024.0 * 1024.0)
            out[name] = float(np.clip(raw / cls.SCALES[name], 0.0, 1.0))
        return out

    def encode(self, metrics: Mapping[str, Any] | None = None) -> SensoryEncoding:
        normalized = self.normalize_metrics(metrics)
        drive = np.zeros(self.num_neurons, dtype=np.float32)
        for ordinal, index in enumerate(self.selected_indices):
            name = self.metric_order[ordinal % len(self.metric_order)]
            # Custom names use the normalized canonical key when possible;
            # unknown keys remain zero rather than becoming random input.
            drive[index] = np.float32(normalized.get(name, 0.0))
        return SensoryEncoding(
            drive=drive,
            normalized_metrics=normalized,
            selected_indices=self.selected_indices,
            metric_order=self.metric_order,
        )


@dataclass(frozen=True)
class NeuralReadout:
    value: float
    selected_indices: tuple[int, ...]
    mean_activity: float
    max_activity: float


def read_population_activity(
    activity: Any,
    indices: Sequence[int],
    *,
    synthetic_dev: bool = False,
) -> NeuralReadout:
    """Read actual activity values at a documented real population."""

    selected = tuple(int(i) for i in indices)
    if not selected and not synthetic_dev:
        raise ValueError("real controller/readout indices are required")
    array = np.asarray(activity if activity is not None else [], dtype=np.float32)
    if selected and (min(selected) < 0 or max(selected) >= array.size):
        raise ValueError("population index is outside the simulation activity")
    values = array[list(selected)] if selected else np.zeros(0, dtype=np.float32)
    mean = float(np.mean(values)) if values.size else 0.0
    maximum = float(np.max(values)) if values.size else 0.0
    # LIF activity is commonly non-negative. Clamp to a bounded control signal
    # so a pathological backend cannot create an unbounded config jump.
    value = float(np.clip(mean, 0.0, 1.0))
    return NeuralReadout(value, selected, mean, maximum)


def _copy_with(config: Any, **updates: Any) -> Any:
    if isinstance(config, Mapping):
        result = dict(config)
        result.update(updates)
        return result
    if is_dataclass(config):
        names = {f.name for f in fields(config)}
        dataclass_updates = {k: v for k, v in updates.items() if k in names}
        result = replace(config, **dataclass_updates) if dataclass_updates else copy.copy(config)
    else:
        result = copy.copy(config)
    for key, value in updates.items():
        try:
            setattr(result, key, value)
        except Exception:
            pass
    return result


def config_value(config: Any, name: str, default: Any = None) -> Any:
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def validate_inference_config(config: Any, *, fixed_max_new_tokens: int | None = None) -> Any:
    """Validate a runtime config without loading a model or changing workload."""

    if config is None:
        raise ValueError("an explicit inference configuration is required")
    budget = config_value(config, "max_new_tokens", None)
    if budget is None or int(budget) <= 0:
        raise ValueError("max_new_tokens must be a positive fixed workload budget")
    if fixed_max_new_tokens is not None and int(budget) != int(fixed_max_new_tokens):
        raise ValueError("optimizer may not change fixed max_new_tokens")
    batch = config_value(config, "batch_size", 1)
    if int(batch) <= 0:
        raise ValueError("batch_size must be positive")
    return config


class TensorFlyController:
    """Map a real neural readout to a validated next inference config."""

    def __init__(
        self,
        controller_indices: Sequence[int] | None,
        *,
        synthetic_dev: bool = False,
        min_batch_size: int = 1,
        max_batch_size: int = 16,
    ) -> None:
        if not controller_indices and not synthetic_dev:
            raise ValueError("real controller/readout indices are required")
        self.controller_indices = tuple(int(i) for i in (controller_indices or ()))
        self.synthetic_dev = bool(synthetic_dev)
        self.min_batch_size = int(min_batch_size)
        self.max_batch_size = int(max_batch_size)
        if self.min_batch_size <= 0 or self.max_batch_size < self.min_batch_size:
            raise ValueError("invalid batch-size bounds")

    def readout(self, activity: Any) -> NeuralReadout:
        return read_population_activity(
            activity, self.controller_indices, synthetic_dev=self.synthetic_dev
        )

    def next_config(self, config: Any, readout: NeuralReadout | float) -> tuple[Any, str]:
        validate_inference_config(config)
        value = float(readout.value if isinstance(readout, NeuralReadout) else readout)
        current = int(config_value(config, "batch_size", self.min_batch_size))
        current = int(np.clip(current, self.min_batch_size, self.max_batch_size))
        if value >= 0.66:
            target, action = min(self.max_batch_size, current + 1), "increase_batch_size"
        elif value <= 0.33:
            target, action = max(self.min_batch_size, current - 1), "decrease_batch_size"
        else:
            target, action = current, "hold_batch_size"
        return _copy_with(config, batch_size=target), action


__all__ = [
    "METRIC_ORDER",
    "SensoryEncoding",
    "SensoryEncoder",
    "NeuralReadout",
    "read_population_activity",
    "validate_inference_config",
    "TensorFlyController",
    "config_value",
]
