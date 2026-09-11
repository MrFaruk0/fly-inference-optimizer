"""Engineered LIF dynamics over a checksum-verified MaleCNS graph.

The graph, IDs, contact counts and transmitter annotations are source data.
Membranes, signs, reward current and controller interpretation are explicitly
engineered approximations; this is not a reconstruction of fly physiology.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .dataset import MaleCNSDataset, prepare_malecns

N_MALECNS_NEURONS = 166_700
N_MALECNS_EDGES = 25_582_938


@dataclass
class SimulationConfig:
    decay: float = 0.95
    threshold: float = 1.0
    reset: float = 0.0
    input_gain: float = 1.0
    contact_gain: float = 0.025
    seed: int = 0
    # This label protects claims at the API boundary.
    dynamics_note: str = "Engineered LIF/contact-count approximation; not reconstructed fly physiology."

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        if self.threshold <= 0 or self.contact_gain < 0:
            raise ValueError("threshold must be positive and contact_gain non-negative")


@dataclass
class ActivitySnapshot:
    step: int
    spikes: int
    mean_voltage: float
    active_body_ids: list[str] = field(default_factory=list)


class MaleCNSSimulation:
    """A simulation that refuses an unprovenanced graph by default.

    Pass the object from ``prepare_malecns()`` to avoid preparation twice.
    ``synthetic_dev=True`` is the sole testing/developer opt-in and cannot be
    reached after any source load error.
    """

    def __init__(self, dataset: Optional[MaleCNSDataset] = None, config: Optional[SimulationConfig] = None, *, cache_dir: str | None = None, synthetic_dev: bool = False) -> None:
        self.config = config or SimulationConfig()
        self.dataset = dataset if dataset is not None else prepare_malecns(cache_dir, synthetic_dev=synthetic_dev)
        self.is_synthetic = bool(self.dataset.report.get("synthetic_dev", False))
        if self.is_synthetic and not synthetic_dev:
            raise RuntimeError("Synthetic dataset requires explicit synthetic_dev=True.")
        self.data_source = "synthetic-dev-fixture" if self.is_synthetic else "official MaleCNS v1.0 (checksum-verified)"
        self.body_ids = self.dataset.body_ids
        self.pre_index = self.dataset.pre_index
        self.post_index = self.dataset.post_index
        self.synapse_count = self.dataset.synapse_count
        self.voltage = np.zeros(self.dataset.num_neurons, dtype=np.float32)
        self.activity = np.zeros(self.dataset.num_neurons, dtype=np.float32)
        self.spike_counts = np.zeros(self.dataset.num_neurons, dtype=np.uint32)
        self.modulatory_current = np.zeros(self.dataset.num_neurons, dtype=np.float32)
        self.current_step = 0
        self._edge_effect = self._engineered_edge_effects()

    @property
    def num_neurons(self) -> int: return self.dataset.num_neurons
    @property
    def num_edges(self) -> int: return self.dataset.num_edges
    @property
    def is_full_malecns_scale(self) -> bool:
        return self.num_neurons == N_MALECNS_NEURONS and self.num_edges == N_MALECNS_EDGES

    def _engineered_edge_effects(self) -> np.ndarray:
        annotations = {int(row["body_id"]): row for row in self.dataset.annotations}
        # A coarse transmitter label-to-sign map is consciously not a receptor model.
        signs = np.ones(self.num_neurons, dtype=np.float32)
        for index, body_id in enumerate(self.body_ids):
            transmitter = str(annotations.get(int(body_id), {}).get("neurotransmitter", "")).lower()
            if transmitter in {"gaba", "glutamate"}:
                signs[index] = -1.0
        return (np.log1p(self.synapse_count.astype(np.float32)) * self.config.contact_gain * signs[self.pre_index]).astype(np.float32)

    def reset(self) -> None:
        self.voltage.fill(0); self.activity.fill(0); self.spike_counts.fill(0); self.modulatory_current.fill(0); self.current_step = 0

    reset_dynamics = reset

    def apply_modulatory_current(self, indices: np.ndarray | list[int] | tuple[int, ...], current: float) -> None:
        """Apply an engineered reward signal to identified real dopamine IDs."""
        selected = np.asarray(indices, dtype=np.int64)
        if selected.size and (selected.min() < 0 or selected.max() >= self.num_neurons):
            raise ValueError("modulatory index outside prepared MaleCNS graph")
        self.modulatory_current[selected] += np.float32(current)

    def step(self, external_drive: Optional[np.ndarray] = None, modulatory_current: Optional[np.ndarray] = None) -> int:
        """Advance one LIF step. An omitted input is exactly zero—not random."""
        drive = np.zeros(self.num_neurons, dtype=np.float32) if external_drive is None else np.asarray(external_drive, dtype=np.float32)
        if drive.shape != (self.num_neurons,):
            raise ValueError(f"external_drive must have shape ({self.num_neurons},)")
        if modulatory_current is not None:
            mod = np.asarray(modulatory_current, dtype=np.float32)
            if mod.shape != drive.shape: raise ValueError("modulatory_current shape mismatch")
            drive = drive + mod
        drive = drive + self.modulatory_current
        self.modulatory_current.fill(0)
        recurrent = np.zeros(self.num_neurons, dtype=np.float32)
        firing = self.activity > 0
        if np.any(firing):
            edge_mask = firing[self.pre_index]
            np.add.at(recurrent, self.post_index[edge_mask], self._edge_effect[edge_mask])
        self.voltage = self.config.decay * self.voltage + self.config.input_gain * drive + recurrent
        fired = self.voltage >= self.config.threshold
        self.activity.fill(0); self.activity[fired] = 1.0
        self.spike_counts[fired] += 1; self.voltage[fired] = self.config.reset
        self.current_step += 1
        return int(np.count_nonzero(fired))

    def run(self, steps: int, external_drive: Optional[np.ndarray] = None, modulatory_current: Optional[np.ndarray] = None) -> list[ActivitySnapshot]:
        if steps < 1: raise ValueError("steps must be positive")
        return [self.snapshot(self.step(external_drive, modulatory_current)) for _ in range(steps)]

    def snapshot(self, spikes: Optional[int] = None, limit: int = 256) -> ActivitySnapshot:
        active = np.flatnonzero(self.activity)[:limit]
        return ActivitySnapshot(self.current_step, int(np.count_nonzero(self.activity) if spikes is None else spikes), float(self.voltage.mean()), [str(int(self.body_ids[i])) for i in active])

    def activity_by_body_id(self, only_active: bool = False) -> Dict[str, float]:
        indices = np.flatnonzero(self.activity) if only_active else np.arange(self.num_neurons)
        return {str(int(self.body_ids[i])): float(self.activity[i]) for i in indices}

    def summary(self) -> Dict[str, Any]:
        return {"dataset": self.data_source, "is_synthetic": self.is_synthetic, "neurons": self.num_neurons, "edges": self.num_edges, "graph_report": self.dataset.report, "dynamics": self.config.dynamics_note}


__all__ = ["N_MALECNS_NEURONS", "N_MALECNS_EDGES", "SimulationConfig", "ActivitySnapshot", "MaleCNSSimulation"]
