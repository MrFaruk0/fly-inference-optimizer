"""TensorFly public API: real MaleCNS by default, synthetic only by opt-in."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import (DatasetPreparationError, MaleCNSDataset, download_selected_skeletons,
                      export_viewer_morphology, prepare_malecns)
from .inference import DEFAULT_PROMPTS, MODEL_9B, InferenceConfig, QwenInference
from .populations import PopulationRegistry, build_population_registry
from .simulation import ActivitySnapshot, MaleCNSSimulation, SimulationConfig


@dataclass
class PreparedTensorFly:
    dataset: MaleCNSDataset
    populations: PopulationRegistry
    viewer_morphology: Path | None


def prepare(cache_dir: str | Path | None = None, *, build_viewer: bool = True, synthetic_dev: bool = False) -> PreparedTensorFly:
    """Prepare official data, auditable populations, and real viewer geometry.

    The normal path is fail-closed and downloads Janelia sources itself.
    ``synthetic_dev`` exists exclusively for tests/development.
    """
    dataset = prepare_malecns(cache_dir, synthetic_dev=synthetic_dev)
    populations = build_population_registry(dataset)
    viewer_path: Path | None = None
    if build_viewer:
        # Browser subset: every selected population if small; otherwise a
        # deterministic evenly-spaced context-preserving cap per population.
        def cap(values: np.ndarray, maximum: int = 256) -> np.ndarray:
            return values if len(values) <= maximum else values[np.linspace(0, len(values) - 1, maximum, dtype=int)]
        selected = np.unique(np.concatenate([cap(populations.sensory_body_ids), cap(populations.dopamine_body_ids), cap(populations.controller_body_ids)]))
        context = dataset.body_ids[np.linspace(0, dataset.num_neurons - 1, min(256, dataset.num_neurons), dtype=int)]
        selected = np.unique(np.concatenate([selected, context]))
        skeleton_dir = download_selected_skeletons(selected, dataset.cache_dir)
        labels = {str(int(value)): name for name, values in populations.body_ids.items() for value in values}
        viewer_path = export_viewer_morphology(dataset, selected, skeleton_dir, Path("viewer") / "real-morphology.json", labels)
    return PreparedTensorFly(dataset, populations, viewer_path)


from .controller import NeuralReadout, SensoryEncoder, TensorFlyController
from .experiment import TensorFlyExperiment, TrialRecord
from .optimizer import ConfigurationSpace, RewardWeights, compute_reward

__all__ = [
    "DEFAULT_PROMPTS", "MODEL_9B", "InferenceConfig", "QwenInference", "DatasetPreparationError", "MaleCNSDataset", "prepare_malecns", "prepare",
    "PreparedTensorFly", "PopulationRegistry", "build_population_registry", "ActivitySnapshot", "MaleCNSSimulation", "SimulationConfig",
    "SensoryEncoder", "NeuralReadout", "TensorFlyController", "TensorFlyExperiment", "TrialRecord", "ConfigurationSpace", "RewardWeights", "compute_reward",
]
__version__ = "0.2.0"
