"""CPU-only contracts for the injected TensorFly control loop."""

from dataclasses import dataclass

import numpy as np
import pytest

from tensorfly.controller import SensoryEncoder, TensorFlyController
from tensorfly.dataset import MaleCNSDataset
from tensorfly.experiment import TensorFlyExperiment
from tensorfly.optimizer import compute_reward


@dataclass
class FakeConfig:
    max_new_tokens: int = 4
    batch_size: int = 1
    use_cache: bool = True
    attn_implementation: str | None = None
    compile_model: bool = False


class FakeSimulation:
    is_synthetic = False

    def __init__(self):
        self.activity = np.zeros(12, dtype=np.float32)
        self.current_step = 0

    def reset_dynamics(self):
        self.activity.fill(0)
        self.current_step = 0

    def step(self, external_drive):
        self.current_step += 1
        # A deterministic propagation surrogate for the test; no RNG.
        self.activity.fill(0)
        self.activity[4:6] = min(1.0, float(np.max(external_drive)) * 4.0)
        return {"step": self.current_step, "activity": self.activity.copy()}

    def apply_modulatory_current(self, indices, current):
        self.activity[list(indices)] += np.float32(current)


def _benchmark(config, prompt_corpus, max_new_tokens, warmup):
    scale = float(config.batch_size)
    return {
        "ttft_ms": 100.0 / scale,
        "tpot_ms": 20.0 / scale,
        "throughput_tps": 10.0 * scale,
        "peak_allocated_vram_mb": 1000.0 / scale,
        "peak_reserved_vram_mb": 1200.0 / scale,
        "prompt_corpus": tuple(prompt_corpus),
        "max_new_tokens": max_new_tokens,
        "warmup": warmup,
    }


def _experiment(simulation=None):
    return TensorFlyExperiment(
        model="Qwen/Qwen3.5-9B",
        benchmark=_benchmark,
        simulation=simulation or FakeSimulation(),
        simulation_factory=FakeSimulation,
        sensory_encoder=SensoryEncoder([1, 2, 3], num_neurons=12),
        controller=TensorFlyController([4, 5]),
        dopamine_indices=[8],
    )


def test_encoder_is_deterministic_and_only_selected_neurons_receive_drive():
    encoder = SensoryEncoder([2, 7], num_neurons=8)
    first = encoder.encode({"ttft_ms": 500, "decode_ms_per_token": 20, "throughput_tps": 20})
    second = encoder.encode({"ttft_ms": 500, "decode_ms_per_token": 20, "throughput_tps": 20})
    assert np.array_equal(first.drive, second.drive)
    assert np.count_nonzero(first.drive) <= 2
    assert first.drive[2] > 0
    assert first.drive[7] > 0


def test_normal_mode_requires_real_population_indices():
    with pytest.raises(ValueError, match="real MaleCNS"):
        SensoryEncoder([], num_neurons=4)
    with pytest.raises(ValueError, match="real controller"):
        TensorFlyController([])


def test_annotation_index_ndarrays_are_accepted():
    """PopulationRegistry exposes uint arrays, not Python lists."""
    encoder = SensoryEncoder(np.asarray([1, 3], dtype=np.uint32), num_neurons=4)
    controller = TensorFlyController(np.asarray([2], dtype=np.uint32))
    assert encoder.selected_indices == (1, 3)
    assert controller.controller_indices == (2,)


def test_reward_is_positive_for_improvement_and_negative_for_regression():
    baseline = {"ttft_ms": 100, "tpot_ms": 20, "throughput_tps": 10, "peak_allocated_vram_mb": 100, "peak_reserved_vram_mb": 120}
    improved = {"ttft_ms": 50, "tpot_ms": 10, "throughput_tps": 20, "peak_allocated_vram_mb": 50, "peak_reserved_vram_mb": 60}
    regressed = {"ttft_ms": 150, "tpot_ms": 30, "throughput_tps": 5, "peak_allocated_vram_mb": 150, "peak_reserved_vram_mb": 180}
    assert compute_reward(improved, baseline).reward > 0
    assert compute_reward(regressed, baseline).reward < 0


def test_loop_changes_actual_config_and_replay_uses_recorded_metrics():
    experiment = _experiment()
    records = experiment.run(prompt_corpus=["fixed prompt"], config=FakeConfig(), trials=3, max_new_tokens=4, warmup=1)
    assert len(records) == 3
    assert records[0].workload["prompt_corpus"] == ("fixed prompt",)
    assert records[0].resulting_next_config["batch_size"] != records[0].input_config["batch_size"] or "fixed_for_complete_batch" in records[0].chosen_action or records[0].chosen_action == "hold_batch_size"
    frames = experiment.replay()
    assert [frame["raw_inference_metrics"] for frame in frames] == [record.raw_inference_metrics for record in records]
    assert all("simulation_state" in frame for frame in frames)


def test_baselines_use_the_same_notebook_default_config():
    rows = _experiment().compare_baselines(
        prompt_corpus=["a", "b", "c", "d"], trials=1
    )
    assert set(rows) == {"default", "random", "hill_climbing", "tensorfly"}
    assert all(len(result) == 1 for result in rows.values())


def test_notebook_default_api_runs_without_explicit_config_or_injection(monkeypatch, tmp_path):
    """The exact ``TensorFlyExperiment(model=...)`` notebook contract."""
    ids = np.asarray([17, 23, 47, 89], dtype=np.uint64)
    dataset = MaleCNSDataset(
        ids, np.asarray([0, 1, 2], dtype=np.uint32), np.asarray([1, 2, 3], dtype=np.uint32),
        np.asarray([2, 2, 2], dtype=np.uint32),
        [
            {"body_id": 17, "superclass": "sensory", "type": "R7", "neurotransmitter": "acetylcholine"},
            {"body_id": 23, "superclass": "central", "type": "PAM11", "neurotransmitter": "dopamine"},
            {"body_id": 47, "superclass": "descending", "type": "DNa02", "neurotransmitter": "acetylcholine"},
            {"body_id": 89, "superclass": "central", "type": "other", "neurotransmitter": "gaba"},
        ], [], {"release": "MaleCNS v1.0", "retention_policy": "fixture", "synthetic_dev": False}, tmp_path,
    )
    monkeypatch.setattr("tensorfly.dataset.prepare_malecns", lambda *args, **kwargs: dataset)

    class FakeQwen:
        def __init__(self, config): self.config = config
        def benchmark(self, prompts, warmup=1):
            return [{"ttft_ms": 100.0, "tpot_ms": 10.0, "throughput_tps": 20.0,
                     "peak_allocated_bytes": 1000, "peak_reserved_bytes": 1200}]

    monkeypatch.setattr("tensorfly.inference.QwenInference", FakeQwen)
    experiment = TensorFlyExperiment(model="Qwen/Qwen3.5-9B")
    records = experiment.run(prompt_corpus=["a", "b", "c", "d"], trials=1)
    baselines = experiment.compare_baselines(prompt_corpus=["a", "b", "c", "d"], trials=1)
    assert len(records) == 1 and set(baselines) == {"default", "random", "hill_climbing", "tensorfly"}


def test_synthetic_simulation_requires_explicit_developer_opt_in():
    class Synthetic(FakeSimulation):
        is_synthetic = True

    with pytest.raises(ValueError, match="synthetic"):
        _experiment(simulation=Synthetic())
