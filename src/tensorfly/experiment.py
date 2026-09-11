"""The actual TensorFly experiment loop.

Qwen is intentionally injected as ``benchmark``.  Importing this module, and
constructing an experiment, never downloads or runs a model; the Colab
notebook supplies a real benchmark callable when it is executed.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .controller import (
    NeuralReadout,
    SensoryEncoder,
    TensorFlyController,
    validate_inference_config,
)
from .optimizer import (
    ConfigurationSpace,
    RewardResult,
    RewardWeights,
    compute_reward,
    workload_signature,
)


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _plain(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {str(k): _plain(v) for k, v in vars(value).items() if not str(k).startswith("_")}
    return value


def _config_plain(config: Any) -> dict[str, Any]:
    result = _plain(config)
    if isinstance(result, dict):
        return result
    return {"value": result}


def _metrics_plain(result: Any) -> dict[str, Any]:
    result = _plain(result)
    if isinstance(result, list) and result and all(isinstance(item, Mapping) for item in result):
        # QwenInference.benchmark returns one measured row per batch.  Keep
        # the raw rows for provenance while exposing deterministic means for
        # the controller/reward functions.
        numeric: dict[str, list[float]] = {}
        for row in result:
            for key, value in row.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric.setdefault(str(key), []).append(float(value))
        return {
            **{key: float(np.mean(values)) for key, values in numeric.items()},
            "batch_metrics": result,
        }
    if not isinstance(result, Mapping):
        raise TypeError("benchmark must return a mapping or dataclass of raw metrics")
    metrics = dict(result)
    # A benchmark is allowed to return nested metadata, but required metrics
    # must stay directly accessible for reward and replay provenance.
    for key in ("ttft_ms", "decode_ms_per_token", "throughput_tps", "tokens_per_second"):
        if key not in metrics:
            continue
    return metrics


def _invoke_benchmark(
    benchmark: Callable[..., Any],
    config: Any,
    prompt_corpus: Sequence[str],
    max_new_tokens: int,
    warmup: int,
) -> dict[str, Any]:
    """Call injected benchmark without constructing a Qwen object."""
    kwargs = {
        "config": config,
        "prompt_corpus": tuple(prompt_corpus),
        "prompts": tuple(prompt_corpus),
        "max_new_tokens": int(max_new_tokens),
        "token_budget": int(max_new_tokens),
        "warmup": int(warmup),
    }
    try:
        signature = inspect.signature(benchmark)
        parameters = signature.parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        if accepts_kwargs:
            return _metrics_plain(benchmark(**kwargs))
        selected = {name: value for name, value in kwargs.items() if name in parameters}
        if selected:
            return _metrics_plain(benchmark(**selected))
    except (TypeError, ValueError):
        pass
    # Lightweight fakes frequently expose a single positional config.
    for args in ((config,), (config, tuple(prompt_corpus)), ()): 
        try:
            return _metrics_plain(benchmark(*args))
        except TypeError:
            continue
    raise TypeError("benchmark callable does not accept a supported signature")


@dataclass
class TrialRecord:
    trial_index: int
    strategy: str
    input_config: dict[str, Any]
    raw_inference_metrics: dict[str, Any]
    normalized_sensory_encoding: dict[str, float]
    neural_population_activities: dict[str, Any]
    controller_readout: dict[str, Any]
    chosen_action: str
    resulting_next_config: dict[str, Any]
    reward: float
    dopamine_modulatory_response: float
    simulation_state: dict[str, Any]
    workload: dict[str, Any]


class TensorFlyExperiment:
    """Run benchmark → sensory drive → MaleCNS dynamics → controller loop."""

    def __init__(
        self,
        *,
        model: str,
        benchmark: Callable[..., Any] | None = None,
        simulation: Any | None = None,
        sensory_encoder: SensoryEncoder | None = None,
        controller: TensorFlyController | None = None,
        dopamine_indices: Sequence[int] | None = None,
        synthetic_dev: bool = False,
        simulation_factory: Callable[[], Any] | None = None,
        reward_weights: RewardWeights | None = None,
        configuration_space: ConfigurationSpace | None = None,
        population_body_ids: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        if not model:
            raise ValueError("model identifier is required for provenance")
        # The notebook-facing default is real preparation plus a lazy Qwen
        # callable.  Dependency injection remains useful for CPU-only tests.
        registry = None
        if simulation is None:
            from .dataset import prepare_malecns
            from .populations import build_population_registry
            from .simulation import MaleCNSSimulation
            dataset = prepare_malecns(synthetic_dev=synthetic_dev)
            registry = build_population_registry(dataset)
            simulation = MaleCNSSimulation(dataset, synthetic_dev=synthetic_dev)
            simulation_factory = lambda: MaleCNSSimulation(dataset, synthetic_dev=synthetic_dev)
        if benchmark is None:
            from .inference import InferenceConfig, QwenInference
            def benchmark(*, config: Any, prompt_corpus: Sequence[str], max_new_tokens: int, warmup: int) -> Any:
                effective = replace(config, max_new_tokens=max_new_tokens) if is_dataclass(config) else InferenceConfig(**{**dict(config), "max_new_tokens": max_new_tokens})
                return QwenInference(effective).benchmark(prompt_corpus, warmup=warmup)
        if not callable(benchmark):
            raise TypeError("benchmark must be callable")
        self.synthetic_dev = bool(synthetic_dev)
        if not self.synthetic_dev and not hasattr(simulation, "is_synthetic"):
            raise ValueError("simulation provenance must explicitly declare real MaleCNS data")
        source_label = str(getattr(simulation, "data_source", "")).lower()
        if (bool(getattr(simulation, "is_synthetic", False)) or "synthetic" in source_label) and not self.synthetic_dev:
            raise ValueError("normal TensorFly execution refuses synthetic simulation data")
        self.model = str(model)
        self.benchmark = benchmark
        self.simulation = simulation
        self.simulation_factory = simulation_factory
        if registry is not None:
            sensory_encoder = sensory_encoder or SensoryEncoder(registry.sensory_indices, num_neurons=simulation.num_neurons, synthetic_dev=synthetic_dev)
            controller = controller or TensorFlyController(registry.controller_indices, synthetic_dev=synthetic_dev, min_batch_size=1, max_batch_size=4)
            dopamine_indices = dopamine_indices if dopamine_indices is not None else registry.dopamine_indices
            population_body_ids = population_body_ids or registry.body_ids
        if sensory_encoder is None or controller is None:
            raise ValueError("real sensory encoder and controller/readout are required")
        self.sensory_encoder = sensory_encoder
        self.controller = controller
        self.dopamine_indices = tuple(
            int(i) for i in (() if dopamine_indices is None else dopamine_indices)
        )
        if not self.dopamine_indices and not self.synthetic_dev:
            raise ValueError("real dopaminergic/modulatory indices are required")
        self.reward_weights = reward_weights or RewardWeights()
        self.configuration_space = configuration_space or ConfigurationSpace()
        self.population_body_ids = {
            str(k): tuple(int(x) for x in v) for k, v in (population_body_ids or {}).items()
        }
        self.records: list[TrialRecord] = []
        from .inference import InferenceConfig
        self.default_config = InferenceConfig(model_id=self.model, batch_size=2)
        self.metadata: dict[str, Any] = {
            "model": self.model,
            "synthetic_dev": self.synthetic_dev,
            "population_body_ids": _plain(self.population_body_ids),
            "metric_encoding": "fixed canonical metric order onto selected real sensory indices",
            "physiology_status": "engineered LIF/control approximation; not reconstructed fly physiology",
        }

    def _new_simulation(self) -> Any:
        if self.simulation_factory is not None:
            candidate = self.simulation_factory()
            if not self.synthetic_dev and not hasattr(candidate, "is_synthetic"):
                raise ValueError("simulation provenance must explicitly declare real MaleCNS data")
            if (bool(getattr(candidate, "is_synthetic", False)) or "synthetic" in str(getattr(candidate, "data_source", "")).lower()) and not self.synthetic_dev:
                raise ValueError("simulation factory returned synthetic data in normal mode")
            return candidate
        reset = getattr(self.simulation, "reset_dynamics", None)
        if not callable(reset):
            reset = getattr(self.simulation, "reset", None)
        if callable(reset):
            reset()
        return self.simulation

    @staticmethod
    def _activity(simulation: Any, snapshot: Any = None) -> np.ndarray:
        activity = getattr(simulation, "activity", None)
        if activity is None and isinstance(snapshot, Mapping):
            activity = snapshot.get("activity")
        if activity is None and hasattr(snapshot, "activity"):
            activity = getattr(snapshot, "activity")
        if activity is None:
            raise ValueError("simulation must expose actual activity for controller readout")
        return np.asarray(activity, dtype=np.float32)

    @staticmethod
    def _snapshot(simulation: Any, fallback: Any = None) -> dict[str, Any]:
        by_body_id = getattr(simulation, "activity_by_body_id", None)
        if callable(by_body_id):
            state = {"activity_by_body_id": by_body_id(only_active=True)}
            if fallback is not None:
                state["snapshot"] = _config_plain(fallback)
            return state
        recorder = getattr(simulation, "record_snapshot", None)
        if not callable(recorder):
            recorder = getattr(simulation, "snapshot", None)
        if callable(recorder):
            try:
                return _config_plain(recorder())
            except Exception:
                pass
        if fallback is not None:
            return _config_plain(fallback)
        state: dict[str, Any] = {}
        for name in ("current_step", "activity", "voltage", "spike_counts"):
            if hasattr(simulation, name):
                state[name] = _plain(getattr(simulation, name))
        return state

    def _step(self, simulation: Any, drive: np.ndarray) -> tuple[Any, np.ndarray]:
        step = getattr(simulation, "step", None)
        if not callable(step):
            raise ValueError("simulation must expose step(external_drive=...)")
        existing_activity = getattr(simulation, "activity", None)
        if existing_activity is not None and np.asarray(existing_activity).size < drive.size:
            raise ValueError("sensory drive is larger than the simulation activity state")
        try:
            snapshot = step(external_drive=drive)
        except TypeError:
            # A minimal CPU fake may accept the drive positionally, but never
            # silently omit the drive in normal mode.
            try:
                snapshot = step(drive)
            except TypeError as exc:
                raise ValueError("simulation.step must accept the sensory drive") from exc
        return snapshot, self._activity(simulation, snapshot)

    def _apply_dopamine(self, simulation: Any, current: float) -> None:
        if not self.dopamine_indices:
            return
        for name in ("apply_modulatory_current", "apply_reward"):
            method = getattr(simulation, name, None)
            if callable(method):
                try:
                    method(self.dopamine_indices, float(current))
                except TypeError:
                    method(float(current), self.dopamine_indices)
                return
        activity = getattr(simulation, "activity", None)
        if activity is None:
            raise ValueError("simulation must support modulatory current injection")
        array = np.asarray(activity)
        if max(self.dopamine_indices) >= array.size:
            raise ValueError("dopaminergic index is outside simulation activity")
        array[list(self.dopamine_indices)] += np.float32(current)

    def _run_strategy(
        self,
        strategy: str,
        initial_config: Any,
        *,
        prompt_corpus: Sequence[str],
        trials: int,
        max_new_tokens: int,
        warmup: int,
        rng: np.random.Generator | None = None,
        configuration_space: ConfigurationSpace | None = None,
    ) -> list[TrialRecord]:
        if trials <= 0:
            raise ValueError("trials must be positive")
        validate_inference_config(initial_config, fixed_max_new_tokens=max_new_tokens)
        sim = self._new_simulation()
        records: list[TrialRecord] = []
        config = initial_config
        baseline: Mapping[str, Any] | None = None
        previous_reward = 0.0
        search_space = configuration_space or self.configuration_space
        for index in range(int(trials)):
            validate_inference_config(config, fixed_max_new_tokens=max_new_tokens)
            input_config = _config_plain(config)
            metrics = _invoke_benchmark(self.benchmark, config, prompt_corpus, max_new_tokens, warmup)
            if baseline is None:
                baseline = metrics
            reward_result: RewardResult = compute_reward(metrics, baseline, weights=self.reward_weights)
            encoding = self.sensory_encoder.encode(metrics)
            snapshot, activity = self._step(sim, encoding.drive)
            readout: NeuralReadout = self.controller.readout(activity)
            activities = {
                "sensory": {"indices": list(encoding.selected_indices), "mean": float(np.mean(activity[list(encoding.selected_indices)])) if encoding.selected_indices else 0.0},
                "controller": asdict(readout),
                "dopaminergic": {"indices": list(self.dopamine_indices), "current": reward_result.dopamine_current},
            }
            if strategy == "tensorfly":
                next_config, action = self.controller.next_config(config, readout)
            elif strategy == "default":
                next_config, action = config, "fixed_default"
            elif strategy == "random":
                if rng is None:
                    raise ValueError("random strategy requires an explicit seeded RNG")
                next_config, action = search_space.deterministic_random(config, rng), "random_search"
            elif strategy == "hill_climbing":
                direction = 1 if reward_result.reward >= previous_reward else -1
                batch = int(getattr(config, "batch_size", config.get("batch_size", 1) if isinstance(config, Mapping) else 1))
                choices = sorted(search_space.batch_sizes)
                pos = min(range(len(choices)), key=lambda i: abs(choices[i] - batch))
                pos = max(0, min(len(choices) - 1, pos + direction))
                next_config, action = self.configuration_space.candidate(config, batch_size=choices[pos]), "hill_climb"
            else:
                raise ValueError(f"unknown strategy: {strategy}")
            next_batch = int(getattr(next_config, "batch_size", next_config.get("batch_size", 1) if isinstance(next_config, Mapping) else 1))
            if len(prompt_corpus) % next_batch:
                next_config, action = config, f"{action}_fixed_for_complete_batch"
            self._apply_dopamine(sim, reward_result.dopamine_current)
            state = self._snapshot(sim, snapshot)
            record = TrialRecord(
                trial_index=index,
                strategy=strategy,
                input_config=input_config,
                raw_inference_metrics=metrics,
                normalized_sensory_encoding=encoding.normalized_metrics,
                neural_population_activities=activities,
                controller_readout=asdict(readout),
                chosen_action=action,
                resulting_next_config=_config_plain(next_config),
                reward=reward_result.reward,
                dopamine_modulatory_response=reward_result.dopamine_current,
                simulation_state=state,
                workload=workload_signature(prompt_corpus, max_new_tokens, warmup),
            )
            records.append(record)
            previous_reward = reward_result.reward
            config = next_config
        return records

    def run(
        self,
        *,
        prompt_corpus: Sequence[str],
        trials: int = 20,
        config: Any | None = None,
        max_new_tokens: int | None = None,
        warmup: int = 1,
    ) -> list[TrialRecord]:
        """Run TensorFly with a fixed prompt/token workload."""
        if not prompt_corpus:
            raise ValueError("prompt_corpus must not be empty")
        if config is None:
            config = self.default_config
        budget = int(max_new_tokens if max_new_tokens is not None else getattr(config, "max_new_tokens", config.get("max_new_tokens", 0) if isinstance(config, Mapping) else 0))
        if len(prompt_corpus) % int(getattr(config, "batch_size", config.get("batch_size", 1) if isinstance(config, Mapping) else 1)):
            raise ValueError("prompt_corpus length must be divisible by the fixed batch_size")
        records = self._run_strategy("tensorfly", config, prompt_corpus=prompt_corpus, trials=trials, max_new_tokens=budget, warmup=warmup, configuration_space=self.configuration_space.for_prompt_count(len(prompt_corpus)))
        self.records = records
        return records

    def compare_baselines(
        self,
        *,
        prompt_corpus: Sequence[str],
        config: Any | None = None,
        trials: int = 20,
        max_new_tokens: int | None = None,
        warmup: int = 1,
        seed: int = 0,
    ) -> dict[str, list[TrialRecord]]:
        """Run default, random, hill-climbing, and TensorFly equally."""
        if not prompt_corpus:
            raise ValueError("prompt_corpus must not be empty")
        if config is None:
            config = self.default_config
        budget = int(max_new_tokens if max_new_tokens is not None else getattr(config, "max_new_tokens", config.get("max_new_tokens", 0) if isinstance(config, Mapping) else 0))
        initial_batch = int(getattr(config, "batch_size", config.get("batch_size", 1) if isinstance(config, Mapping) else 1))
        if len(prompt_corpus) % initial_batch:
            raise ValueError("prompt_corpus length must be divisible by the fixed batch_size")
        search_space = self.configuration_space.for_prompt_count(len(prompt_corpus))
        results: dict[str, list[TrialRecord]] = {}
        for strategy in ("default", "random", "hill_climbing", "tensorfly"):
            results[strategy] = self._run_strategy(
                strategy,
                config,
                prompt_corpus=prompt_corpus,
                trials=trials,
                max_new_tokens=budget,
                warmup=warmup,
                rng=np.random.default_rng(seed) if strategy == "random" else None,
                configuration_space=search_space,
            )
        self.records = [record for rows in results.values() for record in rows]
        return results

    def replay(self, strategy: str | None = None) -> list[dict[str, Any]]:
        """Return frames made exclusively from recorded experiment rows."""
        selected = [r for r in self.records if strategy is None or r.strategy == strategy]
        return [_plain(asdict(record)) for record in selected]

    def export_replay(self, path: str | Path, strategy: str | None = None) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": "tensorfly-experiment-replay/1", "meta": _plain(self.metadata), "frames": self.replay(strategy)}
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return destination

    def export_viewer_replay(self, path: str | Path = "viewer/tensorfly_replay.json", strategy: str | None = None) -> Path:
        """Write actual recorded metrics/states in the production-viewer schema."""
        destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
        rows = [record for record in self.records if strategy is None or record.strategy == strategy]
        frames = [{
            "trial": record.trial_index, "config": record.input_config,
            "next_config": record.resulting_next_config, "chosen_action": record.chosen_action,
            "reward": record.reward, "dopamine_modulatory_response": record.dopamine_modulatory_response,
            **record.raw_inference_metrics, **record.simulation_state,
        } for record in rows]
        payload = {"schema": "tensorfly-real-replay/1", "meta": {**_plain(self.metadata), "provenance": {"is_synthetic": self.synthetic_dev, "source": getattr(self.simulation, "data_source", "unknown")}}, "frames": frames}
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return destination

    def export_video(self, path: str | Path = "viewer/tensorfly_replay.json") -> Path:
        """Export the replay consumed by the real-morphology Three.js renderer.

        Rendering/capture happens in that viewer so metrics are never timed by
        the rendering process.  The resulting browser capture is the video.
        """
        return self.export_viewer_replay(path)


__all__ = ["TrialRecord", "TensorFlyExperiment"]
