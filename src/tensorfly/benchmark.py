"""TensorFly inference benchmarking: isolated timing + config comparisons.

Notebook usage::

    from tensorfly.benchmark import run_benchmark, compare_configs
    from tensorfly.runtime import get_runtime
    from tensorfly.simulation import MaleCNSSimulation, SimulationConfig

    rt = get_runtime()
    sim = MaleCNSSimulation(SimulationConfig(num_neurons=2000, num_edges=8000))
    sim.build()
    result = run_benchmark(sim, steps=5, repeats=3, runtime=rt)
    result.save("benchmark.json")
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Callable, Dict, List, Optional

from .simulation import MaleCNSSimulation

try:
    from .runtime import Runtime, get_runtime
except Exception:  # pragma: no cover - import guard for minimal installs
    Runtime = Any  # type: ignore[misc, assignment]
    get_runtime = None  # type: ignore[assignment]


@dataclass
class BenchmarkConfig:
    """Knobs for an isolated inference-timing run."""

    steps: int = 10
    repeats: int = 5
    warmup: int = 1
    label: str = "default"

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.repeats <= 0 or self.warmup < 0:
            raise ValueError("steps/repeats must be >0 and warmup >=0")


@dataclass
class BenchmarkResult:
    """Timing statistics + provenance metadata for one benchmark run."""

    label: str
    steps: int
    repeats: int
    warmup: int
    # Per-repeat wall time for ``steps`` inference steps (seconds).
    repeat_seconds: List[float] = field(default_factory=list)
    # Derived per-step latency in milliseconds.
    ms_per_step_mean: float = 0.0
    ms_per_step_median: float = 0.0
    ms_per_step_stdev: float = 0.0
    steps_per_second: float = 0.0
    total_spikes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return dest


def _time_call(fn: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    out = fn()
    return time.perf_counter() - start, out


def time_inference(
    step_fn: Callable[[], Any],
    repeats: int = 5,
    warmup: int = 1,
) -> List[float]:
    """Time ``step_fn`` in isolation (warmup runs excluded from results)."""
    for _ in range(max(0, warmup)):
        step_fn()
    times: List[float] = []
    for _ in range(repeats):
        dt, _ = _time_call(step_fn)
        times.append(dt)
    return times


def _summarize(
    label: str,
    cfg: BenchmarkConfig,
    repeat_seconds: List[float],
    total_spikes: int,
    metadata: Dict[str, Any],
) -> BenchmarkResult:
    per_step_ms = [(t / cfg.steps) * 1000.0 for t in repeat_seconds]
    avg_ms = mean(per_step_ms) if per_step_ms else 0.0
    med_ms = median(per_step_ms) if per_step_ms else 0.0
    sd_ms = stdev(per_step_ms) if len(per_step_ms) > 1 else 0.0
    sps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    return BenchmarkResult(
        label=label,
        steps=cfg.steps,
        repeats=cfg.repeats,
        warmup=cfg.warmup,
        repeat_seconds=list(repeat_seconds),
        ms_per_step_mean=avg_ms,
        ms_per_step_median=med_ms,
        ms_per_step_stdev=sd_ms,
        steps_per_second=sps,
        total_spikes=total_spikes,
        metadata=dict(metadata),
    )


def run_benchmark(
    sim: MaleCNSSimulation,
    steps: int = 10,
    repeats: int = 5,
    warmup: int = 1,
    label: str = "default",
    runtime: Optional[Any] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    reset_between_repeats: bool = True,
) -> BenchmarkResult:
    """Benchmark isolated ``sim.step()`` timing (no build/setup included).

    The simulation is warmed up, then each repeat runs ``steps`` fresh
    inference steps. Only the step loop is timed, so config comparisons
    isolate inference cost from graph construction.

    Repeats are fair by default (``reset_between_repeats=True``): volatile
    dynamics (voltage/activity) are reset via
    ``sim.reset_dynamics()`` before warmup and before every timed repeat
    so repeats do not drift on prior spiking history. Set to ``False`` to
    preserve the legacy cumulative behaviour; drift is then reported in
    metadata (``repeat_spikes`` + ``drift``) either way.
    """
    cfg = BenchmarkConfig(steps=steps, repeats=repeats, warmup=warmup, label=label)
    if not sim.built:
        sim.build()

    resetter = getattr(sim, "reset_dynamics", None)

    def _maybe_reset() -> None:
        if reset_between_repeats and callable(resetter):
            try:
                resetter()
            except Exception:
                pass

    def one_repeat() -> None:
        for _ in range(cfg.steps):
            sim.step()

    # Warmup (excluded), then timed repeats with per-repeat spike tracking.
    for _ in range(cfg.warmup):
        _maybe_reset()
        one_repeat()
    repeat_seconds: List[float] = []
    repeat_spikes: List[int] = []
    for _ in range(cfg.repeats):
        _maybe_reset()
        spikes_before = int(sim.spike_counts.sum())
        start = time.perf_counter()
        one_repeat()
        elapsed = time.perf_counter() - start
        repeat_seconds.append(elapsed)
        repeat_spikes.append(int(sim.spike_counts.sum()) - spikes_before)
    total_spikes = int(sum(repeat_spikes))

    # Drift report: identical repeats should produce similar spike counts
    # when reset; cumulative mode is expected to drift and that is surfaced.
    if repeat_spikes:
        drift_range = int(max(repeat_spikes) - min(repeat_spikes))
        drift_mean = float(mean(repeat_spikes))
    else:
        drift_range = 0
        drift_mean = 0.0

    metadata: Dict[str, Any] = {
        "simulation": sim.summary(),
        "config": asdict(cfg),
        "reset_between_repeats": bool(reset_between_repeats),
        "repeat_spikes": list(repeat_spikes),
        "drift": {
            "repeat_spikes": list(repeat_spikes),
            "range": drift_range,
            "mean": drift_mean,
        },
    }
    if runtime is not None:
        try:
            metadata["runtime"] = runtime.to_dict()
        except Exception:
            metadata["runtime"] = {"profile": str(getattr(runtime, "profile", runtime))}
    elif get_runtime is not None:
        try:
            metadata["runtime"] = get_runtime().to_dict()  # type: ignore[misc]
        except Exception:
            pass
    if extra_metadata:
        metadata.update(extra_metadata)

    return _summarize(cfg.label, cfg, repeat_seconds, total_spikes, metadata)


def compare_configs(
    sim_factory: Callable[[], MaleCNSSimulation],
    configs: List[BenchmarkConfig],
    runtime: Optional[Any] = None,
    reset_between_repeats: bool = True,
) -> List[BenchmarkResult]:
    """Run isolated timing for several configs, each on a fresh simulation.

    A fresh simulation per config keeps comparisons fair (identical start
    state up to the config label) and records per-config metadata.

    If ``sim_factory`` accepts a single ``BenchmarkConfig`` argument it is
    called as ``sim_factory(cfg)`` so configs actually affect the built
    graph/seed; zero-arg factories keep working unchanged (called as
    ``sim_factory()``).
    """
    import inspect

    results: List[BenchmarkResult] = []
    try:
        takes_cfg = len(inspect.signature(sim_factory).parameters) >= 1
    except Exception:
        takes_cfg = False
    for cfg in configs:
        try:
            sim = sim_factory(cfg) if takes_cfg else sim_factory()  # type: ignore[call-arg]
        except TypeError:
            # Factory advertised params but rejects the config: use no-arg.
            sim = sim_factory()  # type: ignore[call-arg]
        results.append(
            run_benchmark(
                sim,
                steps=cfg.steps,
                repeats=cfg.repeats,
                warmup=cfg.warmup,
                label=cfg.label,
                runtime=runtime,
                reset_between_repeats=reset_between_repeats,
            )
        )
    return results


def save_comparison(results: List[BenchmarkResult], path: str | Path) -> Path:
    """Persist a config-comparison table as JSON."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": [r.to_dict() for r in results],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "time_inference",
    "run_benchmark",
    "compare_configs",
    "save_comparison",
]
