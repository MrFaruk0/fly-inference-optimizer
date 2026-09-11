"""TensorFly MaleCNS-scale connectome simulation (memory-efficient scaffold).

Full-network target (MaleCNS v1.0):
    * ``N_NEURONS = 166_700`` neurons
    * ``N_EDGES   = 25_600_000`` directed edges (~25.6M)

Design goals:
    * compact integer indices (CSR ``row_ptr`` + ``col_idx`` int32/int64),
    * float32 state arrays (voltage, thresholds, activity),
    * vectorized NumPy updates (no Python per-neuron loops),
    * optional on-disk memmap / cache directory for large graphs,
    * deterministic RNG + event / activity-snapshot recording.

Provenance honesty:
    * When ``edge_source`` is ``None``, the simulation builds a *synthetic*
      scaffold graph whose degree distribution merely resembles a connectome.
      It is explicitly flagged ``is_synthetic=True`` and
      ``data_source="synthetic-scaffold (NOT real MaleCNS v1.0)"``.
    * When a real source file is supplied (``.npz`` CSR dump or edgelist
      ``.csv``), the loader uses it and marks ``is_synthetic=False``.
    * Nothing in synthetic mode pretends to be real MaleCNS data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Full-network MaleCNS v1.0 target constants
# ---------------------------------------------------------------------------

N_MALECNS_NEURONS: int = 166_700
N_MALECNS_EDGES: int = 25_600_000

SYNTHETIC_MARKER = "synthetic-scaffold (NOT real MaleCNS v1.0)"
REAL_MARKER_PREFIX = "real-source:"

#: In-memory caps so long runs cannot grow event/history lists unbounded.
MAX_EVENT_LOG = 4096
MAX_ACTIVITY_HISTORY = 1_000_000
MAX_SNAPSHOTS = 4096
#: Number of spiking ids retained per step/snapshot (sliced before list()).
SAMPLE_EVENTS_K = 16

CacheMode = Literal["ram", "memmap"]


@dataclass
class SimulationConfig:
    """Configuration for a connectome-scale simulation."""

    num_neurons: int = N_MALECNS_NEURONS
    num_edges: int = N_MALECNS_EDGES
    seed: int = 0
    # Mean out-degree used only by the synthetic scaffold generator.
    synthetic_mean_degree: int = 150
    dtype: str = "float32"  # state/weight precision label
    cache_dir: Optional[Path] = None
    use_memmap: bool = False
    # Leaky integrate-and-fire defaults.
    decay: float = 0.95
    threshold: float = 1.0
    reset: float = 0.0
    input_gain: float = 0.5

    def __post_init__(self) -> None:
        if self.num_neurons <= 0:
            raise ValueError("num_neurons must be positive")
        if self.num_edges < 0:
            raise ValueError("num_edges must be non-negative")
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        if self.use_memmap and self.cache_dir is None:
            raise ValueError("use_memmap=True requires cache_dir to be set")


@dataclass
class ActivitySnapshot:
    """One deterministic recording of network state."""

    step: int
    spikes: int
    mean_voltage: float
    max_voltage: float
    # Lightweight deterministic event log: first K spiking neuron ids.
    sample_events: List[int] = field(default_factory=list)


class MaleCNSSimulation:
    """Memory-efficient CSR connectome simulation.

    Notebook usage::

        from tensorfly.simulation import MaleCNSSimulation, SimulationConfig
        sim = MaleCNSSimulation(SimulationConfig(num_neurons=5000,
                                                num_edges=20000, seed=0))
        sim.build()          # synthetic scaffold, clearly flagged
        out = sim.run(steps=10)
        sim.activity_history # deterministic per-step spike counts
    """

    def __init__(
        self,
        config: SimulationConfig | None = None,
        edge_source: str | Path | None = None,
    ) -> None:
        self.config: SimulationConfig = config or SimulationConfig()
        self.edge_source: Optional[Path] = (
            Path(edge_source) if edge_source is not None else None
        )
        self.rng = np.random.default_rng(self.config.seed)

        n = self.config.num_neurons
        self.voltage: np.ndarray = np.zeros(n, dtype=np.float32)
        self.threshold: np.ndarray = np.full(n, self.config.threshold, dtype=np.float32)
        # Per-neuron spike counter (compact int32) + instantaneous activity.
        self.spike_counts: np.ndarray = np.zeros(n, dtype=np.int32)
        self.activity: np.ndarray = np.zeros(n, dtype=np.float32)

        # CSR connectivity: row_ptr (N+1, int64), col_idx (E, int32),
        # weights (E, float32). Allocated in build().
        self.row_ptr: Optional[np.ndarray] = None
        self.col_idx: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None

        self.is_synthetic: bool = True
        self.data_source: str = SYNTHETIC_MARKER
        self.built: bool = False
        self.current_step: int = 0

        self.activity_history: List[int] = []
        self.event_log: List[Dict[str, Any]] = []
        self.snapshots: List[ActivitySnapshot] = []

        self._memmap_files: List[Path] = []

    # -- construction ------------------------------------------------------

    @property
    def num_neurons(self) -> int:
        return self.config.num_neurons

    @property
    def num_edges(self) -> int:
        if self.col_idx is None:
            return 0
        return int(self.col_idx.shape[0])

    @property
    def is_full_malecns_scale(self) -> bool:
        # Use actual loaded counts (not the requested config budget) so a
        # real source with a different size cannot be misreported as full
        # scale, and vice versa.
        return (
            self.num_neurons == N_MALECNS_NEURONS
            and self.num_edges == N_MALECNS_EDGES
        )

    def build(self) -> "MaleCNSSimulation":
        """Allocate CSR connectivity + reset state (deterministic)."""
        # Re-seed so repeated build() calls are deterministic.
        self.rng = np.random.default_rng(self.config.seed)
        if self.edge_source is not None and Path(self.edge_source).exists():
            self._build_from_real_source(Path(self.edge_source))
        else:
            if self.edge_source is not None:
                # Explicit path that does not exist: do not silently invent
                # data; still build a synthetic scaffold but record the miss.
                self.event_log.append(
                    {
                        "type": "missing-source",
                        "source": str(self.edge_source),
                        "note": (
                            "Requested edge_source not found; built "
                            "synthetic scaffold instead. NOT real MaleCNS."
                        ),
                    }
                )
                if len(self.event_log) > MAX_EVENT_LOG:
                    del self.event_log[: len(self.event_log) - MAX_EVENT_LOG]
            self._build_synthetic_scaffold()
        self.built = True
        return self

    # -- synthetic scaffold -------------------------------------------------

    def _allocate_arrays(
        self, row_ptr: np.ndarray, col_idx: np.ndarray, weights: np.ndarray
    ) -> None:
        if self.config.use_memmap:
            assert self.config.cache_dir is not None
            cache = self.config.cache_dir
            cache.mkdir(parents=True, exist_ok=True)
            # Include actual dims + seed + pid so distinct graphs sharing a
            # cache_dir/seed cannot collide on the same filenames.
            import os

            n = int(row_ptr.shape[0] - 1) if row_ptr.size else 0
            e = int(col_idx.shape[0])
            tag = (
                f"n{n}_e{e}_seed{self.config.seed}_pid{os.getpid()}"
                f"_{row_ptr.dtype}_{col_idx.dtype}_{weights.dtype}"
            )
            # Sanitize dtype strings for filenames ('int64' stays, no spaces).
            tag = tag.replace(" ", "")
            self._memmap_files = [
                cache / f"row_ptr_{tag}.dat",
                cache / f"col_idx_{tag}.dat",
                cache / f"weights_{tag}.dat",
            ]
            for path, arr in zip(
                self._memmap_files, (row_ptr, col_idx, weights)
            ):
                mm = np.memmap(path, dtype=arr.dtype, mode="w+", shape=arr.shape)
                mm[:] = arr[:]
                mm.flush()
            self.row_ptr = np.memmap(
                self._memmap_files[0], dtype=row_ptr.dtype, mode="r", shape=row_ptr.shape
            )
            self.col_idx = np.memmap(
                self._memmap_files[1], dtype=col_idx.dtype, mode="r", shape=col_idx.shape
            )
            self.weights = np.memmap(
                self._memmap_files[2], dtype=weights.dtype, mode="r", shape=weights.shape
            )
        else:
            self.row_ptr = row_ptr
            self.col_idx = col_idx
            self.weights = weights

    def _build_synthetic_scaffold(self) -> None:
        """Vectorized synthetic CSR graph (clearly NOT real MaleCNS)."""
        n = self.config.num_neurons
        target_edges = self.config.num_edges

        # Deterministic degree sequence around synthetic_mean_degree, scaled
        # so the total matches target_edges without materializing an edgelist.
        raw = self.rng.integers(
            low=max(1, self.config.synthetic_mean_degree // 2),
            high=self.config.synthetic_mean_degree * 2,
            size=n,
            dtype=np.int64,
        )
        total = int(raw.sum())
        if total <= 0:
            raw[:] = 1
            total = n
        # Scale to the requested edge budget.
        scaled = np.maximum(1, np.round(raw * (target_edges / total)).astype(np.int64))
        # Fix rounding drift deterministically (adjust tail entries).
        drift = int(scaled.sum()) - target_edges
        idx = n - 1
        while drift > 0 and idx >= 0:
            take = min(drift, int(scaled[idx]) - 1)
            scaled[idx] -= take
            drift -= take
            idx -= 1
        idx = 0
        while drift < 0:
            scaled[idx % n] += 1
            drift += 1
            idx += 1

        row_ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(scaled, out=row_ptr[1:])
        e = int(row_ptr[-1])

        # Compact integer targets: vectorized per-row random targets.
        # For very large E this stays in int32/float32 (4 bytes each).
        col_idx = self.rng.integers(0, n, size=e, dtype=np.int64).astype(np.int32)
        weights = self.rng.normal(0.0, 0.15, size=e).astype(np.float32)

        self._allocate_arrays(row_ptr, col_idx, weights)
        self.is_synthetic = True
        self.data_source = SYNTHETIC_MARKER
        self.event_log.append(
            {
                "type": "build",
                "source": self.data_source,
                "num_neurons": n,
                "num_edges": e,
                "synthetic": True,
                "warning": (
                    "Synthetic scaffold graph for memory/layout benchmarking "
                    "only. NOT real MaleCNS v1.0 connectivity."
                ),
            }
        )
        if len(self.event_log) > MAX_EVENT_LOG:
            del self.event_log[: len(self.event_log) - MAX_EVENT_LOG]

    # -- real source --------------------------------------------------------

    def _build_from_real_source(self, path: Path) -> None:
        """Load real connectivity when the user supplies a source file.

        Supported:
          * ``.npz`` with CSR arrays (``row_ptr``/``col_idx``/``weights`` or
            ``indptr``/``indices``/``data``), or edgelist arrays
            (``sources``/``targets``[, ``weights``]).
          * ``.csv`` edgelist with ``source,target[,weight]`` columns.
        """
        suffix = path.suffix.lower()
        try:
            if suffix == ".npz":
                self._load_npz_source(path)
            elif suffix == ".csv":
                self._load_csv_edgelist(path)
            else:
                raise ValueError(f"Unsupported edge_source format: {suffix}")
        except Exception as exc:
            # Graceful degradation: keep provenance honest, fall back to
            # synthetic instead of crashing notebook workflows.
            self.event_log.append(
                {
                    "type": "source-load-failed",
                    "source": str(path),
                    "error": str(exc),
                    "note": "Fell back to synthetic scaffold. NOT real MaleCNS.",
                }
            )
            if len(self.event_log) > MAX_EVENT_LOG:
                del self.event_log[: len(self.event_log) - MAX_EVENT_LOG]
            self._build_synthetic_scaffold()
            return

        self.is_synthetic = False
        self.data_source = f"{REAL_MARKER_PREFIX}{path.resolve()}"
        self.event_log.append(
            {
                "type": "build",
                "source": self.data_source,
                "num_neurons": self.num_neurons,
                "num_edges": self.num_edges,
                "synthetic": False,
            }
        )
        if len(self.event_log) > MAX_EVENT_LOG:
            del self.event_log[: len(self.event_log) - MAX_EVENT_LOG]

    def _load_npz_source(self, path: Path) -> None:
        data = np.load(path, allow_pickle=False)
        keys = set(data.files)
        if {"row_ptr", "col_idx"}.issubset(keys) or {
            "indptr",
            "indices",
        }.issubset(keys):
            row_ptr = np.asarray(
                data["row_ptr"] if "row_ptr" in keys else data["indptr"],
                dtype=np.int64,
            )
            col_idx = np.asarray(
                data["col_idx"] if "col_idx" in keys else data["indices"],
                dtype=np.int32,
            )
            if "weights" in keys:
                weights = np.asarray(data["weights"], dtype=np.float32)
            elif "data" in keys:
                weights = np.asarray(data["data"], dtype=np.float32)
            else:
                weights = np.ones(col_idx.shape[0], dtype=np.float32)
            self._validate_csr_arrays(row_ptr, col_idx, weights)
            n = int(row_ptr.shape[0] - 1)
            if n != self.config.num_neurons:
                # Adopt the source's neuron count but keep the event honest.
                self.config.num_neurons = n
                self.voltage = np.zeros(n, dtype=np.float32)
                self.threshold = np.full(n, self.config.threshold, dtype=np.float32)
                self.spike_counts = np.zeros(n, dtype=np.int32)
                self.activity = np.zeros(n, dtype=np.float32)
            self._allocate_arrays(row_ptr, col_idx, weights)
            return
        if {"sources", "targets"}.issubset(keys):
            sources = np.asarray(data["sources"], dtype=np.int64)
            targets = np.asarray(data["targets"], dtype=np.int64)
            w = (
                np.asarray(data["weights"], dtype=np.float32)
                if "weights" in keys
                else np.ones(targets.shape[0], dtype=np.float32)
            )
            self._build_csr_from_edgelist(sources, targets, w)
            return
        raise ValueError(f"Unrecognized .npz keys: {sorted(keys)}")

    def _load_csv_edgelist(self, path: Path) -> None:
        import csv

        sources: List[int] = []
        targets: List[int] = []
        weights: List[float] = []
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or not {
                "source",
                "target",
            }.issubset(set(reader.fieldnames)):
                raise ValueError("CSV edgelist needs source,target[,weight] header")
            for row in reader:
                sources.append(int(row["source"]))
                targets.append(int(row["target"]))
                weights.append(float(row.get("weight", 1.0)))
        self._build_csr_from_edgelist(
            np.asarray(sources, dtype=np.int64),
            np.asarray(targets, dtype=np.int64),
            np.asarray(weights, dtype=np.float32),
        )

    def _validate_csr_arrays(
        self, row_ptr: np.ndarray, col_idx: np.ndarray, weights: np.ndarray
    ) -> None:
        """Validate CSR arrays before a source may be marked real.

        Any failure raises ``ValueError`` so the caller falls back to a
        synthetic scaffold and provenance stays synthetic.
        """
        if row_ptr.ndim != 1 or col_idx.ndim != 1 or weights.ndim != 1:
            raise ValueError("CSR arrays must be 1-D")
        if row_ptr.shape[0] < 2:
            raise ValueError("row_ptr must have length num_neurons+1 >= 2")
        n = int(row_ptr.shape[0] - 1)
        if n <= 0:
            raise ValueError("CSR implies non-positive num_neurons")
        if int(row_ptr[0]) != 0:
            raise ValueError("row_ptr must start at 0")
        # Monotonic non-decreasing (no negative degrees).
        if bool((np.diff(row_ptr) < 0).any()):
            raise ValueError("row_ptr must be non-decreasing")
        e = int(row_ptr[-1])
        if e < 0:
            raise ValueError("CSR edge count must be non-negative")
        if int(col_idx.shape[0]) != e:
            raise ValueError(
                f"col_idx length {col_idx.shape[0]} != row_ptr total {e}"
            )
        if int(weights.shape[0]) != e:
            raise ValueError(
                f"weights length {weights.shape[0]} != row_ptr total {e}"
            )
        if e > 0:
            cmin = int(col_idx.min())
            cmax = int(col_idx.max())
            if cmin < 0 or cmax >= n:
                raise ValueError(
                    f"col_idx out of bounds [{cmin}, {cmax}] for n={n}"
                )
        if not np.all(np.isfinite(row_ptr)):
            raise ValueError("row_ptr must be finite")
        if e > 0 and not bool(np.all(np.isfinite(weights))):
            raise ValueError("weights must be finite")

    def _build_csr_from_edgelist(
        self, sources: np.ndarray, targets: np.ndarray, weights: np.ndarray
    ) -> None:
        if sources.shape[0] != targets.shape[0] or targets.shape[0] != weights.shape[0]:
            raise ValueError("Edgelist sources/targets/weights length mismatch")
        n = self.config.num_neurons
        if sources.size:
            if int(sources.min()) < 0 or int(targets.min()) < 0:
                raise ValueError("Edgelist node ids must be non-negative")
            if sources.max() >= n or targets.max() >= n:
                raise ValueError("Edgelist node ids exceed configured num_neurons")
        if not bool(np.all(np.isfinite(weights))):
            raise ValueError("Edgelist weights must be finite")
        counts = np.bincount(sources, minlength=n).astype(np.int64)
        row_ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=row_ptr[1:])
        order = np.argsort(sources, kind="stable")
        col_idx = np.asarray(targets[order], dtype=np.int32)
        w = np.asarray(weights[order], dtype=np.float32)
        self._allocate_arrays(row_ptr, col_idx, w)

    # -- dynamics (vectorized) ----------------------------------------------

    def _require_built(self) -> None:
        if not self.built or self.row_ptr is None or self.col_idx is None:
            raise RuntimeError("Call build() before stepping the simulation.")

    def reset_dynamics(self, clear_history: bool = False) -> None:
        """Reset volatile dynamics (voltage/activity) to zeros.

        Connectivity is preserved. Used by benchmarks to make repeats fair:
        each repeat starts from the same dynamical state instead of drifting
        on prior spiking history. ``spike_counts``/``current_step`` are kept
        unless ``clear_history`` is set.
        """
        n = self.config.num_neurons
        # Re-allocate if a real source adopted a different neuron count.
        if self.voltage.shape[0] != n:
            n = int(self.voltage.shape[0])
        self.voltage = np.zeros(n, dtype=np.float32)
        self.activity = np.zeros(n, dtype=np.float32)
        if clear_history:
            self.current_step = 0
            self.activity_history.clear()
            self.event_log.clear()
            self.snapshots.clear()

    def _append_event(self, event: Dict[str, Any]) -> None:
        self.event_log.append(event)
        # Cap in-memory log (keep most recent).
        overflow = len(self.event_log) - MAX_EVENT_LOG
        if overflow > 0:
            del self.event_log[:overflow]

    def _append_history(self, spikes: int) -> None:
        self.activity_history.append(spikes)
        overflow = len(self.activity_history) - MAX_ACTIVITY_HISTORY
        if overflow > 0:
            del self.activity_history[:overflow]

    def step(self, external_drive: Optional[np.ndarray] = None) -> int:
        """Advance one vectorized LIF step. Returns the spike count."""
        self._require_built()
        assert self.row_ptr is not None and self.col_idx is not None
        assert self.weights is not None

        n = self.num_neurons
        if external_drive is None:
            drive = self.rng.normal(0.0, self.config.input_gain, size=n).astype(
                np.float32
            )
        else:
            drive = np.asarray(external_drive, dtype=np.float32).reshape(n)

        # Sparse synaptic current: for each presynaptic spike, add its row's
        # weights to the postsynaptic targets. Fully vectorized via repeat +
        # scatter-add (no Python loops over neurons/edges).
        spiking_prev = self.activity.astype(np.bool_)
        syn_current = np.zeros(n, dtype=np.float32)
        if spiking_prev.any():
            row_ptr = self.row_ptr
            counts = (row_ptr[1:] - row_ptr[:-1]).astype(np.int64)
            firing_rows = np.flatnonzero(spiking_prev)
            firing_counts = counts[firing_rows]
            total_firing_edges = int(firing_counts.sum())
            if total_firing_edges > 0:
                starts = row_ptr[firing_rows]
                # Flat positions of outgoing edges for all firing neurons.
                edge_pos = np.empty(total_firing_edges, dtype=np.int64)
                cursor = 0
                for c, s in zip(firing_counts.tolist(), starts.tolist()):
                    c = int(c)
                    edge_pos[cursor : cursor + c] = np.arange(
                        int(s), int(s) + c, dtype=np.int64
                    )
                    cursor += c
                post = self.col_idx[edge_pos].astype(np.int64)
                w = self.weights[edge_pos]
                np.add.at(syn_current, post, w)

        # Leaky integrate-and-fire update (vectorized float32).
        self.voltage = (
            self.voltage * np.float32(self.config.decay)
            + syn_current
            + drive.astype(np.float32)
        )
        fired = self.voltage >= self.threshold
        n_spikes = int(fired.sum())
        self.spike_counts[fired] += 1
        self.voltage[fired] = np.float32(self.config.reset)
        self.activity = fired.astype(np.float32)

        self.current_step += 1
        self._append_history(n_spikes)
        # Slice the index array *before* materializing a Python list so a
        # dense firing step cannot allocate a full-population list.
        fired_idx = np.flatnonzero(fired)
        sample = fired_idx[:SAMPLE_EVENTS_K].tolist()
        self._append_event(
            {
                "type": "step",
                "step": self.current_step,
                "spikes": n_spikes,
                "sample_events": sample,
            }
        )
        return n_spikes

    def run(self, steps: int = 10) -> Dict[str, Any]:
        """Run ``steps`` vectorized updates and record a snapshot."""
        self._require_built()
        if steps <= 0:
            raise ValueError("steps must be positive")
        for _ in range(steps):
            self.step()
        return self.record_snapshot()

    def record_snapshot(self) -> Dict[str, Any]:
        """Record a deterministic activity snapshot of the current state."""
        active_idx = np.flatnonzero(self.activity)
        snap = ActivitySnapshot(
            step=self.current_step,
            spikes=int(self.activity_history[-1]) if self.activity_history else 0,
            mean_voltage=float(np.mean(self.voltage)) if self.voltage.size else 0.0,
            max_voltage=float(np.max(self.voltage)) if self.voltage.size else 0.0,
            sample_events=active_idx[:SAMPLE_EVENTS_K].tolist(),
        )
        self.snapshots.append(snap)
        overflow = len(self.snapshots) - MAX_SNAPSHOTS
        if overflow > 0:
            del self.snapshots[:overflow]
        return asdict(snap)

    # -- metadata / persistence ----------------------------------------------

    def summary(self) -> Dict[str, Any]:
        mem_mb = self.estimate_memory_mb()
        return {
            "num_neurons": self.num_neurons,
            "num_edges": self.num_edges,
            "is_synthetic": self.is_synthetic,
            "data_source": self.data_source,
            "full_malecns_scale": self.is_full_malecns_scale,
            "target_neurons": N_MALECNS_NEURONS,
            "target_edges": N_MALECNS_EDGES,
            "dtypes": {
                "row_ptr": str(self.row_ptr.dtype) if self.row_ptr is not None else None,
                "col_idx": str(self.col_idx.dtype) if self.col_idx is not None else None,
                "weights": str(self.weights.dtype) if self.weights is not None else None,
                "voltage": str(self.voltage.dtype),
            },
            "estimated_memory_mb": mem_mb,
            "steps_run": self.current_step,
            "total_spikes": int(self.spike_counts.sum()),
            "use_memmap": self.config.use_memmap,
            "cache_dir": str(self.config.cache_dir)
            if self.config.cache_dir
            else None,
            "seed": self.config.seed,
        }

    def estimate_memory_mb(self) -> float:
        total_bytes = (
            self.voltage.nbytes + self.threshold.nbytes + self.spike_counts.nbytes
        )
        for arr in (self.row_ptr, self.col_idx, self.weights):
            if arr is not None:
                total_bytes += int(arr.nbytes)
        return round(total_bytes / (1024.0**2), 3)

    def save_snapshot(self, path: str | Path) -> Path:
        """Persist summary + activity history + events as JSON."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.summary(),
            "activity_history": list(self.activity_history),
            "snapshots": [asdict(s) for s in self.snapshots],
            "events": self.event_log[-512:],
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        import json

        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return dest


__all__ = [
    "N_MALECNS_NEURONS",
    "N_MALECNS_EDGES",
    "SYNTHETIC_MARKER",
    "REAL_MARKER_PREFIX",
    "MAX_EVENT_LOG",
    "MAX_ACTIVITY_HISTORY",
    "MAX_SNAPSHOTS",
    "SAMPLE_EVENTS_K",
    "SimulationConfig",
    "ActivitySnapshot",
    "MaleCNSSimulation",
]
