"""Tests for tensorfly.simulation — CPU-only, small synthetic graphs."""

from __future__ import annotations

import json

import numpy as np

from tensorfly.simulation import (
    N_MALECNS_EDGES,
    N_MALECNS_NEURONS,
    SYNTHETIC_MARKER,
    MaleCNSSimulation,
    SimulationConfig,
)


def test_full_network_target_constants():
    assert N_MALECNS_NEURONS == 166_700
    assert N_MALECNS_EDGES == 25_600_000


def test_synthetic_scaffold_is_clearly_marked_and_compact():
    sim = MaleCNSSimulation(SimulationConfig(num_neurons=256, num_edges=1024, seed=0))
    sim.build()
    assert sim.built
    assert sim.is_synthetic is True
    assert sim.data_source == SYNTHETIC_MARKER
    assert "NOT real MaleCNS" in sim.data_source
    # CSR shapes + compact dtypes.
    assert sim.row_ptr is not None and sim.col_idx is not None
    assert sim.row_ptr.shape == (257,)
    assert sim.col_idx.shape[0] == 1024
    assert sim.row_ptr.dtype == np.int64
    assert sim.col_idx.dtype == np.int32
    assert sim.weights is not None and sim.weights.dtype == np.float32
    assert sim.voltage.dtype == np.float32
    assert sim.summary()["is_synthetic"] is True


def test_deterministic_build_and_run():
    cfg = SimulationConfig(num_neurons=128, num_edges=512, seed=123)
    a = MaleCNSSimulation(cfg).build()
    b = MaleCNSSimulation(SimulationConfig(num_neurons=128, num_edges=512, seed=123)).build()
    assert np.array_equal(a.col_idx, b.col_idx)
    assert np.array_equal(a.weights, b.weights)

    ra = a.run(steps=5)
    rb = b.run(steps=5)
    assert a.activity_history == b.activity_history
    assert ra["spikes"] == rb["spikes"]
    assert len(a.snapshots) == 1
    assert a.snapshots[0].step == 5


def test_vectorized_step_records_events_and_snapshot(tmp_path):
    sim = MaleCNSSimulation(SimulationConfig(num_neurons=64, num_edges=256, seed=7))
    sim.build()
    spikes = sim.step()
    assert isinstance(spikes, int) and spikes >= 0
    assert len(sim.activity_history) == 1
    assert sim.event_log[-1]["type"] == "step"

    snap = sim.record_snapshot()
    assert snap["step"] == 1

    dest = tmp_path / "snap.json"
    sim.save_snapshot(dest)
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["summary"]["num_neurons"] == 64
    assert payload["activity_history"] == sim.activity_history


def test_real_npz_source_is_used_and_marked_real(tmp_path):
    n, e = 32, 96
    rng = np.random.default_rng(0)
    row_ptr = np.zeros(n + 1, dtype=np.int64)
    row_ptr[1:] = np.cumsum(np.full(n, e // n, dtype=np.int64))
    col_idx = rng.integers(0, n, size=e).astype(np.int32)
    weights = np.ones(e, dtype=np.float32)
    src = tmp_path / "edges.npz"
    np.savez(src, row_ptr=row_ptr, col_idx=col_idx, weights=weights)

    sim = MaleCNSSimulation(
        SimulationConfig(num_neurons=n, num_edges=e, seed=0), edge_source=src
    ).build()
    assert sim.is_synthetic is False
    assert sim.data_source.startswith("real-source:")
    assert sim.num_edges == e
    sim.run(steps=2)
    assert len(sim.activity_history) == 2


def test_memmap_cache_option(tmp_path):
    sim = MaleCNSSimulation(
        SimulationConfig(
            num_neurons=64,
            num_edges=256,
            seed=0,
            cache_dir=tmp_path / "cache",
            use_memmap=True,
        )
    ).build()
    assert sim.row_ptr is not None
    assert (tmp_path / "cache").exists()
    sim.run(steps=2)
    assert len(sim.activity_history) == 2
