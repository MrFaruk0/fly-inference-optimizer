"""Regression tests for reviewer findings (1)-(10). CPU-only, no GPU/deps."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from tensorfly import inference as inf_mod
from tensorfly.benchmark import BenchmarkConfig, compare_configs, run_benchmark
from tensorfly.inference import InferenceConfig, QwenInference
from tensorfly.replay import ReplayRecorder
from tensorfly.runtime import get_runtime
from tensorfly.simulation import MaleCNSSimulation, SimulationConfig


# (10) default trust_remote_code False unless explicit.
def test_trust_remote_code_defaults_false():
    assert InferenceConfig().trust_remote_code is False
    assert QwenInference(profile_name="CPU", model_id="m").trust_remote_code is False
    assert QwenInference(profile_name="CPU", model_id="m").metadata["trust_remote_code"] is False
    explicit = QwenInference(profile_name="CPU", model_id="m", trust_remote_code=True)
    assert explicit.trust_remote_code is True
    cfg = InferenceConfig(trust_remote_code=True)
    via_cfg = QwenInference(config=cfg)
    assert via_cfg.trust_remote_code is True


# (6) InferenceConfig override precedence with Optional sentinels.
def test_inference_config_precedence_explicit_wins():
    cfg = InferenceConfig(
        profile_name="A100", max_new_tokens=64, temperature=0.7,
        do_sample=True, trust_remote_code=True,
    )
    inf = QwenInference(config=cfg)
    assert inf.max_new_tokens == 64
    assert inf.temperature == 0.7
    assert inf.do_sample is True
    assert inf.trust_remote_code is True
    # Explicit falsy values must win over truthy config.
    inf2 = QwenInference(
        config=cfg, max_new_tokens=32, temperature=0.0,
        do_sample=False, trust_remote_code=False,
    )
    assert inf2.max_new_tokens == 32
    assert inf2.temperature == 0.0
    assert inf2.do_sample is False
    assert inf2.trust_remote_code is False
    # Omitted kwargs fall back to config/defaults.
    inf3 = QwenInference(config=cfg, model_id="explicit-model")
    assert inf3.max_new_tokens == 64
    assert inf3.model_id == "explicit-model"


# (1) multimodal loader preferred; fallback only when unavailable; IDs retained.
def _install_fake_transformers(multimodal: bool):
    fake = types.ModuleType("transformers")
    calls = {}

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id, trust_remote_code=False):
            calls["processor_model"] = model_id
            calls["processor_trust"] = trust_remote_code
            return FakeProcessor()

        def __call__(self, prompt, return_tensors=None):
            return {"input_ids": [1, 2]}

        def decode(self, ids, skip_special_tokens=True):
            return "ok"

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, trust_remote_code=False):
            calls["tokenizer_model"] = model_id
            return FakeTokenizer()

    class FakeMultiModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["model_cls"] = "multi"
            calls["model_id"] = model_id
            calls["kwargs"] = kwargs
            return object()

    class FakeCausalModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["model_cls"] = "causal"
            calls["model_id"] = model_id
            calls["kwargs"] = kwargs
            return object()

    fake.AutoTokenizer = FakeTokenizer
    if multimodal:
        fake.AutoProcessor = FakeProcessor
        fake.AutoModelForImageTextToText = FakeMultiModel
    else:
        # No multimodal loader at all: explicit fallback expected.
        fake.AutoModelForCausalLM = FakeCausalModel
    sys.modules["transformers"] = fake
    return calls


def test_load_prefers_multimodal_and_keeps_exact_ids():
    calls = _install_fake_transformers(multimodal=True)
    try:
        inf = QwenInference(profile_name="A100", vram_gb=40.0)
        # Exact requested IDs retained.
        assert inf.model_id == "Qwen/Qwen3.5-9B"
        inf.load()
        assert inf._processor_kind == "processor"
        assert inf._loader_kind == "AutoModelForImageTextToText"
        assert calls["processor_model"] == "Qwen/Qwen3.5-9B"
        assert calls["model_id"] == "Qwen/Qwen3.5-9B"
        assert inf.metadata["loader"] == "AutoModelForImageTextToText"
    finally:
        sys.modules.pop("transformers", None)


def test_load_falls_back_only_when_multimodal_unavailable(capsys):
    calls = _install_fake_transformers(multimodal=False)
    try:
        inf = QwenInference(profile_name="CPU", model_id="Qwen/Qwen3.5-4B")
        inf.load()
        assert inf._loader_kind == "AutoModelForCausalLM-fallback"
        assert calls["model_cls"] == "causal"
        assert calls["model_id"] == "Qwen/Qwen3.5-4B"
        out = capsys.readouterr().out
        assert "falling back" in out.lower()
    finally:
        sys.modules.pop("transformers", None)


# (2) corrupt NPZ stays synthetic.
def test_corrupt_npz_stays_synthetic(tmp_path):
    n, e = 16, 48
    # Non-monotonic row_ptr + out-of-bounds col_idx.
    row_ptr = np.zeros(n + 1, dtype=np.int64)
    row_ptr[1:] = np.arange(1, n + 1) * 3
    row_ptr[5] = row_ptr[4] - 2  # break monotonicity
    col_idx = np.full(e, 9999, dtype=np.int32)  # out of bounds
    weights = np.ones(e, dtype=np.float32)
    src = tmp_path / "bad.npz"
    np.savez(src, row_ptr=row_ptr, col_idx=col_idx, weights=weights)
    sim = MaleCNSSimulation(SimulationConfig(num_neurons=n, num_edges=e, seed=0), edge_source=src).build()
    assert sim.is_synthetic is True
    assert "NOT real MaleCNS" in sim.data_source
    assert any(ev.get("type") == "source-load-failed" for ev in sim.event_log)


def test_mismatched_lengths_stay_synthetic(tmp_path):
    n = 8
    row_ptr = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16], dtype=np.int64)
    col_idx = np.zeros(10, dtype=np.int32)  # should be 16
    weights = np.ones(16, dtype=np.float32)
    src = tmp_path / "bad2.npz"
    np.savez(src, row_ptr=row_ptr, col_idx=col_idx, weights=weights)
    sim = MaleCNSSimulation(SimulationConfig(num_neurons=n, num_edges=16, seed=0), edge_source=src).build()
    assert sim.is_synthetic is True


# (3) is_full_malecns_scale uses actual counts.
def test_full_scale_uses_actual_counts(tmp_path):
    n, e = 32, 96
    rng = np.random.default_rng(0)
    row_ptr = np.zeros(n + 1, dtype=np.int64)
    row_ptr[1:] = np.cumsum(np.full(n, e // n, dtype=np.int64))
    col_idx = rng.integers(0, n, size=e).astype(np.int32)
    weights = np.ones(e, dtype=np.float32)
    src = tmp_path / "edges.npz"
    np.savez(src, row_ptr=row_ptr, col_idx=col_idx, weights=weights)
    # Config claims full scale but actual source is tiny -> must be False.
    from tensorfly.simulation import N_MALECNS_EDGES, N_MALECNS_NEURONS

    sim = MaleCNSSimulation(
        SimulationConfig(num_neurons=N_MALECNS_NEURONS, num_edges=N_MALECNS_EDGES, seed=0),
        edge_source=src,
    ).build()
    assert sim.num_edges == e
    assert sim.is_full_malecns_scale is False


# (4) benchmark reset/fair + drift + configs affect factory.
def test_benchmark_reports_drift_and_resets():
    sim = MaleCNSSimulation(SimulationConfig(num_neurons=64, num_edges=256, seed=0)).build()
    res = run_benchmark(sim, steps=3, repeats=3, warmup=1)
    assert "repeat_spikes" in res.metadata
    assert "drift" in res.metadata
    assert res.metadata["reset_between_repeats"] is True
    assert len(res.metadata["repeat_spikes"]) == 3
    assert res.total_spikes == sum(res.metadata["repeat_spikes"])


def test_compare_configs_factory_receives_cfg():
    seen = []

    def factory(cfg=None):
        if isinstance(cfg, BenchmarkConfig):
            seen.append(cfg.label)
        return MaleCNSSimulation(SimulationConfig(num_neurons=32, num_edges=64, seed=0)).build()

    cfgs = [BenchmarkConfig(steps=2, repeats=1, warmup=0, label="a"),
            BenchmarkConfig(steps=2, repeats=1, warmup=0, label="b")]
    results = compare_configs(factory, cfgs)
    assert len(results) == 2
    assert seen == ["a", "b"]


# (5) caps + slice before list.
def test_event_log_capped_and_samples_sliced(monkeypatch):
    import tensorfly.simulation as sim_mod

    monkeypatch.setattr(sim_mod, "MAX_EVENT_LOG", 8)
    sim = MaleCNSSimulation(SimulationConfig(num_neurons=32, num_edges=64, seed=1)).build()
    for _ in range(20):
        sim.step()
    assert len(sim.event_log) <= 8
    for ev in sim.event_log:
        if ev.get("type") == "step":
            assert len(ev["sample_events"]) <= 16
    snap = sim.record_snapshot()
    assert len(snap["sample_events"]) <= 16


# (7) startup_summary must not swallow fallback notice.
def test_startup_summary_emits_fallback(capsys):
    rt = get_runtime(gpu_name_override="", gpu_present_override=False)
    text = rt.startup_summary()
    out = capsys.readouterr().out
    assert "TensorFly fallback" in out
    assert "Qwen3.5-4B" in text


# (8) replay metadata + clamp + event index validation.
def test_replay_meta_includes_provenance():
    rec = ReplayRecorder(model="m", prompt="p")
    rec.record_snapshot(t=0.0, sensory=[0.1, 0.2], dopamine=[0.3], controller=[0.4])
    doc = rec.to_dict()
    assert "is_synthetic" in doc["meta"]
    assert "data_source" in doc["meta"]
    assert "provenance" in doc["meta"]
    assert doc["meta"]["provenance"]["data_source"] == doc["meta"]["data_source"]


def test_replay_activity_clamped():
    rec = ReplayRecorder(model="m")
    frame = rec.record_snapshot(
        t=0.0, sensory=[-5.0, 0.5, 999.0, float("nan"), float("inf")],
        dopamine=[0.2], controller=[0.3],
    )
    assert frame["sensory"] == [0.0, 0.5, 1.0, 0.0, 0.0]
    frame2 = rec.record_snapshot(t=1.0, dopamine_mean=float("inf"), sensory=[0.1], dopamine=None, controller=[0.1])
    # dopamine present? frame2 has sensory+controller so dopamine_mean stored
    assert frame2["dopamine_mean"] == 0.0
    with pytest.raises(ValueError):
        rec.record_snapshot(t=float("nan"), sensory=[0.1], dopamine=[0.1], controller=[0.1])


def test_replay_event_index_validated():
    rec = ReplayRecorder(model="m")
    rec.record_snapshot(t=0.0, sensory=[0.1], dopamine=[0.1], controller=[0.1])
    rec.record_snapshot(t=1.0, sensory=[0.2], dopamine=[0.2], controller=[0.2])
    with pytest.raises(IndexError):
        rec.record_event("config_change", "x", frame_index=99)
    with pytest.raises(IndexError):
        rec.record_event("config_change", "x", frame_index=-1)


# (9) memmap filenames unique across shapes.
def test_memmap_filenames_unique(tmp_path):
    a = MaleCNSSimulation(SimulationConfig(num_neurons=32, num_edges=64, seed=0,
                                           cache_dir=tmp_path / "cache", use_memmap=True)).build()
    b = MaleCNSSimulation(SimulationConfig(num_neurons=64, num_edges=128, seed=0,
                                           cache_dir=tmp_path / "cache", use_memmap=True)).build()
    names_a = {p.name for p in a._memmap_files}
    names_b = {p.name for p in b._memmap_files}
    assert names_a.isdisjoint(names_b)
