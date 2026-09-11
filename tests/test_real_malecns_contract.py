"""Lightweight contracts; no Janelia download and no model execution."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorfly.dataset import DatasetPreparationError, MaleCNSDataset, SOURCE_SPECS, prepare_malecns, verify_source
from tensorfly.inference import _TokenTimingStreamer
from tensorfly.populations import build_population_registry
from tensorfly.simulation import MaleCNSSimulation


def fixture_dataset(tmp_path: Path) -> MaleCNSDataset:
    ids = np.asarray([17, 2**54 + 11, 2**54 + 99, 2**54 + 201], dtype=np.uint64)
    annotations = [
        {"body_id": int(ids[0]), "superclass": "sensory", "type": "R7", "neurotransmitter": "acetylcholine"},
        {"body_id": int(ids[1]), "superclass": "central", "type": "PAM11", "neurotransmitter": "dopamine"},
        {"body_id": int(ids[2]), "superclass": "descending", "type": "DNa02", "neurotransmitter": "acetylcholine"},
        {"body_id": int(ids[3]), "superclass": "central", "type": "other", "neurotransmitter": "gaba"},
    ]
    return MaleCNSDataset(ids, np.asarray([0, 1, 2], np.uint32), np.asarray([1, 2, 3], np.uint32), np.asarray([4, 2, 1], np.uint32), annotations, [], {"release": "MaleCNS v1.0", "retention_policy": "fixture real schema", "synthetic_dev": False}, tmp_path)


def test_official_sources_are_pinned_and_large_graph_count_is_exact():
    assert SOURCE_SPECS["edges.feather"]["bytes"] == 1_051_241_946
    assert len(SOURCE_SPECS["edges.feather"]["sha256"]) == 64
    from tensorfly.simulation import N_MALECNS_EDGES
    assert N_MALECNS_EDGES == 25_582_938


def test_hash_verification_is_fail_closed(tmp_path):
    path = tmp_path / "x"; path.write_bytes(b"bad")
    with pytest.raises(DatasetPreparationError):
        verify_source(path, SOURCE_SPECS["annotations.feather"])


def test_normal_prepare_never_switches_to_synthetic(monkeypatch, tmp_path):
    import tensorfly.dataset as module
    monkeypatch.setattr(module, "_download_verified", lambda *a, **k: (_ for _ in ()).throw(DatasetPreparationError("offline")))
    with pytest.raises(DatasetPreparationError, match="offline"):
        prepare_malecns(tmp_path)
    assert prepare_malecns(tmp_path, synthetic_dev=True).report["synthetic_dev"] is True


def test_uint64_mapping_and_annotation_populations_are_real_and_reversible(tmp_path):
    dataset = fixture_dataset(tmp_path)
    assert dataset.index_for_body_id(str(2**54 + 99)) == 2
    assert dataset.body_id_for_index(1) == 2**54 + 11
    registry = build_population_registry(dataset)
    assert registry.sensory_body_ids.tolist() == [17]
    assert registry.dopamine_body_ids.tolist() == [2**54 + 11]
    assert registry.controller_body_ids.tolist() == [2**54 + 99]


def test_default_external_drive_is_zero_not_gaussian(tmp_path):
    simulation = MaleCNSSimulation(fixture_dataset(tmp_path))
    simulation.step()
    assert not simulation.activity.any()
    assert np.allclose(simulation.voltage, 0)


def test_ttft_streamer_stops_on_first_generated_token_not_end():
    now = iter([1.0, 3.0, 9.0])
    streamer = _TokenTimingStreamer(3, clock=lambda: next(now))
    streamer.put([1, 2, 3])  # prompt
    streamer.put([4]); first = streamer.first_token_at
    streamer.put([5]); streamer.end()
    assert first == 1.0 and streamer.end_at == 9.0 and streamer.generated_tokens == 2


def test_qwen_loader_uses_left_padding_and_current_dtype_keyword():
    source = Path("src/tensorfly/inference.py").read_text(encoding="utf-8")
    assert 'padding_side = "left"' in source
    assert '"dtype": getattr(torch, self.config.dtype)' in source
    assert '"torch_dtype": getattr(torch, self.config.dtype)' not in source


def test_production_viewer_rejects_procedural_geometry():
    app = Path("viewer/app.js").read_text(encoding="utf-8")
    assert "Math.random" not in app and "Gaussian" not in app
    assert "tensorfly-real-morphology/1" in app
    assert "String(id)" in app
