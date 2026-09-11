"""Tests for tensorfly.replay — CPU-only, no GPU/transformers."""

from __future__ import annotations

import json

from tensorfly.replay import REPLAY_SCHEMA, ReplayRecorder


def _small_pops():
    return {"sensory": 4, "dopamine": 2, "controller": 2}


def test_replay_schema_and_meta_fields():
    rec = ReplayRecorder(
        model="fly-qwen-test",
        prompt="describe the optic lobe response",
        populations=_small_pops(),
        config={"id": "cfg-A", "lr": 0.001},
    )
    rec.record_snapshot(
        t=0.0,
        sensory=[0.1, 0.2, 0.3, 0.4],
        dopamine=[0.5, 0.6],
        controller=[0.7, 0.8],
        trial=0,
        tokens=3,
        throughput_tps=36.0,
        ttft_ms=121.5,
        decode_ms_per_token=23.3,
        reward=0.5,
        config_id="cfg-A",
        config={"id": "cfg-A", "lr": 0.001},
    )
    doc = rec.to_dict()
    assert doc["schema"] == "fly-cns-replay/1"
    assert REPLAY_SCHEMA == "fly-cns-replay/1"
    assert doc["meta"]["model"] == "fly-qwen-test"
    assert doc["meta"]["prompt"] == "describe the optic lobe response"
    assert doc["meta"]["populations"] == _small_pops()
    assert doc["meta"]["config"]["id"] == "cfg-A"
    assert "best" in doc["meta"]
    frame = doc["frames"][0]
    # Viewer-required snapshot/inference/config fields.
    for key in (
        "t",
        "trial",
        "tokens",
        "throughput_tps",
        "ttft_ms",
        "decode_ms_per_token",
        "reward",
        "config_id",
        "config",
        "sensory",
        "dopamine",
        "controller",
    ):
        assert key in frame


def test_replay_events_and_best_tracking():
    rec = ReplayRecorder(model="m", prompt="p", populations=_small_pops())
    rec.record_snapshot(
        t=0.0, sensory=[0.1] * 4, dopamine=[0.2] * 2, controller=[0.3] * 2,
        reward=0.4, config_id="cfg-A",
    )
    rec.record_inference(
        t=0.25, sensory=[0.2] * 4, dopamine=[0.3] * 2, controller=[0.4] * 2,
        tokens=10, throughput_tps=38.0, ttft_ms=120.0,
        decode_ms_per_token=24.0, reward=0.9, config_id="cfg-B",
        config={"id": "cfg-B"},
    )
    rec.record_event("config_change", "cfg-A -> cfg-B")
    doc = rec.to_dict()
    assert doc["frames"][1]["event"] == {
        "type": "config_change",
        "label": "cfg-A -> cfg-B",
    }
    assert doc["meta"]["best"] == {"config": "cfg-B", "reward": 0.9}


def test_replay_frames_sorted_and_save_roundtrip(tmp_path):
    rec = ReplayRecorder(model="m", populations=_small_pops())
    rec.record_snapshot(
        t=1.0, sensory=[0.5] * 4, dopamine=[0.5] * 2, controller=[0.5] * 2,
    )
    rec.record_snapshot(
        t=0.0, sensory=[0.1] * 4, dopamine=[0.1] * 2, controller=[0.1] * 2,
    )
    doc = rec.to_dict()
    assert [f["t"] for f in doc["frames"]] == [0.0, 1.0]

    dest = rec.save(tmp_path / "run" / "replay.json")
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["schema"] == "fly-cns-replay/1"
    assert len(payload["frames"]) == 2
    assert payload["meta"]["populations"]["sensory"] == 4
