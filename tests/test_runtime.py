"""Tests for tensorfly.runtime — must run without a GPU / torch."""

from __future__ import annotations

import json
from pathlib import Path

from tensorfly.runtime import PROFILES, get_runtime, select_profile


def test_profiles_exist_for_a100_l4_t4_and_cpu():
    assert set(["A100", "L4", "T4", "CPU"]).issubset(set(PROFILES.keys()))
    assert PROFILES["A100"].batch_size >= PROFILES["L4"].batch_size >= PROFILES["T4"].batch_size


def test_select_profile_matches_known_gpus():
    assert select_profile("NVIDIA A100-SXM4-40GB", True).name == "A100"
    assert select_profile("nvidia l4", True).name == "L4"
    assert select_profile("Tesla T4", True).name == "T4"


def test_select_profile_unknown_gpu_falls_back_to_t4_with_print(capsys):
    profile = select_profile("Mystery GPU 9000", True)
    assert profile.name == "T4"
    out = capsys.readouterr().out
    assert "Falling back" in out and "T4" in out


def test_select_profile_no_gpu_falls_back_to_cpu_with_print(capsys):
    profile = select_profile("", False)
    assert profile.name == "CPU"
    out = capsys.readouterr().out
    assert "Falling back" in out and "CPU" in out


def test_get_runtime_runs_without_gpu_and_records_metadata(tmp_path):
    rt = get_runtime(
        gpu_name_override="",
        gpu_present_override=False,
        extra={"test": True},
    )
    assert rt.profile_name == "CPU"
    assert rt.device == "cpu"

    dest = tmp_path / "meta" / "runtime.json"
    saved = rt.save_metadata(dest)
    assert Path(saved).exists()
    payload = json.loads(Path(saved).read_text(encoding="utf-8"))
    assert payload["profile"]["name"] == "CPU"
    assert "gpu" in payload and "system" in payload
    assert payload["extra"]["test"] is True
    # Detection fields are present even without torch/CUDA.
    assert "torch_cuda_available" in payload["gpu"]
