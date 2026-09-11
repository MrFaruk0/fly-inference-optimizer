"""Tests for tensorfly.inference + runtime additions — no transformers/GPU."""

from __future__ import annotations

import sys

from tensorfly.inference import (
    FALLBACK_MESSAGE,
    MODEL_4B,
    MODEL_9B,
    QwenInference,
    resolve_dtype,
    select_qwen_model,
)
from tensorfly.runtime import PROFILES, get_runtime


def test_a100_and_l4_default_to_9b_without_fallback_print(capsys):
    assert select_qwen_model("A100", 40.0) == MODEL_9B
    assert select_qwen_model("L4", 24.0) == MODEL_9B
    assert select_qwen_model("A100", None) == MODEL_9B
    assert select_qwen_model("L4", None) == MODEL_9B
    out = capsys.readouterr().out
    assert out == ""
    assert MODEL_9B == "Qwen/Qwen3.5-9B"


def test_t4_and_cpu_default_to_4b_with_exact_fallback_print(capsys):
    assert select_qwen_model("T4", 16.0) == MODEL_4B
    out = capsys.readouterr().out.strip()
    assert out == FALLBACK_MESSAGE

    assert select_qwen_model("CPU", None) == MODEL_4B
    out = capsys.readouterr().out.strip()
    assert out == FALLBACK_MESSAGE
    assert MODEL_4B == "Qwen/Qwen3.5-4B"


def test_insufficient_vram_falls_back_to_4b_with_exact_print(capsys):
    assert select_qwen_model("A100", 8.0) == MODEL_4B
    assert capsys.readouterr().out.strip() == FALLBACK_MESSAGE
    assert (
        FALLBACK_MESSAGE
        == "TensorFly fallback: Qwen3.5-4B because current GPU memory "
        "is insufficient for the 9B benchmark profile."
    )

    assert select_qwen_model("L4", 12.0) == MODEL_4B
    assert capsys.readouterr().out.strip() == FALLBACK_MESSAGE


def test_resolve_dtype_defaults():
    assert resolve_dtype("A100") == "bfloat16"
    assert resolve_dtype("L4") == "float16"
    assert resolve_dtype("T4") == "float16"
    assert resolve_dtype("CPU") == "float32"


def test_inference_import_is_lazy():
    # Importing the module must not pull transformers.
    assert "transformers" not in sys.modules


def test_qwen_inference_metadata_without_backend(capsys):
    inf = QwenInference(profile_name="A100", vram_gb=40.0)
    assert inf.model_id == MODEL_9B
    assert inf.dtype_name == "bfloat16"
    meta = inf.metadata
    assert meta["model"] == MODEL_9B
    assert meta["profile"] == "A100"
    assert meta["dtype"] == "bfloat16"
    assert "device" in meta
    capsys.readouterr()  # drain any fallback prints

    inf_small = QwenInference(profile_name="T4", vram_gb=16.0)
    assert inf_small.model_id == MODEL_4B
    assert inf_small.dtype_name == "float16"
    assert capsys.readouterr().out.strip() == FALLBACK_MESSAGE


def test_explicit_model_override_skips_fallback_print(capsys):
    inf = QwenInference(profile_name="CPU", model_id="my-org/custom")
    assert inf.model_id == "my-org/custom"
    assert capsys.readouterr().out == ""


def test_generate_with_metrics_uses_fake_backend_without_torch():
    inf = QwenInference(profile_name="CPU", model_id="fake-model")
    # dtype/device resolution must not need torch.

    class FakeTokenizer:
        def __call__(self, prompt, return_tensors=None):
            return {"input_ids": [1, 2, 3]}

        def decode(self, ids, skip_special_tokens=True):
            return "hello world"

    class FakeModel:
        def generate(self, **kwargs):
            # 3 input ids + 4 new ids.
            return [1, 2, 3, 4, 5, 6, 7]

    inf._tokenizer = FakeTokenizer()
    inf._model = FakeModel()
    inf._loaded = True
    out = inf.generate_with_metrics("hi", max_new_tokens=4)
    assert out["tokens"] == 4
    assert out["text"] == "hello world"
    assert out["throughput_tps"] > 0
    assert out["ttft_ms"] >= 0
    assert out["decode_ms_per_token"] >= 0
    assert out["model"] == "fake-model"


def test_runtime_a100_dtype_is_bfloat16():
    assert PROFILES["A100"].dtype == "bfloat16"


def test_runtime_startup_summary_prints_required_fields(capsys):
    rt = get_runtime(
        gpu_name_override="NVIDIA A100-SXM4-40GB",
        gpu_present_override=True,
        extra={},
    )
    # Force deterministic VRAM so summary is stable without drivers.
    rt.gpu.vram_gb = 40.0
    rt.system.ram_gb = 64.0
    rt.gpu.cuda_version = "12.4"
    rt.gpu.torch_version = "2.3.0"
    text = rt.print_startup_summary()
    out = capsys.readouterr().out
    assert text in out
    for needle in [
        "A100",
        "40.0",
        "64.0",
        "12.4",
        "2.3.0",
        "A100",
        MODEL_9B,
    ]:
        assert needle in text
    lowered = text.lower()
    assert "gpu" in lowered
    assert "vram" in lowered
    assert "ram" in lowered
    assert "cuda" in lowered
    assert "torch" in lowered
    assert "profile" in lowered
    assert "model" in lowered
