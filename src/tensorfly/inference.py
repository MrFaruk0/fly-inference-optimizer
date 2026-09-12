"""Measured Qwen inference for the TensorFly experiment.

This module deliberately does not run a model at import time. ``load()`` and
``benchmark()`` are intended for the Colab runtime; tests inject small fakes.
The benchmark is text-only and fixes both prompt corpus and token budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from threading import Thread
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional

MODEL_9B = "Qwen/Qwen3.5-9B"
MODEL_4B = "Qwen/Qwen3.5-4B"


def select_qwen_model(
    requested_model: str = MODEL_9B,
    *,
    vram_gb: float | None = None,
) -> tuple[str, str | None]:
    """Resolve the model without hiding a VRAM-driven fallback.

    A 9B BF16 multimodal checkpoint is not a credible direct-load target on a
    16 GB T4.  The caller receives an explicit notice which is persisted in
    the experiment metadata and printed by the Colab notebook.
    """
    if requested_model != MODEL_9B or vram_gb is None or vram_gb >= 22.0:
        return requested_model, None
    return (
        MODEL_4B,
        f"Requested {MODEL_9B}, but detected {vram_gb:.1f} GB VRAM; "
        f"using {MODEL_4B}. This is an explicit T4-class fallback, not a silent substitution.",
    )


@dataclass(frozen=True)
class InferenceConfig:
    """Real generation knobs; max_new_tokens defines the fixed workload."""

    model_id: str = MODEL_9B
    revision: Optional[str] = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    batch_size: int = 1
    use_cache: bool = True
    attn_implementation: Optional[str] = None
    compile_model: bool = False
    padding_strategy: str = "longest"
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.0
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.padding_strategy not in {"longest", "max_length"}:
            raise ValueError("padding_strategy must be 'longest' or 'max_length'")
        if self.do_sample or self.temperature != 0.0:
            raise ValueError("TensorFly measurements require greedy deterministic generation")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_PROMPTS: tuple[str, ...] = (
    "Explain why separating prefill latency from decode latency matters in language model serving.",
    "Summarize a reproducible method for measuring time to first token.",
    "Give a concise explanation of cache-aware autoregressive decoding.",
    "Describe one limitation of mapping benchmark metrics onto a neural simulation.",
)


def prompt_corpus_fingerprint(prompts: Iterable[str]) -> str:
    """Fingerprint the exact fixed workload without storing private prompts."""
    return sha256("\0".join(str(prompt) for prompt in prompts).encode("utf-8")).hexdigest()


class _TokenTimingStreamer:
    """Raw-token streamer: TTFT stops at first generated token, not end."""

    def __init__(self, prompt_tokens: int, clock=perf_counter) -> None:
        self.prompt_tokens = int(prompt_tokens)
        self.clock = clock
        self._seen_prompt = False
        self.first_token_at: Optional[float] = None
        self.end_at: Optional[float] = None
        self.generated_tokens = 0

    @staticmethod
    def _count(value: Any) -> int:
        try:
            shape = value.shape
            if len(shape) == 0:
                return 1
            return int(shape[-1] if len(shape) > 1 else shape[0])
        except Exception:
            try:
                return len(value)
            except Exception:
                return 1

    def put(self, value: Any) -> None:
        count = self._count(value)
        # HF generation calls a streamer first with the complete prompt. Do
        # not use a length comparison: a one-token prompt is otherwise
        # indistinguishable from the first generated token.
        if not self._seen_prompt:
            self._seen_prompt = True
            return
        now = self.clock()
        if self.first_token_at is None:
            self.first_token_at = now
        self.generated_tokens += count

    def end(self) -> None:
        self.end_at = self.clock()


def _cuda_elapsed_ms(torch: Any, start_event: Any, end_event: Any) -> float:
    torch.cuda.synchronize()
    return float(start_event.elapsed_time(end_event))


class QwenInference:
    """Lazy Colab-only loader and measured runner for Qwen3.5-9B."""

    def __init__(self, config: Optional[InferenceConfig] = None, **overrides: Any) -> None:
        values = (config or InferenceConfig()).to_dict()
        values.update({key: value for key, value in overrides.items() if value is not None})
        self.config = InferenceConfig(**values)
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None

    @property
    def metadata(self) -> Dict[str, Any]:
        metadata = {**self.config.to_dict(), "loaded": self._model is not None}
        if self._torch is not None:
            metadata["torch_version"] = str(getattr(self._torch, "__version__", "unknown"))
            metadata["cuda_version"] = str(getattr(getattr(self._torch, "version", None), "cuda", None))
            if self.config.device.startswith("cuda") and self._torch.cuda.is_available():
                metadata["gpu"] = str(self._torch.cuda.get_device_name(0))
                metadata["vram_bytes"] = int(self._torch.cuda.get_device_properties(0).total_memory)
        try:
            import transformers
            metadata["transformers_version"] = str(transformers.__version__)
        except ImportError:
            pass
        return metadata

    def load(self) -> "QwenInference":
        """Load only when the Colab user invokes the actual benchmark."""
        if self._model is not None:
            return self
        try:
            import torch
            import transformers
        except ImportError as exc:  # pragma: no cover - Colab-only dependency
            raise ImportError("Install tensorfly[inference] in the Colab runtime.") from exc
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("TensorFly Qwen benchmark requires a CUDA Colab runtime.")
        processor_cls = getattr(transformers, "AutoProcessor", None)
        if processor_cls is None:
            raise RuntimeError("Installed transformers lacks AutoProcessor; upgrade Transformers.")
        source_kwargs: Dict[str, Any] = {"trust_remote_code": self.config.trust_remote_code}
        if self.config.revision:
            source_kwargs["revision"] = self.config.revision
        self._processor = processor_cls.from_pretrained(self.config.model_id, **source_kwargs)
        # Decoder-only generation must use left padding.  AutoProcessor wraps
        # a tokenizer for this checkpoint, so update both possible surfaces.
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
        if hasattr(self._processor, "padding_side"):
            self._processor.padding_side = "left"
        model_cls = getattr(transformers, "AutoModelForMultimodalLM", None)
        # Text-only Qwen builds sometimes expose only this official fallback.
        if model_cls is None:
            model_cls = getattr(transformers, "AutoModelForCausalLM", None)
        if model_cls is None:
            raise RuntimeError("Installed transformers has no supported Qwen model loader.")
        load_kwargs: Dict[str, Any] = {**source_kwargs, "dtype": getattr(torch, self.config.dtype), "low_cpu_mem_usage": True}
        if self.config.attn_implementation:
            load_kwargs["attn_implementation"] = self.config.attn_implementation
        if self.config.device.startswith("cuda"):
            load_kwargs["device_map"] = "auto"
        self._model = model_cls.from_pretrained(self.config.model_id, **load_kwargs)
        if self.config.compile_model:
            self._model = torch.compile(self._model)
        self._model.eval()
        self._torch = torch
        return self

    def _inputs(self, prompts: List[str]) -> Dict[str, Any]:
        assert self._processor is not None
        # Qwen's current official example uses AutoProcessor plus a chat
        # template.  Keeping the text-only benchmark in that documented path
        # avoids treating this multimodal checkpoint as an arbitrary CausalLM.
        if hasattr(self._processor, "apply_chat_template"):
            messages = [
                [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                for prompt in prompts
            ]
            try:
                payload = self._processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                )
            except (TypeError, ValueError):
                # Some released processor versions expose the documented
                # single-conversation template only. Render every prompt with
                # that same template, then batch the resulting text.
                rendered = [
                    self._processor.apply_chat_template(
                        conversation, add_generation_prompt=True, tokenize=False
                    )
                    for conversation in messages
                ]
                payload = self._processor(text=rendered, return_tensors="pt", padding=True)
        else:  # Compatibility fallback for a future processor surface.
            payload = self._processor(text=prompts, return_tensors="pt", padding=True)
        return {key: value.to(self.config.device) if hasattr(value, "to") else value for key, value in dict(payload).items()}

    @staticmethod
    def _token_count(input_ids: Any) -> int:
        try:
            return int(input_ids.shape[-1])
        except Exception:
            return len(input_ids)

    def warmup(self, prompts: Iterable[str], steps: int = 1) -> None:
        batch = list(prompts)[: self.config.batch_size]
        if len(batch) != self.config.batch_size:
            raise ValueError("warmup needs a complete batch")
        for _ in range(steps):
            self.benchmark_batch(batch, record_memory=False)

    def benchmark_batch(self, prompts: List[str], record_memory: bool = True) -> Dict[str, Any]:
        """Measure independent prefill, true TTFT, decode/TPOT, and memory."""
        self.load()
        assert self._torch is not None and self._model is not None
        torch = self._torch
        if len(prompts) != self.config.batch_size:
            raise ValueError("Each benchmark batch must equal config.batch_size.")
        inputs = self._inputs(prompts)
        prompt_tokens = self._token_count(inputs["input_ids"])
        cuda = self.config.device.startswith("cuda")
        if cuda and record_memory:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            if cuda:
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record(); self._model(**inputs, use_cache=self.config.use_cache); end.record()
                prefill_ms = _cuda_elapsed_ms(torch, begin, end)
            else:  # developer diagnostic, never an accepted Colab result
                began = perf_counter(); self._model(**inputs, use_cache=self.config.use_cache); prefill_ms = (perf_counter() - began) * 1000.0
        streamer = _TokenTimingStreamer(prompt_tokens)
        generation_kwargs: Dict[str, Any] = {**inputs, "max_new_tokens": self.config.max_new_tokens, "do_sample": False, "use_cache": self.config.use_cache, "streamer": streamer}
        request_start = perf_counter()
        generation_error: list[BaseException] = []
        def generate() -> None:
            try:
                self._model.generate(**generation_kwargs)
            except BaseException as exc:  # surface worker failures to Colab
                generation_error.append(exc)
        worker = Thread(target=generate, daemon=True)
        worker.start(); worker.join()
        if cuda:
            torch.cuda.synchronize()
        if generation_error:
            raise RuntimeError("Qwen generation failed while measuring TTFT") from generation_error[0]
        request_end = streamer.end_at or perf_counter()
        if streamer.first_token_at is None or streamer.generated_tokens < 1:
            raise RuntimeError("Generation yielded no tokens; TTFT cannot be measured.")
        ttft_ms = (streamer.first_token_at - request_start) * 1000.0
        end_to_end_ms = (request_end - request_start) * 1000.0
        decode_ms = max(0.0, (request_end - streamer.first_token_at) * 1000.0)
        tpot_ms = decode_ms / max(1, streamer.generated_tokens - 1)
        memory = {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
        if cuda and record_memory:
            torch.cuda.synchronize()
            memory = {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
        result = {
            "prefill_latency_ms": float(prefill_ms), "ttft_ms": float(ttft_ms), "decode_latency_ms": float(decode_ms), "tpot_ms": float(tpot_ms),
            "throughput_tps": float(streamer.generated_tokens / max(1e-9, end_to_end_ms / 1000.0)), "end_to_end_latency_ms": float(end_to_end_ms),
            "generated_tokens": int(streamer.generated_tokens), "prompt_tokens": int(prompt_tokens), "batch_size": self.config.batch_size,
            "prompt_corpus_sha256": prompt_corpus_fingerprint(prompts), "config": self.config.to_dict(),
            "runtime": self.metadata, **memory,
        }
        if memory["peak_allocated_bytes"] is not None:
            result["peak_allocated_vram_mb"] = memory["peak_allocated_bytes"] / (1024.0 * 1024.0)
            result["peak_reserved_vram_mb"] = memory["peak_reserved_bytes"] / (1024.0 * 1024.0)
        return result

    def benchmark(self, prompts: Iterable[str], warmup: int = 1) -> List[Dict[str, Any]]:
        corpus = list(prompts)
        if not corpus:
            raise ValueError("prompt corpus is empty")
        if len(corpus) % self.config.batch_size:
            raise ValueError("prompt corpus length must be divisible by batch_size")
        self.warmup(corpus, steps=warmup)
        return [self.benchmark_batch(corpus[i : i + self.config.batch_size]) for i in range(0, len(corpus), self.config.batch_size)]


__all__ = ["MODEL_9B", "MODEL_4B", "DEFAULT_PROMPTS", "InferenceConfig", "QwenInference", "prompt_corpus_fingerprint", "select_qwen_model"]
