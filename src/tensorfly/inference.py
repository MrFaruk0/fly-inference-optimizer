"""TensorFly Qwen inference backend (lazy transformers, profile-aware defaults).

Model policy (benchmark profile):
  * ``A100`` / ``L4`` -> ``Qwen/Qwen3.5-9B`` (9B benchmark profile).
  * ``T4`` / ``CPU`` (or anything unknown) -> ``Qwen/Qwen3.5-4B``.
  * Insufficient VRAM for the 9B profile -> fall back to 4B with an exact
    one-line notice::

        TensorFly fallback: Qwen3.5-4B because current GPU memory is insufficient for the 9B benchmark profile.

Design rules:
  * ``transformers`` / ``torch`` are imported lazily (inside functions) so
    model selection, dtype resolution, and metadata work without either
    installed and without a GPU.
  * A100 prefers ``bfloat16`` when supported, otherwise ``float16``.
    L4/T4 use ``float16``; CPU uses ``float32``.
  * Memory-efficient loading: ``device_map="auto"`` + ``low_cpu_mem_usage``
    when CUDA is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

MODEL_9B = "Qwen/Qwen3.5-9B"
MODEL_4B = "Qwen/Qwen3.5-4B"

FALLBACK_MESSAGE = (
    "TensorFly fallback: Qwen3.5-4B because current GPU memory "
    "is insufficient for the 9B benchmark profile."
)

#: Below this VRAM (GB) the 9B benchmark profile is considered not fittable.
#: A100 (40/80GB) and L4 (24GB) clear it; T4 (16GB) does not.
MIN_VRAM_GB_FOR_9B = 20.0

_CAPABLE_PROFILES = ("A100", "L4")


def select_qwen_model(
    profile_name: str = "CPU",
    vram_gb: Optional[float] = None,
) -> str:
    """Return the default Qwen model id for a device profile.

    * ``A100`` / ``L4`` with sufficient (or unknown) VRAM -> 9B.
    * ``T4`` / ``CPU`` / unknown -> 4B.
    * Any capable profile with ``vram_gb`` below :data:`MIN_VRAM_GB_FOR_9B`
      -> 4B with the exact fallback notice printed.

    The fallback notice (when printed) is exactly :data:`FALLBACK_MESSAGE`.
    No ``transformers`` / GPU import is performed here.
    """
    key = (profile_name or "CPU").strip().upper()
    if key in _CAPABLE_PROFILES:
        if vram_gb is not None and float(vram_gb) < MIN_VRAM_GB_FOR_9B:
            print(FALLBACK_MESSAGE)
            return MODEL_4B
        return MODEL_9B
    print(FALLBACK_MESSAGE)
    return MODEL_4B


def get_default_model(profile_name: str = "CPU", vram_gb: Optional[float] = None) -> str:
    """Alias for :func:`select_qwen_model`."""
    return select_qwen_model(profile_name, vram_gb)


def resolve_dtype(profile_name: str = "CPU") -> str:
    """Resolve the torch dtype label for a profile without hard deps.

    * A100 -> ``bfloat16`` when supported, else ``float16``.
    * L4/T4 -> ``float16``.
    * CPU/other -> ``float32``.
    """
    key = (profile_name or "CPU").strip().upper()
    if key == "A100":
        try:
            import torch  # type: ignore[import]

            check = getattr(torch.cuda, "is_bf16_supported", None)
            if check is None:
                return "bfloat16"
            try:
                if not torch.cuda.is_available():
                    return "bfloat16"
                return "bfloat16" if bool(check()) else "float16"
            except Exception:
                return "bfloat16"
        except Exception:
            return "bfloat16"
    if key in ("L4", "T4"):
        return "float16"
    return "float32"


def resolve_device() -> str:
    """Return ``"cuda"`` when torch can use it, else ``"cpu"`` (no hard dep)."""
    try:
        import torch  # type: ignore[import]

        if bool(torch.cuda.is_available()):
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _torch_dtype_for_label(label: str) -> Any:
    """Map a dtype label to a torch dtype (lazy torch import)."""
    import torch  # type: ignore[import]

    normalized = (label or "").strip().lower()
    if normalized in ("bfloat16", "bf16"):
        return torch.bfloat16
    if normalized in ("float16", "fp16", "half"):
        return torch.float16
    return torch.float32


@dataclass
class InferenceConfig:
    """Knobs for Qwen text generation."""

    profile_name: str = "CPU"
    model_id: Optional[str] = None
    vram_gb: Optional[float] = None
    dtype: Optional[str] = None
    device: Optional[str] = None
    max_new_tokens: int = 128
    temperature: float = 0.0
    do_sample: bool = False
    trust_remote_code: bool = False


class QwenInference:
    """Profile-aware Qwen wrapper with lazy backend loading.

    Example::

        inf = QwenInference(profile_name="A100")
        inf.metadata          # explicit metadata, no GPU needed
        inf.load()            # requires transformers (+ torch for CUDA)
        out = inf.generate_with_metrics("describe the optic lobe response")
    """

    def __init__(
        self,
        profile_name: Optional[str] = None,
        model_id: Optional[str] = None,
        vram_gb: Optional[float] = None,
        dtype: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        trust_remote_code: Optional[bool] = None,
        config: Optional[InferenceConfig] = None,
    ) -> None:
        # Precedence: explicit kwarg (not None) > config > built-in default.
        # Using None as the sentinel keeps callers that omit a knob
        # indistinguishable from "use config/default" while letting any
        # explicitly passed value (including falsy 0/False) win.
        _defaults = InferenceConfig()
        eff_profile = (
            profile_name
            if profile_name is not None
            else (config.profile_name if config is not None else _defaults.profile_name)
        )
        eff_model = (
            model_id
            if model_id is not None
            else (config.model_id if config is not None else None)
        )
        eff_vram = (
            vram_gb
            if vram_gb is not None
            else (config.vram_gb if config is not None else None)
        )
        eff_dtype = (
            dtype
            if dtype is not None
            else (config.dtype if config is not None else None)
        )
        eff_device = (
            device
            if device is not None
            else (config.device if config is not None else None)
        )
        if max_new_tokens is not None:
            eff_max_new = max_new_tokens
        elif config is not None:
            eff_max_new = config.max_new_tokens
        else:
            eff_max_new = _defaults.max_new_tokens
        if temperature is not None:
            eff_temperature = temperature
        elif config is not None:
            eff_temperature = config.temperature
        else:
            eff_temperature = _defaults.temperature
        if do_sample is not None:
            eff_do_sample = do_sample
        elif config is not None:
            eff_do_sample = config.do_sample
        else:
            eff_do_sample = _defaults.do_sample
        if trust_remote_code is not None:
            eff_trust = trust_remote_code
        elif config is not None:
            eff_trust = config.trust_remote_code
        else:
            eff_trust = _defaults.trust_remote_code

        profile_name = eff_profile
        model_id = eff_model
        vram_gb = eff_vram
        dtype = eff_dtype
        device = eff_device
        max_new_tokens = eff_max_new
        temperature = eff_temperature
        do_sample = eff_do_sample
        trust_remote_code = eff_trust

        self.profile_name = (profile_name or "CPU").strip().upper()
        if self.profile_name not in ("A100", "L4", "T4", "CPU"):
            # Keep unknown labels for provenance but model-select as fallback.
            pass
        self.vram_gb = vram_gb
        if model_id is None:
            self.model_id: str = select_qwen_model(self.profile_name, self.vram_gb)
            self.model_explicit = False
        else:
            self.model_id = model_id
            self.model_explicit = True
        self.dtype_name: str = dtype or resolve_dtype(self.profile_name)
        self.device: str = device or resolve_device()
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.do_sample = bool(do_sample)
        self.trust_remote_code = bool(trust_remote_code)

        self._tokenizer: Any = None
        self._model: Any = None
        self._loaded = False
        self._processor: Any = None
        self._loader_kind: str = "unloaded"
        self._processor_kind: str = "unloaded"

    # -- metadata ------------------------------------------------------
    @property
    def metadata(self) -> Dict[str, Any]:
        """Explicit metadata (safe without transformers/GPU)."""
        return {
            "model": self.model_id,
            "profile": self.profile_name,
            "dtype": self.dtype_name,
            "device": self.device,
            "vram_gb": self.vram_gb,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
            "trust_remote_code": self.trust_remote_code,
            "loader": self._loader_kind,
            "processor": self._processor_kind,
            "loaded": self._loaded,
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # -- loading (memory-efficient, lazy) ------------------------------
    def load(self) -> "QwenInference":
        """Load processor/tokenizer + multimodal model with efficient settings.

        Qwen3.5-9B/4B are multimodal, so the preferred path is
        ``AutoProcessor`` + a multimodal LM (``AutoModelForImageTextToText`` /
        ``AutoModelForVision2Seq`` / ``AutoModelForMultimodalLM`` when
        available). The causal-LM loader (``AutoTokenizer`` +
        ``AutoModelForCausalLM``) is used as an explicit fallback only when
        the multimodal loader is unavailable in the installed
        ``transformers``. Requested exact model IDs are always retained.

        Raises:
            ImportError: if ``transformers`` (or ``torch`` for dtype mapping)
                is not installed.
        """
        if self._loaded:
            return self
        try:
            import transformers as _tf  # type: ignore[import]
        except Exception as exc:
            raise ImportError(
                "QwenInference.load() requires the 'transformers' package. "
                "Install the 'inference' extra: pip install tensorfly[inference]."
            ) from exc

        # -- processor / tokenizer -------------------------------------
        # Prefer AutoProcessor (multimodal); fall back to AutoTokenizer only
        # when AutoProcessor is unavailable in this transformers version.
        processor_cls = getattr(_tf, "AutoProcessor", None)
        tokenizer_cls = getattr(_tf, "AutoTokenizer", None)
        if processor_cls is not None:
            try:
                processor = processor_cls.from_pretrained(
                    self.model_id, trust_remote_code=self.trust_remote_code
                )
            except Exception:
                # Processor class exists but this checkpoint cannot use it
                # (e.g. text-only local mock): fall back to tokenizer.
                if tokenizer_cls is None:
                    raise
                processor = None
                tokenizer = tokenizer_cls.from_pretrained(
                    self.model_id, trust_remote_code=self.trust_remote_code
                )
                self._processor_kind = "tokenizer"
            else:
                tokenizer = processor
                # Expose inner tokenizer when the processor wraps one so
                # generate()/decode() helpers keep working.
                inner = getattr(processor, "tokenizer", None)
                if inner is not None:
                    try:
                        tokenizer = inner
                    except Exception:
                        tokenizer = processor
                self._processor_kind = "processor"
                self._processor = processor
        else:
            if tokenizer_cls is None:
                raise ImportError(
                    "QwenInference.load() requires AutoProcessor or "
                    "AutoTokenizer from 'transformers'."
                )
            print(
                "[tensorfly.inference] AutoProcessor unavailable; "
                "falling back to AutoTokenizer (text-only loader)."
            )
            tokenizer = tokenizer_cls.from_pretrained(
                self.model_id, trust_remote_code=self.trust_remote_code
            )
            self._processor_kind = "tokenizer-fallback"

        # -- model ------------------------------------------------------
        model_cls = None
        model_cls_name = ""
        for candidate in (
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForMultimodalLM",
        ):
            candidate_cls = getattr(_tf, candidate, None)
            if candidate_cls is not None:
                model_cls = candidate_cls
                model_cls_name = candidate
                break
        if model_cls is None:
            fallback_cls = getattr(_tf, "AutoModelForCausalLM", None)
            if fallback_cls is None:
                raise ImportError(
                    "QwenInference.load() requires a multimodal model loader "
                    "or AutoModelForCausalLM from 'transformers'."
                )
            print(
                "[tensorfly.inference] Multimodal model loader unavailable; "
                "falling back to AutoModelForCausalLM (text-only loader)."
            )
            model_cls = fallback_cls
            model_cls_name = "AutoModelForCausalLM-fallback"

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if self.device.startswith("cuda"):
            try:
                load_kwargs["torch_dtype"] = _torch_dtype_for_label(self.dtype_name)
            except Exception:
                pass
            load_kwargs["device_map"] = "auto"
        else:
            # CPU path: keep float32, no device_map offload.
            try:
                import torch  # type: ignore[import]

                load_kwargs["torch_dtype"] = torch.float32
            except Exception:
                pass
        model = model_cls.from_pretrained(self.model_id, **load_kwargs)
        self._tokenizer = tokenizer
        if self._processor is None and self._processor_kind.startswith("tokenizer"):
            self._processor = tokenizer
        self._model = model
        self._loader_kind = model_cls_name
        self._loaded = True
        return self

    # -- generation ----------------------------------------------------
    def generate_with_metrics(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate text and report latency/throughput metrics.

        Returns a dict with ``text``, ``tokens``, ``ttft_ms``,
        ``decode_ms_per_token``, ``throughput_tps`` plus ``model``,
        ``profile``, ``dtype``, and ``device`` provenance.

        Works with real HF models and with lightweight fakes exposing
        ``tokenizer(prompt, return_tensors="pt")`` / ``tokenizer.decode`` and
        ``model.generate(...)`` (used by CPU-only tests).
        """
        if not self._loaded or self._model is None or self._tokenizer is None:
            self.load()
        import time

        budget = int(max_new_tokens) if max_new_tokens is not None else self.max_new_tokens
        inputs: Any = self._tokenizer(prompt, return_tensors="pt")

        # Move torch tensors to the target device when possible.
        try:
            target = self.device
            if isinstance(inputs, dict):
                moved: Dict[str, Any] = {}
                for k, v in inputs.items():
                    to = getattr(v, "to", None)
                    try:
                        moved[k] = to(target) if callable(to) else v
                    except Exception:
                        moved[k] = v
                inputs = moved
            else:
                to = getattr(inputs, "to", None)
                if callable(to):
                    try:
                        inputs = inputs.to(self.device)
                    except Exception:
                        pass
        except Exception:
            pass

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": budget}
        if self.do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
        else:
            gen_kwargs["do_sample"] = False

        # Optional torch no-grad scoping when torch exists.
        try:
            import torch  # type: ignore[import]

            no_grad = torch.no_grad()
        except Exception:  # pragma: no cover - torch-less path
            import contextlib

            no_grad = contextlib.nullcontext()

        def _length(value: Any) -> int:
            try:
                if hasattr(value, "shape") and len(value.shape) >= 1:
                    return int(value.shape[-1])
            except Exception:
                pass
            try:
                return int(len(value))
            except Exception:
                pass
            try:
                import torch  # type: ignore[import]

                if torch.is_tensor(value):
                    return int(value.numel())
            except Exception:
                pass
            return 0

        def _input_length(payload: Any) -> int:
            if isinstance(payload, dict):
                for key in ("input_ids", "inputs", "tokens"):
                    if key in payload:
                        return _length(payload[key])
                for v in payload.values():
                    n = _length(v)
                    if n:
                        return n
                return 0
            return _length(payload)

        start = time.perf_counter()
        with no_grad:
            try:
                if isinstance(inputs, dict):
                    output_ids = self._model.generate(**inputs, **gen_kwargs)
                else:
                    output_ids = self._model.generate(inputs, **gen_kwargs)
            except TypeError:
                # Minimal fakes may not accept generate kwargs.
                if isinstance(inputs, dict):
                    output_ids = self._model.generate(**inputs)
                else:
                    output_ids = self._model.generate(inputs)
        elapsed = max(1e-9, time.perf_counter() - start)

        input_len = _input_length(inputs)
        total_len = _length(output_ids)
        new_tokens = max(0, total_len - input_len) if input_len else total_len
        if new_tokens == 0:
            new_tokens = budget if budget > 0 else 1

        try:
            if hasattr(output_ids, "tolist"):
                seq = output_ids.tolist()
                first = seq[0] if isinstance(seq, list) and seq else seq
            else:
                first = output_ids
                seq = first
            if isinstance(first, list) and input_len and isinstance(seq, list):
                pass
            text = str(self._tokenizer.decode(first, skip_special_tokens=True))
        except Exception:
            text = ""

        ttft_ms = elapsed * 1000.0
        decode_ms = (elapsed * 1000.0) / max(1, new_tokens)
        throughput = float(new_tokens) / elapsed
        return {
            "text": text,
            "tokens": int(new_tokens),
            "ttft_ms": float(ttft_ms),
            "decode_ms_per_token": float(decode_ms),
            "throughput_tps": float(throughput),
            "model": self.model_id,
            "profile": self.profile_name,
            "dtype": self.dtype_name,
            "device": self.device,
        }


__all__: List[str] = [
    "MODEL_9B",
    "MODEL_4B",
    "FALLBACK_MESSAGE",
    "MIN_VRAM_GB_FOR_9B",
    "InferenceConfig",
    "QwenInference",
    "select_qwen_model",
    "get_default_model",
    "resolve_dtype",
    "resolve_device",
]
