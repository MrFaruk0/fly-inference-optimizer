"""Transparent reward and equal-budget baseline utilities.

The reward is a numerical engineering objective.  ``dopamine_current`` is a
control signal sent to an identified population by the experiment runner; it
does not represent pleasure, addiction, or reconstructed receptor dynamics.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, asdict, fields, is_dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .controller import config_value


@dataclass(frozen=True)
class RewardWeights:
    ttft: float = 0.35
    tpot: float = 0.30
    throughput: float = 0.25
    memory: float = 0.10

    def __post_init__(self) -> None:
        values = (self.ttft, self.tpot, self.throughput, self.memory)
        if any(float(v) < 0 for v in values) or sum(values) <= 0:
            raise ValueError("reward weights must be non-negative with positive sum")


@dataclass(frozen=True)
class RewardResult:
    reward: float
    normalized_improvements: dict[str, float]
    dopamine_current: float
    weights: dict[str, float]


def _value(metrics: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        try:
            value = float(metrics.get(key, 0.0))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value >= 0:
            if key.endswith("_bytes"):
                value /= 1024.0 * 1024.0
            return value
    return 0.0


def _relative(current: float, baseline: float, *, lower_is_better: bool) -> float:
    if baseline <= 0:
        return 0.0
    change = (baseline - current) / baseline if lower_is_better else (current - baseline) / baseline
    return float(np.clip(change, -1.0, 1.0))


def compute_reward(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    *,
    weights: RewardWeights | None = None,
) -> RewardResult:
    """Compute a bounded, explicitly weighted normalized performance reward."""

    chosen = weights or RewardWeights()
    base = baseline or current
    improvements = {
        "ttft": _relative(
            _value(current, "ttft_ms", "time_to_first_token_ms"),
            _value(base, "ttft_ms", "time_to_first_token_ms"),
            lower_is_better=True,
        ),
        "tpot": _relative(
            _value(current, "decode_ms_per_token", "tpot_ms"),
            _value(base, "decode_ms_per_token", "tpot_ms"),
            lower_is_better=True,
        ),
        "throughput": _relative(
            _value(current, "throughput_tps", "tokens_per_second"),
            _value(base, "throughput_tps", "tokens_per_second"),
            lower_is_better=False,
        ),
        "memory": _relative(
            0.5
            * (
                _value(current, "peak_allocated_vram_mb", "peak_allocated_mb", "peak_allocated_bytes")
                + _value(current, "peak_reserved_vram_mb", "peak_reserved_mb", "peak_reserved_bytes")
            ),
            0.5
            * (
                _value(base, "peak_allocated_vram_mb", "peak_allocated_mb", "peak_allocated_bytes")
                + _value(base, "peak_reserved_vram_mb", "peak_reserved_mb", "peak_reserved_bytes")
            ),
            lower_is_better=True,
        ),
    }
    total_weight = chosen.ttft + chosen.tpot + chosen.throughput + chosen.memory
    reward = (
        chosen.ttft * improvements["ttft"]
        + chosen.tpot * improvements["tpot"]
        + chosen.throughput * improvements["throughput"]
        + chosen.memory * improvements["memory"]
    ) / total_weight
    reward = float(np.clip(reward, -1.0, 1.0))
    return RewardResult(
        reward=reward,
        normalized_improvements=improvements,
        dopamine_current=reward,
        weights=asdict(chosen),
    )


@dataclass(frozen=True)
class ConfigurationSpace:
    """Small, legitimate runtime search space shared by all strategies."""

    batch_sizes: tuple[int, ...] = (1, 2, 4, 8)
    use_cache: tuple[bool, ...] = (True, False)
    # Only expose knobs demonstrated by the benchmark path by default.
    # Attention-kernel/compile candidates are opt-in after hardware-specific
    # validation; they must not trigger costly speculative recompilation.
    attention_implementations: tuple[str | None, ...] = (None,)
    torch_compile: tuple[bool, ...] = (False,)

    def __post_init__(self) -> None:
        if not self.batch_sizes or any(int(x) <= 0 for x in self.batch_sizes):
            raise ValueError("configuration space needs positive batch sizes")
        if not self.use_cache or not self.attention_implementations or not self.torch_compile:
            raise ValueError("configuration space dimensions must not be empty")

    def candidate(self, config: Any, *, batch_size: int | None = None, use_cache: bool | None = None,
                  attention_implementation: str | None = None, torch_compile: bool | None = None) -> Any:
        # Current InferenceConfig names are ``attn_implementation`` and
        # ``compile_model``.  The longer names remain accepted for lightweight
        # fakes and older callers, but the actual Qwen config receives its real
        # runtime fields so candidates are not cosmetic metadata.
        attention_key = "attn_implementation" if (
            isinstance(config, Mapping) and "attn_implementation" in config
        ) or hasattr(config, "attn_implementation") else "attention_implementation"
        compile_key = "compile_model" if (
            isinstance(config, Mapping) and "compile_model" in config
        ) or hasattr(config, "compile_model") else "torch_compile"
        updates = {
            "batch_size": int(batch_size if batch_size is not None else config_value(config, "batch_size", self.batch_sizes[0])),
            "use_cache": bool(use_cache if use_cache is not None else config_value(config, "use_cache", self.use_cache[0])),
            attention_key: attention_implementation if attention_implementation is not None else config_value(config, attention_key, self.attention_implementations[0]),
            compile_key: bool(torch_compile if torch_compile is not None else config_value(config, compile_key, self.torch_compile[0])),
        }
        if isinstance(config, Mapping):
            result = dict(config)
            result.update(updates)
            return result
        if is_dataclass(config):
            known = {field.name for field in fields(config)}
            return replace(config, **{key: value for key, value in updates.items() if key in known})
        result = copy.copy(config)
        for key, value in updates.items():
            setattr(result, key, value)
        return result

    def deterministic_random(self, config: Any, rng: np.random.Generator) -> Any:
        """Generate a reproducible random-search candidate (explicit RNG)."""
        attention = rng.choice(self.attention_implementations)
        if hasattr(attention, "item"):
            attention = attention.item()
        return self.candidate(
            config,
            batch_size=int(rng.choice(self.batch_sizes)),
            use_cache=bool(rng.choice(self.use_cache)),
            attention_implementation=attention,
            torch_compile=bool(rng.choice(self.torch_compile)),
        )

    def for_prompt_count(self, count: int) -> "ConfigurationSpace":
        """Restrict batching choices to complete batches in a fixed corpus."""
        compatible = tuple(size for size in self.batch_sizes if int(count) % int(size) == 0)
        if not compatible:
            raise ValueError("fixed prompt corpus cannot form a complete configured batch")
        return ConfigurationSpace(
            batch_sizes=compatible,
            use_cache=self.use_cache,
            attention_implementations=self.attention_implementations,
            torch_compile=self.torch_compile,
        )


def workload_signature(prompt_corpus: Sequence[str], max_new_tokens: int, warmup: int) -> dict[str, Any]:
    """Signature used to prove every strategy ran the same workload."""
    return {
        "prompt_corpus": tuple(str(p) for p in prompt_corpus),
        "max_new_tokens": int(max_new_tokens),
        "warmup": int(warmup),
    }


__all__ = [
    "RewardWeights",
    "RewardResult",
    "compute_reward",
    "ConfigurationSpace",
    "workload_signature",
]
