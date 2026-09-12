"""TensorFly replay export: ``fly-cns-replay/1`` JSON for the standalone viewer.

The viewer (``viewer/app.js`` + ``viewer/README.md``) is snapshot-driven:
every visual/metric value is interpolated from ``frames``. This recorder
produces exactly that schema without any GPU dependency:

Top-level::

    {
      "schema": "fly-cns-replay/1",
      "meta": {
        "model": "...", "prompt": "...",
        "populations": {"sensory": N, "dopamine": N, "controller": N},
        "config": {...}, "best": {"config": "...", "reward": 0.0}
      },
      "frames": [
        {
          "t": 0.0, "trial": 0, "tokens": 0,
          "throughput_tps": 0.0, "ttft_ms": 0.0,
          "decode_ms_per_token": 0.0, "reward": 0.0,
          "config_id": "cfg-A", "config": {"id": "cfg-A"},
          "event": {"type": "config_change", "label": "..."} | None,
          "sensory": [...], "dopamine": [...], "controller": [...]
        }
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REPLAY_SCHEMA = "fly-cns-replay/1"

DEFAULT_POPULATIONS: Dict[str, int] = {
    "sensory": 3800,
    "dopamine": 550,
    "controller": 1300,
}

_ALLOWED_EVENT_TYPES = ("config_change", "best", "trial_end")


def _as_float_list(values: Any) -> List[float]:
    """Coerce activity arrays to finite floats clamped to [0, 1].

    The viewer expects dense activity in ``[0, 1]`` (clamped); non-finite
    (NaN/Inf) or non-numeric entries become ``0.0`` instead of leaking
    through as invalid JSON or out-of-range visuals.
    """
    import math

    if values is None:
        return []
    try:
        import numpy as np  # type: ignore[import]

        if isinstance(values, np.ndarray):
            values = values.tolist()
    except Exception:
        pass
    try:
        items = list(values)
    except Exception:
        return []
    out: List[float] = []
    for v in items:
        try:
            f = float(v)
        except Exception:
            out.append(0.0)
            continue
        if not math.isfinite(f):
            out.append(0.0)
            continue
        if f < 0.0:
            f = 0.0
        elif f > 1.0:
            f = 1.0
        out.append(f)
    return out


class ReplayRecorder:
    """Collect snapshots/events/inference/config rows and export replay JSON.

    Example::

        rec = ReplayRecorder(model="fly-qwen-9b", prompt="describe ...",
                             populations={"sensory": 8, "dopamine": 4, "controller": 4},
                             config={"id": "cfg-A"})
        rec.record_snapshot(t=0.0, sensory=[...], dopamine=[...], controller=[...],
                            tokens=3, throughput_tps=36.0, reward=0.5)
        rec.record_event("config_change", "cfg-A -> cfg-B")
        rec.save("run.json")
    """

    def __init__(
        self,
        model: str = "fly-qwen",
        prompt: str = "",
        populations: Optional[Dict[str, int]] = None,
        config: Optional[Dict[str, Any]] = None,
        is_synthetic: bool = False,
        data_source: Optional[str] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = str(model)
        self.prompt = str(prompt)
        pops = dict(DEFAULT_POPULATIONS)
        if populations:
            for k, v in populations.items():
                pops[str(k)] = int(v)
        self.populations: Dict[str, int] = pops
        self.config: Dict[str, Any] = dict(config or {})
        self.frames: List[Dict[str, Any]] = []
        self._best: Dict[str, Any] = {"config": None, "reward": float("-inf")}
        self.is_synthetic: bool = bool(is_synthetic)
        self.data_source: str = str(
            data_source
            if data_source is not None
            else "unprovenanced developer replay (rejected by the production viewer)"
        )
        self.provenance: Dict[str, Any] = dict(provenance or {})
        if "data_source" not in self.provenance:
            self.provenance["data_source"] = self.data_source
        self.provenance.setdefault("is_synthetic", self.is_synthetic)

    def record_provenance(
        self,
        is_synthetic: Optional[bool] = None,
        data_source: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Attach/override replay provenance (synthetic vs measured)."""
        if is_synthetic is not None:
            self.is_synthetic = bool(is_synthetic)
            self.provenance["is_synthetic"] = self.is_synthetic
        if data_source is not None:
            self.data_source = str(data_source)
            self.provenance["data_source"] = self.data_source
        if extra:
            self.provenance.update(dict(extra))
        return dict(self.provenance)

    # -- config ----------------------------------------------------------
    def record_config(self, config: Dict[str, Any] | str) -> Dict[str, Any]:
        """Set/replace the run-level ``meta.config`` block."""
        if isinstance(config, str):
            config = {"id": config}
        self.config = dict(config)
        return self.config

    # -- snapshots (incl. inference + config fields) ----------------------
    def record_snapshot(
        self,
        t: float,
        sensory: Any = None,
        dopamine: Any = None,
        controller: Any = None,
        trial: int = 0,
        tokens: int = 0,
        throughput_tps: float = 0.0,
        ttft_ms: float = 0.0,
        decode_ms_per_token: float = 0.0,
        reward: float = 0.0,
        config_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        event: Optional[Dict[str, Any]] = None,
        activity: Any = None,
        dopamine_mean: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Append one snapshot frame.

        At least one activity form is required: dense
        ``sensory``/``dopamine``/``controller`` arrays, a global ``activity``
        array, or a scalar ``dopamine_mean`` broadcast. Missing dense arrays
        are stored as the provided alternatives so the viewer can split /
        broadcast them (see viewer README).
        """
        cfg = dict(config) if isinstance(config, dict) else {}
        if config_id is not None:
            cfg.setdefault("id", config_id)
        cfg_id = str(cfg.get("id", config_id or self.config.get("id", "cfg-0")))

        import math as _math

        try:
            t_f = float(t)
        except Exception as exc:
            raise ValueError(f"t must be numeric: {t!r}") from exc
        if not _math.isfinite(t_f):
            raise ValueError(f"t must be finite: {t!r}")

        frame: Dict[str, Any] = {
            "t": t_f,
            "trial": int(trial),
            "tokens": int(tokens),
            "throughput_tps": float(throughput_tps),
            "ttft_ms": float(ttft_ms),
            "decode_ms_per_token": float(decode_ms_per_token),
            "reward": float(reward),
            "config_id": cfg_id,
            "config": cfg or {"id": cfg_id},
            "event": dict(event) if isinstance(event, dict) else None,
        }
        if sensory is not None:
            frame["sensory"] = _as_float_list(sensory)
        if dopamine is not None:
            frame["dopamine"] = _as_float_list(dopamine)
        if controller is not None:
            frame["controller"] = _as_float_list(controller)
        if activity is not None and (
            sensory is None and dopamine is None and controller is None
        ):
            frame["activity"] = _as_float_list(activity)
        if dopamine_mean is not None and dopamine is None:
            try:
                dm = float(dopamine_mean)
            except Exception as exc:
                raise ValueError(
                    f"dopamine_mean must be numeric: {dopamine_mean!r}"
                ) from exc
            if not _math.isfinite(dm):
                dm = 0.0
            dm = min(1.0, max(0.0, dm))
            frame["dopamine_mean"] = dm

        if (
            "sensory" not in frame
            and "dopamine" not in frame
            and "controller" not in frame
            and "activity" not in frame
            and "dopamine_mean" not in frame
        ):
            raise ValueError(
                "record_snapshot needs sensory/dopamine/controller, "
                "activity, or dopamine_mean activity data."
            )
        if frame["event"] is not None:
            self._validate_event(frame["event"])
        self.frames.append(frame)

        try:
            if float(reward) > float(self._best["reward"]):
                self._best = {"config": cfg_id, "reward": float(reward)}
        except Exception:
            pass
        return frame

    def record_inference(
        self,
        t: float,
        tokens: int = 0,
        throughput_tps: float = 0.0,
        ttft_ms: float = 0.0,
        decode_ms_per_token: float = 0.0,
        reward: float = 0.0,
        sensory: Any = None,
        dopamine: Any = None,
        controller: Any = None,
        trial: int = 0,
        config_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        event: Optional[Dict[str, Any]] = None,
        activity: Any = None,
        dopamine_mean: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convenience wrapper mapping inference metrics to a snapshot."""
        return self.record_snapshot(
            t=t,
            sensory=sensory,
            dopamine=dopamine,
            controller=controller,
            trial=trial,
            tokens=tokens,
            throughput_tps=throughput_tps,
            ttft_ms=ttft_ms,
            decode_ms_per_token=decode_ms_per_token,
            reward=reward,
            config_id=config_id,
            config=config,
            event=event,
            activity=activity,
            dopamine_mean=dopamine_mean,
        )

    # -- events ------------------------------------------------------------
    @staticmethod
    def _validate_event(event: Dict[str, Any]) -> None:
        if not isinstance(event, dict) or "type" not in event:
            raise ValueError("event must be a dict with a 'type' key")
        etype = str(event["type"])
        if etype not in _ALLOWED_EVENT_TYPES:
            raise ValueError(f"unknown event type: {etype!r}")

    def record_event(
        self,
        event_type: str,
        label: str,
        frame_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Attach ``{"type", "label"}`` to an existing snapshot frame.

        Defaults to the most recent frame (viewer renders events from
        ``frames[].event`` only, so events are never standalone).
        """
        if not self.frames:
            raise RuntimeError("record_event needs at least one snapshot frame first.")
        event = {"type": str(event_type), "label": str(label)}
        self._validate_event(event)
        if frame_index is None:
            idx = len(self.frames) - 1
        else:
            try:
                idx = int(frame_index)
            except Exception as exc:
                raise ValueError(f"frame_index must be an integer: {frame_index!r}") from exc
            if idx < 0 or idx >= len(self.frames):
                raise IndexError(
                    f"frame_index {idx} out of range for {len(self.frames)} frames"
                )
        self.frames[idx]["event"] = event
        return event

    # -- export --------------------------------------------------------------
    def best(self) -> Dict[str, Any]:
        """Best ``{config, reward}`` observed (or run config fallback)."""
        if self._best["config"] is not None and self._best["reward"] != float("-inf"):
            return {"config": self._best["config"], "reward": self._best["reward"]}
        return {
            "config": self.config.get("id", "cfg-0"),
            "reward": 0.0,
        }

    def to_dict(self) -> Dict[str, Any]:
        frames = sorted(self.frames, key=lambda f: float(f.get("t", 0.0)))
        return {
            "schema": REPLAY_SCHEMA,
            "meta": {
                "model": self.model,
                "prompt": self.prompt,
                "populations": dict(self.populations),
                "config": dict(self.config),
                "best": self.best(),
                "is_synthetic": bool(self.is_synthetic),
                "data_source": str(self.data_source),
                "provenance": dict(self.provenance),
            },
            "frames": frames,
        }

    def save(self, path: str | Path) -> Path:
        """Write ``fly-cns-replay/1`` JSON (creates parent dirs)."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return dest

    def __len__(self) -> int:
        return len(self.frames)

    def clear(self) -> None:
        self.frames.clear()
        self._best = {"config": None, "reward": float("-inf")}


__all__ = [
    "REPLAY_SCHEMA",
    "DEFAULT_POPULATIONS",
    "ReplayRecorder",
]
