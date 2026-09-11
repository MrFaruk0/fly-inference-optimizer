"""TensorFly runtime: GPU detection, device profiles, and metadata recording.

Notebook-facing entry points:
    >>> from tensorfly.runtime import get_runtime, Runtime
    >>> rt = get_runtime()
    >>> rt.profile_name
    >>> rt.save_metadata("runtime_metadata.json")
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    """Detected GPU device (or lack thereof)."""

    present: bool
    name: str = "CPU (no CUDA GPU detected)"
    index: int = -1
    vram_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    torch_version: Optional[str] = None
    torch_cuda_available: bool = False
    detail: str = ""


@dataclass
class SystemInfo:
    """Host system summary."""

    ram_gb: Optional[float] = None
    cpu_count: int = 0
    platform: str = ""
    python_version: str = ""


@dataclass(frozen=True)
class DeviceProfile:
    """Inference configuration profile for a known accelerator class."""

    name: str
    # Substrings matched (case-insensitive) against the detected GPU name.
    gpu_match: tuple[str, ...]
    # Representative hardware VRAM in GB (for reference only).
    reference_vram_gb: float
    # Workload batch size is deliberately absent: it is a validated actual
    # InferenceConfig knob, not a guessed GPU profile constant.
    dtype: str
    mixed_precision: bool
    num_workers: int
    # Upper bound hint for chunked full-network passes.
    max_neurons_per_chunk: int
    description: str = ""


# Profiles required by the product spec: A100 / L4 / T4 + deterministic CPU
# fallback. Ordered from most to least capable; fallback walks this list.
PROFILES: Dict[str, DeviceProfile] = {
    "A100": DeviceProfile(
        name="A100",
        gpu_match=("A100",),
        reference_vram_gb=40.0,
        dtype="bfloat16",
        mixed_precision=True,
        num_workers=8,
        max_neurons_per_chunk=65536,
        description="NVIDIA A100 40/80GB datacenter GPU. Full-throughput profile.",
    ),
    "L4": DeviceProfile(
        name="L4",
        gpu_match=("L4",),
        reference_vram_gb=24.0,
        dtype="float16",
        mixed_precision=True,
        num_workers=4,
        max_neurons_per_chunk=32768,
        description="NVIDIA L4 24GB Ada Lovelace GPU. Balanced cost/perf profile.",
    ),
    "T4": DeviceProfile(
        name="T4",
        gpu_match=("T4", "TESLA T4"),
        reference_vram_gb=16.0,
        dtype="float16",
        mixed_precision=True,
        num_workers=4,
        max_neurons_per_chunk=16384,
        description="NVIDIA T4 16GB Turing GPU. Lowest-common-denominator GPU profile.",
    ),
    "CPU": DeviceProfile(
        name="CPU",
        gpu_match=(),
        reference_vram_gb=0.0,
        dtype="float32",
        mixed_precision=False,
        num_workers=2,
        max_neurons_per_chunk=8192,
        description="CPU fallback. No CUDA GPU detected or no known GPU matched.",
    ),
}

# Ordered fallback chain when the detected GPU name matches nothing known.
FALLBACK_CHAIN: List[str] = ["A100", "L4", "T4", "CPU"]

# The safe generic-GPU fallback: an unknown CUDA device drops to the smallest
# known GPU profile (T4) instead of assuming A100-class memory.
UNKNOWN_GPU_FALLBACK = "T4"
NO_GPU_FALLBACK = "CPU"


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _detect_torch() -> tuple[Optional[str], bool]:
    """Return (torch_version, torch_cuda_available) without hard dependency."""
    try:
        import torch  # type: ignore[import]

        version: Optional[str] = request_torch_version(torch)
        cuda_ok = bool(torch.cuda.is_available())
        return version, cuda_ok
    except Exception:
        return None, False


def request_torch_version(torch_module: Any) -> Optional[str]:
    return str(getattr(torch_module, "__version__", "unknown"))


def _detect_cuda_version_nvcc() -> Optional[str]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return None
    try:
        out = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, timeout=10
        )
        text = (out.stdout or "") + (out.stderr or "")
        for line in text.splitlines():
            if "release" in line.lower():
                # e.g. "Cuda compilation tools, release 12.4, V12.4.99"
                parts = line.strip().split("release")
                if len(parts) > 1:
                    return parts[1].strip().split(",")[0].strip()
        return None
    except Exception:
        return None


def _detect_gpu_nvidia_smi() -> tuple[str, Optional[float]]:
    """Return (gpu_name, vram_gb) from nvidia-smi, or ('', None) if absent."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return "", None
    try:
        out = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return "", None
        first = out.stdout.strip().splitlines()[0]
        name_part, _, mem_part = first.partition(",")
        name = name_part.strip() or ""
        try:
            mem_mb = float(mem_part.strip())
            vram_gb = round(mem_mb / 1024.0, 2)
        except ValueError:
            vram_gb = None
        return name, vram_gb
    except Exception:
        return "", None


def _detect_gpu_torch() -> tuple[str, Optional[float], int]:
    """Best-effort GPU name/VRAM via torch (index 0). Returns ('', None, -1)."""
    try:
        import torch  # type: ignore[import]

        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return "", None, -1
        name = str(torch.cuda.get_device_name(0))
        try:
            total = int(torch.cuda.get_device_properties(0).total_memory)
            vram_gb: Optional[float] = round(total / (1024.0**3), 2)
        except Exception:
            vram_gb = None
        return name, vram_gb, 0
    except Exception:
        return "", None, -1


def detect_gpu() -> GPUInfo:
    """Detect the primary GPU, combining torch + nvidia-smi + nvcc probes.

    Never raises: absence of torch / drivers yields ``present=False``.
    """
    torch_version, torch_cuda = _detect_torch()
    cuda_version = _detect_cuda_version_nvcc()

    name, vram, index = _detect_gpu_torch()
    if not name:
        smi_name, smi_vram = _detect_gpu_nvidia_smi()
        if smi_name:
            name, vram, index = smi_name, smi_vram, 0

    if name and (torch_cuda or vram is not None or shutil.which("nvidia-smi")):
        # torch may be missing while nvidia-smi exists (e.g. CI images);
        # still report the device as present when a name was resolved.
        present = True
    else:
        present = False

    if not present:
        return GPUInfo(
            present=False,
            name="CPU (no CUDA GPU detected)",
            index=-1,
            vram_gb=None,
            cuda_version=cuda_version,
            torch_version=torch_version,
            torch_cuda_available=torch_cuda,
            detail="No CUDA device reported by torch or nvidia-smi.",
        )

    return GPUInfo(
        present=True,
        name=name,
        index=index if index >= 0 else 0,
        vram_gb=vram,
        cuda_version=cuda_version,
        torch_version=torch_version,
        torch_cuda_available=torch_cuda,
        detail="Detected via torch/nvidia-smi.",
    )


def detect_system() -> SystemInfo:
    """Detect host RAM / CPU / platform without third-party dependencies."""
    ram_gb: Optional[float] = None
    try:
        # Linux /proc/meminfo fast path.
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemTotal:"):
                    kb = float(line.split()[1])
                    ram_gb = round(kb / (1024.0**2), 2)
                    break
    except Exception:
        pass
    if ram_gb is None:
        try:
            import psutil  # type: ignore[import]

            ram_gb = round(float(psutil.virtual_memory().total) / (1024.0**3), 2)
        except Exception:
            ram_gb = None
    return SystemInfo(
        ram_gb=ram_gb,
        cpu_count=os.cpu_count() or 0,
        platform=platform.platform(),
        python_version=platform.python_version(),
    )


# ---------------------------------------------------------------------------
# Profile selection (explicit fallback semantics)
# ---------------------------------------------------------------------------

def select_profile(gpu_name: str = "", gpu_present: bool = False) -> DeviceProfile:
    """Select the A100/L4/T4/CPU profile for a detected GPU name.

    Rules (exactly as specified):
      * name contains "A100" -> A100 profile.
      * name contains "L4"   -> L4 profile.
      * name contains "T4"   -> T4 profile.
      * unknown CUDA GPU name -> explicit fallback print + T4 profile
        (safe lowest-common GPU denominator).
      * no GPU at all        -> explicit fallback print + CPU profile.
    """
    upper = (gpu_name or "").upper()
    for key in ("A100", "L4", "T4"):
        profile = PROFILES[key]
        if any(token.upper() in upper for token in profile.gpu_match):
            return profile

    if gpu_present:
        print(
            f"[tensorfly.runtime] Unknown GPU '{gpu_name}'. "
            f"Falling back to '{UNKNOWN_GPU_FALLBACK}' profile "
            f"(safe default for unknown CUDA devices)."
        )
        return PROFILES[UNKNOWN_GPU_FALLBACK]

    print(
        "[tensorfly.runtime] No CUDA GPU detected. "
        f"Falling back to '{NO_GPU_FALLBACK}' profile (CPU inference)."
    )
    return PROFILES[NO_GPU_FALLBACK]


# ---------------------------------------------------------------------------
# Runtime facade (notebook-facing)
# ---------------------------------------------------------------------------

@dataclass
class Runtime:
    """Notebook-facing runtime handle: detection + profile + metadata."""

    gpu: GPUInfo
    system: SystemInfo
    profile: DeviceProfile
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def profile_name(self) -> str:
        return self.profile.name

    @property
    def device(self) -> str:
        """Convenience device string ('cuda:0' or 'cpu').

        Returns a ``cuda`` label only when torch can actually use it;
        otherwise ``cpu`` (GPU presence is still recorded in ``gpu``).
        """
        if self.gpu.present and self.gpu.torch_cuda_available:
            return f"cuda:{max(self.gpu.index, 0)}"
        return "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu": asdict(self.gpu),
            "system": asdict(self.system),
            "profile": asdict(self.profile),
            "device": self.device,
            "created_at": self.created_at,
            "extra": dict(self.extra),
        }

    def save_metadata(self, path: str | Path) -> Path:
        """Record runtime metadata as JSON (creates parent dirs)."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return dest

    def startup_summary(self, model: Optional[str] = None) -> str:
        """Build a one-block startup summary string.

        Includes GPU name, VRAM, RAM, CUDA, torch, profile, and model so
        notebook boot logs show exactly what will run where. Never raises
        for missing torch/CUDA: unavailable values render as ``n/a``.

        The required Qwen fallback notice (when the profile resolves to
        4B) is never swallowed: it is printed to stdout even when the
        model id is auto-resolved here.
        """
        model_id = model or self.extra.get("model")
        if model_id is None:
            try:
                from .inference import select_qwen_model

                # Do NOT redirect stdout: select_qwen_model prints the
                # exact fallback line for 4B profiles and that notice is
                # required to stay visible in boot logs.
                model_id = select_qwen_model(
                    self.profile.name, getattr(self.gpu, "vram_gb", None)
                )
            except Exception:
                model_id = f"profile-default:{self.profile.name}"
        vram = (
            f"{self.gpu.vram_gb} GB"
            if self.gpu.vram_gb is not None
            else "n/a"
        )
        ram = (
            f"{self.system.ram_gb} GB"
            if self.system.ram_gb is not None
            else "n/a"
        )
        cuda = self.gpu.cuda_version or "n/a"
        torch_ver = self.gpu.torch_version or "n/a"
        lines = [
            "[tensorfly] startup summary",
            f"GPU: {self.gpu.name} (present={self.gpu.present})",
            f"VRAM: {vram}",
            f"RAM: {ram} (cpus={self.system.cpu_count})",
            f"CUDA: {cuda} (torch_cuda_available={self.gpu.torch_cuda_available})",
            f"torch: {torch_ver}",
            (
                f"profile: {self.profile.name} "
                f"(dtype={self.profile.dtype}; batch size is selected by the experiment)"
            ),
            f"model: {model_id}",
            f"device: {self.device}",
        ]
        return "\n".join(lines)

    def print_startup_summary(self, model: Optional[str] = None) -> str:
        """Print :meth:`startup_summary` and return it (for tests/logs)."""
        summary = self.startup_summary(model=model)
        print(summary)
        return summary


def get_runtime(
    gpu_name_override: Optional[str] = None,
    gpu_present_override: Optional[bool] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Runtime:
    """Build the notebook-facing :class:`Runtime`.

    Args:
        gpu_name_override: force a GPU name (useful for tests / docs).
        gpu_present_override: force GPU presence flag.
        extra: free-form metadata merged into the record.
    """
    gpu = detect_gpu()
    if gpu_name_override is not None:
        gpu.name = gpu_name_override
    if gpu_present_override is not None:
        gpu.present = gpu_present_override
        if not gpu.present and gpu.index != -1:
            gpu.index = -1

    profile = select_profile(gpu_name=gpu.name, gpu_present=gpu.present)
    system = detect_system()
    return Runtime(gpu=gpu, system=system, profile=profile, extra=dict(extra or {}))


__all__ = [
    "PROFILES",
    "FALLBACK_CHAIN",
    "UNKNOWN_GPU_FALLBACK",
    "NO_GPU_FALLBACK",
    "GPUInfo",
    "SystemInfo",
    "DeviceProfile",
    "Runtime",
    "detect_gpu",
    "detect_system",
    "select_profile",
    "get_runtime",
]
