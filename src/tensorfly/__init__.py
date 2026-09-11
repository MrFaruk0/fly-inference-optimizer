"""TensorFly: MaleCNS-scale connectome inference scaffolding.

Notebook-facing API::

    from tensorfly import get_runtime, MaleCNSSimulation, SimulationConfig, run_benchmark

    rt = get_runtime()                       # GPU detect + A100/L4/T4/CPU profile
    sim = MaleCNSSimulation(SimulationConfig(num_neurons=5000, num_edges=20000))
    sim.build()                              # synthetic scaffold unless real source given
    result = run_benchmark(sim, steps=5, repeats=2)
"""

from .benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    compare_configs,
    run_benchmark,
    save_comparison,
    time_inference,
)
from .inference import (
    FALLBACK_MESSAGE,
    MODEL_4B,
    MODEL_9B,
    InferenceConfig,
    QwenInference,
    get_default_model,
    resolve_device,
    resolve_dtype,
    select_qwen_model,
)
from .replay import REPLAY_SCHEMA, ReplayRecorder
from .runtime import (
    PROFILES,
    DeviceProfile,
    GPUInfo,
    Runtime,
    SystemInfo,
    detect_gpu,
    detect_system,
    get_runtime,
    select_profile,
)
from .simulation import (
    N_MALECNS_EDGES,
    N_MALECNS_NEURONS,
    ActivitySnapshot,
    MaleCNSSimulation,
    SimulationConfig,
)

__all__ = [
    "PROFILES",
    "DeviceProfile",
    "GPUInfo",
    "Runtime",
    "SystemInfo",
    "detect_gpu",
    "detect_system",
    "select_profile",
    "get_runtime",
    "N_MALECNS_NEURONS",
    "N_MALECNS_EDGES",
    "ActivitySnapshot",
    "MaleCNSSimulation",
    "SimulationConfig",
    "BenchmarkConfig",
    "BenchmarkResult",
    "compare_configs",
    "run_benchmark",
    "save_comparison",
    "time_inference",
    "MODEL_9B",
    "MODEL_4B",
    "FALLBACK_MESSAGE",
    "InferenceConfig",
    "QwenInference",
    "select_qwen_model",
    "get_default_model",
    "resolve_dtype",
    "resolve_device",
    "REPLAY_SCHEMA",
    "ReplayRecorder",
]

__version__ = "0.1.0"
