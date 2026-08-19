"""UH-CMA-ES GPU — Inversão sísmica paralelizada (PyTorch)."""

from .seismic_physics import SeismicPhysicsGPU
from .uhcmaes_gpu import run_uhcmaes_gpu, uncertainty_measurement_gpu

__version__ = "1.0.0"
__all__ = ["SeismicPhysicsGPU", "run_uhcmaes_gpu", "uncertainty_measurement_gpu"]
