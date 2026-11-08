"""VortexGyro: Gyrokinetic Tokamak Micro-Turbulence and Plasma Kinetics Simulator."""

__version__ = "0.1.0"

from .grid import SpectralGrid2D
from .gyroaverage import GyroaveragingOperator
from .poisson import GyrokineticPoissonSolver
from .solver import GyrokineticSolver, SimulationConfig, SimulationState
from .diagnostics import TurbulenceDiagnostics

__all__ = [
    "SpectralGrid2D",
    "GyroaveragingOperator",
    "GyrokineticPoissonSolver",
    "GyrokineticSolver",
    "SimulationConfig",
    "SimulationState",
    "TurbulenceDiagnostics",
]
