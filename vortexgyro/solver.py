"""Gyrokinetic Vlasov-Poisson time-integration solver with 4th-order Runge-Kutta."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

from .grid import SpectralGrid2D
from .gyroaverage import GyroaveragingOperator
from .poisson import GyrokineticPoissonSolver


@dataclass
class SimulationConfig:
    """Configuration parameters for gyrokinetic turbulence simulation."""
    nx: int = 64
    ny: int = 64
    lx: float = 64.0              # Domain length x [rho_s]
    ly: float = 64.0              # Domain length y [rho_s]
    dt: float = 0.05              # Normalized time step dt [c_s / L_ref]
    kappa_n: float = 1.0          # Density gradient drive 1 / L_n
    kappa_T: float = 3.0          # Temperature gradient drive 1 / L_T
    tau: float = 1.0              # Temperature ratio Ti / Te
    rho_i: float = 1.0            # Normalized ion gyroradius
    hyper_visc: float = 1e-4      # Hyper-viscosity damping coefficient
    use_pade: bool = False


@dataclass
class SimulationState:
    """Instantaneous state of the gyrokinetic simulation."""
    step: int
    time: float
    g_hat: np.ndarray             # Guiding center density perturbation (Fourier)
    phi_hat: np.ndarray           # Electrostatic potential (Fourier)
    phi_real: np.ndarray          # Electrostatic potential (Real space)
    g_real: np.ndarray            # Guiding center density (Real space)


class GyrokineticSolver:
    """Non-linear 2D Gyrokinetic turbulence solver."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.grid = SpectralGrid2D(config.nx, config.ny, config.lx, config.ly)
        self.gyro = GyroaveragingOperator(config.rho_i, config.use_pade)
        self.poisson = GyrokineticPoissonSolver(self.grid, self.gyro, config.tau)

        # Background gradient drive term: omega_star_T
        # ITG drive: -i * ky * (kappa_n + kappa_T) * J0(kperp * rho_i)
        kperp2 = self.grid.kperp2
        j0 = self.gyro.j0_factor(kperp2)
        total_drive = config.kappa_n + config.kappa_T
        self.drive_operator = -1j * self.grid.KY * total_drive * j0

        # Hyper-dissipation operator to prevent small-scale spectral pileup
        self.dissipation = -config.hyper_visc * (self.grid.kperp2**2)

        # State initialization
        self.step_count = 0
        self.time = 0.0
        self.g_hat = np.zeros((config.nx, config.ny), dtype=complex)
        self.phi_hat = np.zeros((config.nx, config.ny), dtype=complex)

    def initialize_itg_seed(self, amplitude: float = 1e-3, target_ky: float = 0.3) -> None:
        """Seeds small amplitude perturbation in unstable drift wave modes to initiate ITG."""
        # Find mode index closest to optimal growth wavenumber ky ~ 0.3 / rho_s
        ky_diff = np.abs(self.grid.ky - target_ky)
        ky_idx = np.argmin(ky_diff)

        # Symmetric random perturbation in x for target ky mode
        np.random.seed(42)
        phase_noise = np.random.uniform(0, 2 * np.pi, self.config.nx)
        for i in range(self.config.nx):
            self.g_hat[i, ky_idx] = amplitude * np.exp(1j * phase_noise[i])
            # Hermite symmetry for real fields in real space
            neg_idx = (-ky_idx) % self.config.ny
            self.g_hat[i, neg_idx] = np.conj(self.g_hat[i, ky_idx])

        self.phi_hat = self.poisson.solve(self.g_hat)

    def compute_poisson_bracket(self, phi_hat: np.ndarray, g_hat: np.ndarray) -> np.ndarray:
        """Computes non-linear advection bracket [phi, g] = d_x phi * d_y g - d_y phi * d_x g

        using pseudo-spectral method with 2/3 de-aliasing.
        """
        # Gradients of phi
        dphi_dx = self.grid.ifft2(self.grid.diff_x(phi_hat))
        dphi_dy = self.grid.ifft2(self.grid.diff_y(phi_hat))

        # Gradients of g
        dg_dx = self.grid.ifft2(self.grid.diff_x(g_hat))
        dg_dy = self.grid.ifft2(self.grid.diff_y(g_hat))

        # Real space non-linear term: [phi, g]
        bracket_real = dphi_dx * dg_dy - dphi_dy * dg_dx

        # Transform back to Fourier space with de-aliasing filter
        return self.grid.fft2(bracket_real)

    def rhs(self, g_hat: np.ndarray) -> np.ndarray:
        """Evaluates dg_hat/dt:

        dg/dt = Drive(phi) - [phi, g] + Dissipation(g)
        """
        phi_hat = self.poisson.solve(g_hat)

        # Linear background gradient drive
        drive_term = self.drive_operator * phi_hat

        # Non-linear E x B advection
        non_linear_term = self.compute_poisson_bracket(phi_hat, g_hat)

        # Dissipation
        diss_term = self.dissipation * g_hat

        dg_dt = drive_term - non_linear_term + diss_term
        return dg_dt * self.grid.dealias_mask

    def step(self) -> None:
        """Advances simulation by time step dt using 4th-order Runge-Kutta (RK4)."""
        dt = self.config.dt
        g = self.g_hat

        k1 = self.rhs(g)
        k2 = self.rhs(g + 0.5 * dt * k1)
        k3 = self.rhs(g + 0.5 * dt * k2)
        k4 = self.rhs(g + dt * k3)

        self.g_hat = g + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.phi_hat = self.poisson.solve(self.g_hat)

        self.step_count += 1
        self.time += dt

    def run(self, steps: int) -> None:
        """Executes specified number of time steps."""
        for _ in range(steps):
            self.step()

    def current_state(self) -> SimulationState:
        """Returns the current simulation state including real space projections."""
        phi_real = self.grid.ifft2(self.phi_hat)
        g_real = self.grid.ifft2(self.g_hat)
        return SimulationState(
            step=self.step_count,
            time=self.time,
            g_hat=np.copy(self.g_hat),
            phi_hat=np.copy(self.phi_hat),
            phi_real=phi_real,
            g_real=g_real,
        )
