"""Physical diagnostics for gyrokinetic plasma turbulence and zonal flows."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

from .grid import SpectralGrid2D


@dataclass
class DiagnosticSnapshot:
    """Quantitative diagnostics at a single time instant."""
    time: float
    step: int
    electrostatic_energy: float
    zonal_energy: float
    zonal_fraction: float
    turbulent_heat_flux: float
    turbulent_particle_flux: float
    max_shearing_rate: float
    rms_potential: float


class TurbulenceDiagnostics:
    """Calculates physical fluxes, zonal shearing rates, and spectral energy partitions."""

    def __init__(self, grid: SpectralGrid2D):
        self.grid = grid

    def compute_electrostatic_energy(self, phi_hat: np.ndarray) -> float:
        r"""Calculates volume-integrated electrostatic field energy:

        W_E = \frac{1}{2 V} \int |\nabla_\perp \phi|^2 d^2x = \frac{1}{2} \sum_{kx, ky} k_\perp^2 |\hat{\phi}(kx, ky)|^2
        """
        energy_spectrum = 0.5 * self.grid.kperp2 * (np.abs(phi_hat) ** 2)
        # Avoid double-counting or negative frequencies: standard Parseval on real space fields
        phi_real = self.grid.ifft2(phi_hat)
        dphi_dx = self.grid.ifft2(self.grid.diff_x(phi_hat))
        dphi_dy = self.grid.ifft2(self.grid.diff_y(phi_hat))
        we_real = 0.5 * np.mean(dphi_dx**2 + dphi_dy**2)
        return float(we_real)

    def compute_zonal_partition(self, phi_hat: np.ndarray) -> Tuple[float, float, float]:
        r"""Decomposes electrostatic potential into zonal (ky = 0) and drift-wave (ky != 0) components.

        Returns:
            (E_total, E_zonal, zonal_fraction)
        """
        total_energy = self.compute_electrostatic_energy(phi_hat)
        if total_energy <= 1e-18:
            return 0.0, 0.0, 0.0

        # Zonal potential has ky = 0
        zonal_phi_hat = np.zeros_like(phi_hat)
        zonal_phi_hat[:, 0] = phi_hat[:, 0]

        dphi_dx = self.grid.ifft2(self.grid.diff_x(zonal_phi_hat))
        zonal_energy = float(0.5 * np.mean(dphi_dx**2))
        fraction = zonal_energy / total_energy if total_energy > 0 else 0.0

        return total_energy, zonal_energy, float(fraction)

    def compute_turbulent_heat_flux(self, phi_hat: np.ndarray, g_hat: np.ndarray) -> float:
        r"""Calculates radial turbulent convective heat/energy transport:

        Q_i = \langle v_{Ex} \cdot g \rangle = \frac{1}{V} \int \left( -\frac{\partial \phi}{\partial y} \right) g(x, y) d^2x
        """
        # Radial ExB velocity: v_Ex = -dphi/dy
        v_ex = self.grid.ifft2(-self.grid.diff_y(phi_hat))
        g_real = self.grid.ifft2(g_hat)
        heat_flux = float(np.mean(v_ex * g_real))
        return heat_flux

    def compute_turbulent_particle_flux(self, phi_hat: np.ndarray, g_hat: np.ndarray) -> float:
        r"""Calculates anomalous particle flux driven by electrostatic drift fluctuations."""
        # In single-field approximation, particle flux follows correlation between v_Ex and density perturbation
        v_ex = self.grid.ifft2(-self.grid.diff_y(phi_hat))
        density_perturbation = self.grid.ifft2(g_hat)
        gamma = float(np.mean(v_ex * density_perturbation))
        return gamma

    def compute_zonal_shearing_rate(self, phi_hat: np.ndarray) -> float:
        r"""Calculates maximum radial zonal flow shearing rate:

        \omega_{E \times B} = \left| \frac{\partial^2 \phi_{ky=0}}{\partial x^2} \right|_{\max}
        """
        zonal_phi_hat = np.zeros_like(phi_hat)
        zonal_phi_hat[:, 0] = phi_hat[:, 0]

        # Second radial derivative
        d2phi_dx2 = self.grid.ifft2(self.grid.diff_x(self.grid.diff_x(zonal_phi_hat)))
        return float(np.max(np.abs(d2phi_dx2)))

    def evaluate(self, time: float, step: int, phi_hat: np.ndarray, g_hat: np.ndarray) -> DiagnosticSnapshot:
        """Evaluates full suite of turbulence diagnostics for the given state."""
        e_total, e_zonal, z_frac = self.compute_zonal_partition(phi_hat)
        q_heat = self.compute_turbulent_heat_flux(phi_hat, g_hat)
        gamma_p = self.compute_turbulent_particle_flux(phi_hat, g_hat)
        omega_e = self.compute_zonal_shearing_rate(phi_hat)

        phi_real = self.grid.ifft2(phi_hat)
        rms_phi = float(np.sqrt(np.mean(phi_real**2)))

        return DiagnosticSnapshot(
            time=time,
            step=step,
            electrostatic_energy=e_total,
            zonal_energy=e_zonal,
            zonal_fraction=z_frac,
            turbulent_heat_flux=q_heat,
            turbulent_particle_flux=gamma_p,
            max_shearing_rate=omega_e,
            rms_potential=rms_phi,
        )


class DiagnosticTracker:
    """Historical telemetry container for tracking and fitting plasma turbulence growth and saturation."""

    def __init__(self, grid: SpectralGrid2D):
        self.diagnostics = TurbulenceDiagnostics(grid)
        self.snapshots: List[DiagnosticSnapshot] = []

    def record(self, time: float, step: int, phi_hat: np.ndarray, g_hat: np.ndarray) -> DiagnosticSnapshot:
        """Records diagnostic metrics at current time step."""
        snap = self.diagnostics.evaluate(time, step, phi_hat, g_hat)
        self.snapshots.append(snap)
        return snap

    def get_times(self) -> np.ndarray:
        return np.array([s.time for s in self.snapshots])

    def get_energies(self) -> np.ndarray:
        return np.array([s.electrostatic_energy for s in self.snapshots])

    def get_zonal_fractions(self) -> np.ndarray:
        return np.array([s.zonal_fraction for s in self.snapshots])

    def get_heat_fluxes(self) -> np.ndarray:
        return np.array([s.turbulent_heat_flux for s in self.snapshots])

    def fit_linear_growth_rate(self, t_start: float, t_end: float) -> Tuple[float, float]:
        r"""Fits exponential growth rate \gamma where E(t) \propto \exp(2 \gamma t).

        Returns:
            (gamma, r_squared)
        """
        times = self.get_times()
        energies = self.get_energies()

        mask = (times >= t_start) & (times <= t_end) & (energies > 1e-16)
        if np.sum(mask) < 3:
            return 0.0, 0.0

        t_sub = times[mask]
        log_e = np.log(energies[mask])

        # Linear regression: log(E) = 2 * gamma * t + c
        p = np.polyfit(t_sub, log_e, 1)
        two_gamma = p[0]
        gamma = two_gamma / 2.0

        # R-squared calculation
        fit_vals = np.polyval(p, t_sub)
        ss_res = np.sum((log_e - fit_vals) ** 2)
        ss_tot = np.sum((log_e - np.mean(log_e)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        return float(gamma), float(r2)
