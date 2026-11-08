"""Gyrokinetic quasi-neutrality Poisson field solver with zonal flow shielding."""

from __future__ import annotations
import numpy as np
from .grid import SpectralGrid2D
from .gyroaverage import GyroaveragingOperator


class GyrokineticPoissonSolver:
    """Solves the gyrokinetic quasi-neutrality equation in 2D Fourier space."""

    def __init__(
        self,
        grid: SpectralGrid2D,
        gyro: GyroaveragingOperator,
        tau: float = 1.0,  # Ti / Te
    ):
        self.grid = grid
        self.gyro = gyro
        self.tau = tau

        # Precompute denominator response tensor in Fourier space
        self._build_response_operator()

    def _build_response_operator(self) -> None:
        """Precomputes quasi-neutrality dielectric denominator:

        For k_y != 0 (drift waves): D(k) = 1 + tau * (1 - Gamma_0(b))
        For k_y == 0 (zonal flows): D(k) = tau * (1 - Gamma_0(b))
        """
        kperp2 = self.grid.kperp2
        pol = self.gyro.polarization_density_shielding(kperp2)

        # Standard drift wave denominator (adiabatic electrons)
        denom_drift = 1.0 + self.tau * pol

        # Zonal flow denominator (non-adiabatic electrons, ky = 0)
        denom_zonal = self.tau * pol

        # Combined denominator mask
        is_zonal = np.abs(self.grid.KY) < 1e-12
        self.denom = np.where(is_zonal, denom_zonal, denom_drift)

        # Handle k = 0 mode (gauge condition phi_0 = 0)
        self.denom[0, 0] = 1.0

        # Precompute J0 factor
        self.j0 = self.gyro.j0_factor(kperp2)

    def solve(self, g_hat: np.ndarray) -> np.ndarray:
        """Solves for electrostatic potential phi_hat given guiding-center density g_hat."""
        rhs = self.j0 * g_hat
        phi_hat = rhs / self.denom

        # Ensure k=(0,0) mode is zero (gauge invariance)
        phi_hat[0, 0] = 0.0 + 0.0j

        # Apply de-aliasing mask
        return phi_hat * self.grid.dealias_mask

    def get_exb_velocities(self, phi_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculates real space E x B drift velocity components:

        v_Ex = - d phi / dy
        v_Ey =   d phi / dx
        """
        dphi_dx_hat = self.grid.diff_x(phi_hat)
        dphi_dy_hat = self.grid.diff_y(phi_hat)

        v_ex = -self.grid.ifft2(dphi_dy_hat)
        v_ey = self.grid.ifft2(dphi_dx_hat)
        return v_ex, v_ey
