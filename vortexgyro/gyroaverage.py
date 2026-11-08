"""Gyroaveraging operators and Bessel functions for finite Larmor radius (FLR) effects."""

from __future__ import annotations
import numpy as np
from scipy.special import j0, i0e


class GyroaveragingOperator:
    """Computes gyroaveraging and polarization shielding in Fourier space."""

    def __init__(self, rho_i: float = 1.0, use_pade: bool = False):
        self.rho_i = rho_i
        self.use_pade = use_pade

    def j0_factor(self, kperp2: np.ndarray) -> np.ndarray:
        """Evaluates J_0(k_perp * rho_i) in Fourier space.

        Uses exact Bessel function j0 or Padé approximation 1 / (1 + 0.5 * k_perp^2 * rho_i^2).
        """
        b = kperp2 * (self.rho_i**2)
        if self.use_pade:
            return 1.0 / (1.0 + 0.5 * b)
        kperp_rho = np.sqrt(np.maximum(0.0, b))
        return j0(kperp_rho)

    def gamma0_factor(self, kperp2: np.ndarray) -> np.ndarray:
        """Evaluates Gamma_0(b) = I_0(b) * exp(-b) where b = k_perp^2 * rho_i^2.

        Represents ion polarization shielding.
        """
        b = kperp2 * (self.rho_i**2)
        if self.use_pade:
            return 1.0 / (1.0 + b)
        # scipy i0e(b) computes i0(b) * exp(-abs(b)) directly without overflow
        return i0e(b)

    def polarization_density_shielding(self, kperp2: np.ndarray) -> np.ndarray:
        """Computes 1 - Gamma_0(b) = 1 - I_0(b) * exp(-b)."""
        b = kperp2 * (self.rho_i**2)
        if self.use_pade:
            return b / (1.0 + b)
        return 1.0 - i0e(b)

    def gyroaverage_fourier(self, phi_hat: np.ndarray, kperp2: np.ndarray) -> np.ndarray:
        """Applies J0(k_perp * rho_i) gyroaveraging to a field in Fourier space."""
        return phi_hat * self.j0_factor(kperp2)
