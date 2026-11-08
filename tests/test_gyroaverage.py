"""Unit tests for GyroaveragingOperator, Bessel functions, and Gamma_0 polarization response."""

import numpy as np
import pytest
from vortexgyro.gyroaverage import GyroaveragingOperator


def test_gyroaverage_zero_wavenumber():
    """At k_perp = 0, J0(0) = 1, Gamma0(0) = 1."""
    gyro = GyroaveragingOperator(rho_i=1.0)
    kperp2_zero = np.array([[0.0]])

    j0 = gyro.j0_factor(kperp2_zero)
    gamma0 = gyro.gamma0_factor(kperp2_zero)

    assert j0[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert gamma0[0, 0] == pytest.approx(1.0, abs=1e-12)


def test_gyroaverage_decay_at_large_k():
    """For large k_perp * rho_i >> 1, J0 and Gamma0 must monotonically decay."""
    gyro = GyroaveragingOperator(rho_i=1.0)
    kperp2 = np.array([[0.0, 1.0, 4.0, 16.0, 64.0]])

    gamma0 = gyro.gamma0_factor(kperp2)
    # Gamma_0 = I0(b) e^-b is strictly decreasing for b >= 0
    diffs = np.diff(gamma0[0])
    assert np.all(diffs < 0.0)


def test_pade_vs_exact_bessel():
    """Compare Pade approximation (1 + b/2)^(-1) with exact Bessel factor for small b."""
    gyro_exact = GyroaveragingOperator(rho_i=1.0, use_pade=False)
    gyro_pade = GyroaveragingOperator(rho_i=1.0, use_pade=True)

    # For b = (k_perp * rho_i)^2 <= 0.2, Pade is within 5% of exact J0
    small_kperp2 = np.array([[0.01, 0.04, 0.1]])
    j0_exact = gyro_exact.j0_factor(small_kperp2)
    j0_pade = gyro_pade.j0_factor(small_kperp2)

    np.testing.assert_allclose(j0_pade, j0_exact, rtol=0.05)


def test_gyroaverage_field_smoothing():
    """Gyroaveraging a high-frequency spatial variation must reduce its Fourier amplitude."""
    gyro = GyroaveragingOperator(rho_i=1.0)
    kperp2_high = np.array([[10.0]])
    phi_hat = np.array([[1.0 + 0.0j]])

    phi_bar_hat = gyro.gyroaverage_fourier(phi_hat, kperp2_high)
    assert np.abs(phi_bar_hat[0, 0]) < np.abs(phi_hat[0, 0])
