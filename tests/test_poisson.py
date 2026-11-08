"""Unit tests for GyrokineticPoissonSolver, dielectric shielding, and zonal flow response."""

import numpy as np
import pytest
from vortexgyro.grid import SpectralGrid2D
from vortexgyro.gyroaverage import GyroaveragingOperator
from vortexgyro.poisson import GyrokineticPoissonSolver


def test_poisson_solver_initialization():
    grid = SpectralGrid2D(nx=32, ny=32, lx=16.0, ly=16.0)
    gyro = GyroaveragingOperator(rho_i=1.0)
    solver = GyrokineticPoissonSolver(grid, gyro, tau=1.0)

    # Dielectric denominator must be strictly positive everywhere except potentially at DC (0,0)
    assert solver.denom[0, 0] > 0.0
    assert np.all(solver.denom > 0.0)


def test_zonal_vs_drift_shielding():
    """Verify that zonal modes (ky = 0) do not suffer adiabatic electron shielding."""
    grid = SpectralGrid2D(nx=32, ny=32, lx=16.0, ly=16.0)
    gyro = GyroaveragingOperator(rho_i=1.0)
    solver = GyrokineticPoissonSolver(grid, gyro, tau=1.0)

    # For ky = 0, D_zonal = tau * (1 - Gamma_0(b))
    # For ky != 0, D_drift = 1 + tau * (1 - Gamma_0(b))
    # Therefore D_drift > D_zonal for the same kx > 0
    kx_idx = 1
    d_zonal = solver.denom[kx_idx, 0]

    # Find a ky != 0 index
    ky_idx = 1
    d_drift = solver.denom[kx_idx, ky_idx]

    # Because D_drift has +1 (adiabatic electron response), it is larger than D_zonal
    assert d_drift > d_zonal


def test_poisson_linearity():
    """Verify solver satisfies linearity: Solve(a * g1 + b * g2) = a * Solve(g1) + b * Solve(g2)."""
    grid = SpectralGrid2D(nx=32, ny=32, lx=16.0, ly=16.0)
    gyro = GyroaveragingOperator(rho_i=1.0)
    solver = GyrokineticPoissonSolver(grid, gyro, tau=1.0)

    np.random.seed(123)
    g1 = np.random.randn(32, 32) + 1j * np.random.randn(32, 32)
    g2 = np.random.randn(32, 32) + 1j * np.random.randn(32, 32)

    a, b = 2.5, -1.7
    phi_combined = solver.solve(a * g1 + b * g2)
    phi_linear = a * solver.solve(g1) + b * solver.solve(g2)

    np.testing.assert_allclose(phi_combined, phi_linear, atol=1e-12)


def test_zero_charge_density_yields_zero_potential():
    """Zero density perturbation must yield zero potential."""
    grid = SpectralGrid2D(nx=32, ny=32, lx=16.0, ly=16.0)
    gyro = GyroaveragingOperator(rho_i=1.0)
    solver = GyrokineticPoissonSolver(grid, gyro, tau=1.0)

    g_zero = np.zeros((32, 32), dtype=complex)
    phi = solver.solve(g_zero)

    assert np.all(phi == 0.0)
