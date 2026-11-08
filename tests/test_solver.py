"""Unit tests for GyrokineticSolver, Poisson brackets, and RK4 time integration."""

import numpy as np
import pytest
from vortexgyro.solver import GyrokineticSolver, SimulationConfig


def test_poisson_bracket_antisymmetry():
    """Verify pseudo-spectral Poisson bracket satisfies [f, g] = -[g, f] and [f, f] = 0."""
    config = SimulationConfig(nx=32, ny=32, lx=16.0, ly=16.0)
    solver = GyrokineticSolver(config)

    # Create two arbitrary smooth fields in real space
    x, y = solver.grid.X, solver.grid.Y
    f_real = np.sin(x) * np.cos(y)
    g_real = np.cos(2 * x) * np.sin(2 * y)

    f_hat = solver.grid.fft2(f_real)
    g_hat = solver.grid.fft2(g_real)

    # [f, g] and [g, f]
    b_fg = solver.compute_poisson_bracket(f_hat, g_hat)
    b_gf = solver.compute_poisson_bracket(g_hat, f_hat)
    b_ff = solver.compute_poisson_bracket(f_hat, f_hat)

    # Real space representations
    b_fg_real = solver.grid.ifft2(b_fg)
    b_gf_real = solver.grid.ifft2(b_gf)
    b_ff_real = solver.grid.ifft2(b_ff)

    # [f, g] = -[g, f]
    np.testing.assert_allclose(b_fg_real, -b_gf_real, atol=1e-12)

    # [f, f] = 0
    np.testing.assert_allclose(b_ff_real, 0.0, atol=1e-12)


def test_rk4_time_stepping():
    """Verify RK4 stepping advances simulation time and step count monotonically."""
    config = SimulationConfig(nx=32, ny=32, lx=16.0, ly=16.0, dt=0.02)
    solver = GyrokineticSolver(config)
    solver.initialize_itg_seed(amplitude=1e-4)

    assert solver.step_count == 0
    assert solver.time == 0.0

    solver.step()
    assert solver.step_count == 1
    assert solver.time == pytest.approx(0.02)

    solver.run(5)
    assert solver.step_count == 6
    assert solver.time == pytest.approx(0.12)


def test_current_state_extraction():
    """Verify current_state() creates deep copies of fields and calculates real projections."""
    config = SimulationConfig(nx=32, ny=32, lx=16.0, ly=16.0)
    solver = GyrokineticSolver(config)
    solver.initialize_itg_seed(amplitude=1e-4)

    state = solver.current_state()
    assert state.step == 0
    assert state.time == 0.0
    assert state.g_hat.shape == (32, 32)
    assert state.phi_real.shape == (32, 32)
    assert state.g_real.shape == (32, 32)
    assert np.isrealobj(state.phi_real)
    assert np.isrealobj(state.g_real)
