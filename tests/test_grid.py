"""Unit tests for SpectralGrid2D coordinate and Fourier spectral differentiation operations."""

import numpy as np
import pytest
from vortexgyro.grid import SpectralGrid2D


def test_grid_initialization():
    grid = SpectralGrid2D(nx=32, ny=32, lx=16.0, ly=16.0)
    assert grid.nx == 32
    assert grid.ny == 32
    assert grid.lx == 16.0
    assert grid.ly == 16.0
    assert grid.dx == pytest.approx(0.5)
    assert grid.dy == pytest.approx(0.5)


def test_fourier_differentiation():
    """Verify that spectral differentiation of sin(kx * x) matches analytical cosine derivative."""
    nx, ny = 32, 32
    lx, ly = 2 * np.pi, 2 * np.pi
    grid = SpectralGrid2D(nx, ny, lx, ly)

    # Function f(x, y) = sin(2x) * cos(3y)
    f_real = np.sin(2.0 * grid.X) * np.cos(3.0 * grid.Y)
    f_hat = grid.fft2(f_real)

    # df/dx = 2 * cos(2x) * cos(3y)
    df_dx_exact = 2.0 * np.cos(2.0 * grid.X) * np.cos(3.0 * grid.Y)
    df_dx_spectral = grid.ifft2(grid.diff_x(f_hat))

    # df/dy = -3 * sin(2x) * sin(3y)
    df_dy_exact = -3.0 * np.sin(2.0 * grid.X) * np.sin(3.0 * grid.Y)
    df_dy_spectral = grid.ifft2(grid.diff_y(f_hat))

    np.testing.assert_allclose(df_dx_spectral, df_dx_exact, atol=1e-12)
    np.testing.assert_allclose(df_dy_spectral, df_dy_exact, atol=1e-12)


def test_dealiasing_mask():
    """Verify 2/3 Orszag rule zeroing highest wavenumbers."""
    grid = SpectralGrid2D(nx=64, ny=64, lx=32.0, ly=32.0)
    mask = grid.dealias_mask

    # Outer corners must be zeroed out
    assert mask[0, 0] == 1.0  # DC component is preserved
    assert mask[32, 32] == 0.0  # Nyquist corner is zeroed
    assert np.all(mask >= 0.0)
    assert np.all(mask <= 1.0)


def test_laplacian():
    """Verify spectral Laplacian operator kperp2."""
    grid = SpectralGrid2D(nx=32, ny=32, lx=2 * np.pi, ly=2 * np.pi)
    # f(x, y) = sin(x) + cos(y)
    f_real = np.sin(grid.X) + np.cos(grid.Y)
    f_hat = grid.fft2(f_real)

    # Laplacian: d^2 f/dx^2 + d^2 f/dy^2 = -sin(x) - cos(y) = -f
    lap_f_hat = grid.laplacian(f_hat)
    lap_f_real = grid.ifft2(lap_f_hat)

    np.testing.assert_allclose(lap_f_real, -f_real, atol=1e-12)
