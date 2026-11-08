"""Unit tests for turbulence physical diagnostics, fluxes, and growth rate fitting."""

import numpy as np
import pytest
from vortexgyro.grid import SpectralGrid2D
from vortexgyro.diagnostics import TurbulenceDiagnostics, DiagnosticTracker


def test_electrostatic_energy_positive():
    grid = SpectralGrid2D(nx=32, ny=32, lx=16.0, ly=16.0)
    diag = TurbulenceDiagnostics(grid)

    # Synthetic non-zero potential
    phi_real = np.sin(grid.X) * np.cos(grid.Y)
    phi_hat = grid.fft2(phi_real)

    we = diag.compute_electrostatic_energy(phi_hat)
    assert we > 0.0


def test_zonal_partition_bounds():
    grid = SpectralGrid2D(nx=32, ny=32, lx=16.0, ly=16.0)
    diag = TurbulenceDiagnostics(grid)

    # Pure drift wave (ky != 0) with exact box periodicity
    kx = 2 * np.pi / grid.lx
    ky = 2 * np.pi / grid.ly
    phi_drift_real = np.sin(kx * grid.X) * np.sin(ky * grid.Y)
    phi_drift_hat = grid.fft2(phi_drift_real)

    e_tot, e_zonal, frac = diag.compute_zonal_partition(phi_drift_hat)
    assert e_tot > 0.0
    assert frac < 1e-12  # Exact drift wave mode has 0 zonal component

    # Pure zonal mode (ky = 0)
    phi_zonal_real = np.sin(kx * grid.X)  # Dependent only on x
    phi_zonal_hat = grid.fft2(phi_zonal_real)

    e_tot_z, e_zonal_z, frac_z = diag.compute_zonal_partition(phi_zonal_hat)
    assert e_tot_z > 0.0
    assert frac_z > 0.999999  # Pure zonal mode has 100% zonal component


def test_growth_rate_fitting():
    """Verify linear regression recovers synthetic growth rate gamma."""
    grid = SpectralGrid2D(nx=16, ny=16, lx=8.0, ly=8.0)
    tracker = DiagnosticTracker(grid)

    gamma_true = 0.15
    e0 = 1e-4

    times = np.linspace(0.0, 10.0, 50)
    for step, t in enumerate(times):
        # Synthetic energy W(t) = e0 * exp(2 * gamma * t)
        energy_t = e0 * np.exp(2 * gamma_true * t)
        # Create dummy phi_hat with this exact energy
        # Set a single mode: 0.5 * kperp2 * |phi|^2 * nx * ny ...
        # Or test fit_linear_growth_rate by recording snapshots directly
        from vortexgyro.diagnostics import DiagnosticSnapshot
        snap = DiagnosticSnapshot(
            time=t,
            step=step,
            electrostatic_energy=energy_t,
            zonal_energy=0.0,
            zonal_fraction=0.0,
            turbulent_heat_flux=0.0,
            turbulent_particle_flux=0.0,
            max_shearing_rate=0.0,
            rms_potential=np.sqrt(energy_t),
        )
        tracker.snapshots.append(snap)

    gamma_fit, r2 = tracker.fit_linear_growth_rate(0.0, 10.0)
    assert gamma_fit == pytest.approx(gamma_true, rel=1e-3)
    assert r2 > 0.999
