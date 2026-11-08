"""Simulation of Ion Temperature Gradient (ITG) instability growth and turbulent saturation."""

import time
import numpy as np
from vortexgyro.solver import GyrokineticSolver, SimulationConfig
from vortexgyro.diagnostics import DiagnosticTracker


def main():
    print("=== VortexGyro: ITG Instability Growth & Saturation Simulation ===")

    # Simulation setup: 64x64 grid, domain Lx = Ly = 64 rho_s
    config = SimulationConfig(
        nx=64,
        ny=64,
        lx=64.0,
        ly=64.0,
        dt=0.05,
        kappa_n=1.0,     # Density gradient drive
        kappa_T=4.0,     # Strong temperature gradient drive
        tau=1.0,         # Ti / Te = 1
        rho_i=1.0,
        hyper_visc=1e-4,
    )
    solver = GyrokineticSolver(config)
    tracker = DiagnosticTracker(solver.grid)

    print(f"Domain: {config.lx} x {config.ly} rho_s, Resolution: {config.nx} x {config.ny}")
    print(f"Drives: R/L_T = {config.kappa_T}, R/L_n = {config.kappa_n}, dt = {config.dt}")

    # Seed initial perturbation around optimal ITG wavenumber ky ~ 0.3
    solver.initialize_itg_seed(amplitude=1e-4, target_ky=0.3)

    total_steps = 150
    t0 = time.perf_counter()

    print("\nIntegrating Gyrokinetic Vlasov-Poisson system...")
    print("-----------------------------------------------------------------------------------------")
    print(" Step | Time (c_s/R) | Field Energy | Zonal Frac | Radial Heat Flux | ExB Shear Rate")
    print("-----------------------------------------------------------------------------------------")

    for step in range(1, total_steps + 1):
        solver.step()
        if step % 15 == 0 or step == total_steps:
            snap = tracker.record(solver.time, solver.step_count, solver.phi_hat, solver.g_hat)
            print(
                f"{step:5d} | {solver.time:12.2f} | "
                f"{snap.electrostatic_energy:12.4e} | "
                f"{snap.zonal_fraction * 100:9.2f}% | "
                f"{snap.turbulent_heat_flux:+16.4e} | "
                f"{snap.max_shearing_rate:14.4e}"
            )

    elapsed = time.perf_counter() - t0
    print("-----------------------------------------------------------------------------------------")
    print(f"Simulation completed in {elapsed:.2f} seconds ({total_steps / elapsed:.1f} steps/s)")

    # Measure linear growth rate
    gamma, r2 = tracker.fit_linear_growth_rate(1.0, 4.0)
    print(f"\nLinear Stage Analysis (t in [1.0, 4.0]):")
    print(f"  Fitted Linear Growth Rate gamma : {gamma:.4f} c_s/R (R^2 = {r2:.4f})")

    final_snap = tracker.snapshots[-1]
    print(f"\nFinal Turbulence Metrics at t = {solver.time:.2f}:")
    print(f"  Electrostatic Potential RMS    : {final_snap.rms_potential:.4e}")
    print(f"  Zonal Flow Energy Fraction     : {final_snap.zonal_fraction * 100:.2f}%")
    print(f"  Radial Ion Heat Flux (Q_i)     : {final_snap.turbulent_heat_flux:+.4e}")


if __name__ == "__main__":
    main()
