"""Command-line interface for VortexGyro gyrokinetic plasma turbulence simulations."""

from __future__ import annotations
import argparse
import sys
import time
import numpy as np

from . import __version__
from .solver import GyrokineticSolver, SimulationConfig
from .diagnostics import DiagnosticTracker


def cmd_run(args: argparse.Namespace) -> int:
    """Executes non-linear ITG turbulence simulation."""
    print("================================================================================")
    print("                    VortexGyro: Gyrokinetic Plasma Simulation                   ")
    print("================================================================================")
    print(f"Grid Resolution : {args.nx} x {args.ny}")
    print(f"Box Dimensions  : Lx = {args.lx} rho_s, Ly = {args.ly} rho_s")
    print(f"Gradients       : R/L_T = {args.kappa_t}, R/L_n = {args.kappa_n}")
    print(f"Time Step dt    : {args.dt}, Total Steps: {args.steps} (T_max = {args.steps * args.dt:.2f})")
    print("--------------------------------------------------------------------------------")

    config = SimulationConfig(
        nx=args.nx,
        ny=args.ny,
        lx=args.lx,
        ly=args.ly,
        dt=args.dt,
        kappa_n=args.kappa_n,
        kappa_T=args.kappa_t,
        hyper_visc=args.hyper_visc,
        use_pade=args.use_pade,
    )
    solver = GyrokineticSolver(config)
    tracker = DiagnosticTracker(solver.grid)

    print("Initializing ITG drift-wave perturbation seed...")
    solver.initialize_itg_seed(amplitude=args.amplitude, target_ky=args.ky)

    snap0 = tracker.record(solver.time, solver.step_count, solver.phi_hat, solver.g_hat)
    print(f"Step 0: Initial Energy = {snap0.electrostatic_energy:.4e}")

    t_start = time.perf_counter()
    report_interval = max(1, args.steps // 10)

    for step in range(1, args.steps + 1):
        solver.step()
        if step % report_interval == 0 or step == args.steps:
            snap = tracker.record(solver.time, solver.step_count, solver.phi_hat, solver.g_hat)
            print(
                f"Step {step:6d} | t={solver.time:6.2f} | "
                f"W_E={snap.electrostatic_energy:.3e} | "
                f"Zonal%={snap.zonal_fraction*100:5.1f}% | "
                f"Q_i={snap.turbulent_heat_flux:+.3e} | "
                f"w_ExB={snap.max_shearing_rate:.3e}"
            )

    elapsed = time.perf_counter() - t_start
    msteps_per_sec = args.steps / elapsed
    print("--------------------------------------------------------------------------------")
    print(f"Execution completed in {elapsed:.3f} s ({msteps_per_sec:.1f} steps/s)")

    gamma, r2 = tracker.fit_linear_growth_rate(0.0, solver.time * 0.5)
    print(f"Fitted ITG Linear Growth Rate: gamma = {gamma:.4f} (R^2 = {r2:.3f})")
    print("================================================================================")
    return 0


def cmd_damping(args: argparse.Namespace) -> int:
    """Executes Rosenbluth-Hinton collisionless zonal flow damping benchmark."""
    print("================================================================================")
    print("        VortexGyro: Rosenbluth-Hinton Zonal Flow Collisionless Damping          ")
    print("================================================================================")
    print(f"Grid: {args.nx} x {args.ny}, kx = 2pi/{args.lx} = {2 * np.pi / args.lx:.4f}")

    config = SimulationConfig(
        nx=args.nx,
        ny=args.ny,
        lx=args.lx,
        ly=args.ly,
        dt=args.dt,
        kappa_n=0.0,
        kappa_T=0.0,  # Pure damping test without gradient drive
        hyper_visc=1e-5,
    )
    solver = GyrokineticSolver(config)
    tracker = DiagnosticTracker(solver.grid)

    # Initialize pure zonal perturbation at kx = 1 mode (ky = 0)
    solver.g_hat[1, 0] = 1.0
    solver.g_hat[-1, 0] = 1.0
    solver.phi_hat = solver.poisson.solve(solver.g_hat)

    initial_phi = float(np.abs(solver.phi_hat[1, 0]))
    print(f"Initial Zonal Potential amplitude: |phi(kx, ky=0)| = {initial_phi:.4e}")

    for step in range(args.steps):
        solver.step()
        if (step + 1) % max(1, args.steps // 5) == 0:
            phi_current = float(np.abs(solver.phi_hat[1, 0]))
            ratio = phi_current / initial_phi
            print(f"Step {step+1:5d} | t={solver.time:5.2f} | |phi|/|phi_0| = {ratio:.4f}")

    final_phi = float(np.abs(solver.phi_hat[1, 0]))
    residual = final_phi / initial_phi
    print("--------------------------------------------------------------------------------")
    print(f"Final Rosenbluth-Hinton Residual Level: {residual:.4f}")
    print("================================================================================")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Runs performance scalability benchmarks across multiple grid resolutions."""
    print("================================================================================")
    print("              VortexGyro: Gyrokinetic Solver Performance Benchmark              ")
    print("================================================================================")
    resolutions = [32, 64, 128]
    n_steps = args.steps

    for n in resolutions:
        config = SimulationConfig(nx=n, ny=n, dt=0.05, hyper_visc=1e-4)
        solver = GyrokineticSolver(config)
        solver.initialize_itg_seed()

        # Warm-up
        solver.step()

        t0 = time.perf_counter()
        for _ in range(n_steps):
            solver.step()
        dt_elapsed = time.perf_counter() - t0

        steps_sec = n_steps / dt_elapsed
        mgrid_pts_sec = (n * n * n_steps) / (dt_elapsed * 1e6)
        print(f"Resolution {n:3d} x {n:3d} ({n*n:5d} pts): {steps_sec:7.1f} steps/s | {mgrid_pts_sec:6.2f} Mpts/s")

    print("================================================================================")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for VortexGyro."""
    parser = argparse.ArgumentParser(
        prog="vortexgyro",
        description="VortexGyro: High-Precision 5D Gyrokinetic Plasma & Tokamak Turbulence Simulator",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    # Command: run
    p_run = subparsers.add_parser("run", help="Run non-linear ITG turbulence simulation")
    p_run.add_argument("--nx", type=int, default=64, help="Grid points in x (radial)")
    p_run.add_argument("--ny", type=int, default=64, help="Grid points in y (binormal)")
    p_run.add_argument("--lx", type=float, default=64.0, help="Radial box size [rho_s]")
    p_run.add_argument("--ly", type=float, default=64.0, help="Binormal box size [rho_s]")
    p_run.add_argument("--dt", type=float, default=0.05, help="Time step size")
    p_run.add_argument("--steps", type=int, default=200, help="Number of steps to integrate")
    p_run.add_argument("--kappa-t", type=float, default=3.0, help="Temperature gradient drive R/L_T")
    p_run.add_argument("--kappa-n", type=float, default=1.0, help="Density gradient drive R/L_n")
    p_run.add_argument("--amplitude", type=float, default=1e-3, help="Initial perturbation amplitude")
    p_run.add_argument("--ky", type=float, default=0.3, help="Target ky mode to seed")
    p_run.add_argument("--hyper-visc", type=float, default=1e-4, help="Hyper-viscosity coefficient")
    p_run.add_argument("--use-pade", action="store_true", help="Use Pade approximation for Bessel")
    p_run.set_defaults(func=cmd_run)

    # Command: damping
    p_damp = subparsers.add_parser("damping", help="Run Rosenbluth-Hinton collisionless damping test")
    p_damp.add_argument("--nx", type=int, default=64, help="Radial resolution")
    p_damp.add_argument("--ny", type=int, default=16, help="Binormal resolution")
    p_damp.add_argument("--lx", type=float, default=32.0, help="Radial box size")
    p_damp.add_argument("--ly", type=float, default=16.0, help="Binormal box size")
    p_damp.add_argument("--dt", type=float, default=0.05, help="Time step size")
    p_damp.add_argument("--steps", type=int, default=100, help="Number of integration steps")
    p_damp.set_defaults(func=cmd_damping)

    # Command: benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run execution benchmarks across resolutions")
    p_bench.add_argument("--steps", type=int, default=100, help="Number of steps per resolution")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
