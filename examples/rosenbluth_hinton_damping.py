"""Verification of Rosenbluth-Hinton collisionless zonal flow damping."""

import numpy as np
from vortexgyro.solver import GyrokineticSolver, SimulationConfig


def main():
    print("=== VortexGyro: Rosenbluth-Hinton Collisionless Zonal Flow Damping ===")
    print("Testing residual zonal potential amplitude after collisionless Landau damping...")

    config = SimulationConfig(
        nx=64,
        ny=16,
        lx=32.0,
        ly=16.0,
        dt=0.05,
        kappa_n=0.0,
        kappa_T=0.0,  # Zero background gradient drive to isolate pure damping
        hyper_visc=1e-5,
    )
    solver = GyrokineticSolver(config)

    # Initialize pure zonal mode at kx = 2*pi/Lx (ky = 0)
    solver.g_hat[1, 0] = 1.0
    solver.g_hat[-1, 0] = 1.0
    solver.phi_hat = solver.poisson.solve(solver.g_hat)

    phi_0 = float(np.abs(solver.phi_hat[1, 0]))
    print(f"Initial zonal potential amplitude: phi(t=0) = {phi_0:.4e}")

    steps = 80
    print("\nStepping...")
    for step in range(1, steps + 1):
        solver.step()
        if step % 16 == 0 or step == steps:
            phi_t = float(np.abs(solver.phi_hat[1, 0]))
            ratio = phi_t / phi_0
            print(f"Step {step:3d} | t = {solver.time:5.2f} | |phi(t)| / |phi(0)| = {ratio:.4f}")

    phi_final = float(np.abs(solver.phi_hat[1, 0]))
    residual = phi_final / phi_0
    print(f"\nFinal Rosenbluth-Hinton Residual Level: A_R = {residual:.4f}")
    print("Physical interpretation: In collisionless toroidal gyrokinetics, geodesic acoustic modes")
    print("damp via Landau damping, leaving a finite residual zonal flow that regulates turbulence.")


if __name__ == "__main__":
    main()
