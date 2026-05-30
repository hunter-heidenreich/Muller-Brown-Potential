#!/usr/bin/env python3
"""Verify the BAOAB Langevin integrator against an analytic harmonic oscillator.

A 2D isotropic harmonic oscillator V(r) = 0.5*k*|r|^2 has a known canonical
distribution, so it is an exact reference for the integrator's sampling:

    position:  variance kB*T/k   per dof
    velocity:  variance kB*T/m   per dof  (kinetic temperature)

BAOAB is exact for configurational sampling of harmonic systems, so the
position variance should match theory to within statistical error at any
timestep; the velocity variance carries a small O(dt^2) bias. We check a
sweep of friction values (under- to over-damped) and plot the distributions.

Run: uv run python verify_langevin.py
"""

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import Tensor

from src.muller_brown import LangevinSimulator, set_random_seed


class HarmonicPotential(nn.Module):
    """Isotropic 2D harmonic oscillator V(r) = 0.5*k*|r|^2."""

    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = k

    def forward(self, coordinates: Tensor) -> Tensor:
        return 0.5 * self.k * (coordinates**2).sum(dim=-1)

    def force(self, coordinates: Tensor) -> Tensor:
        return -self.k * coordinates


def sample_equilibrium(
    k: float,
    temperature: float,
    friction: float,
    mass: float = 1.0,
    dt: float = 0.01,
    n_particles: int = 2000,
    n_steps: int = 20000,
    save_every: int = 10,
    discard_fraction: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a harmonic-oscillator simulation and return equilibrium position and
    velocity samples (transient discarded, flattened over particles and time)."""
    set_random_seed(seed)
    potential = HarmonicPotential(k=k)
    simulator = LangevinSimulator(
        potential=potential,
        temperature=temperature,
        friction=friction,
        mass=mass,
        dt=dt,
    )
    initial_positions = np.zeros((n_particles, 2))
    results = simulator.simulate(initial_positions, n_steps=n_steps, save_every=save_every, progress=False)

    start = int(results["positions"].shape[0] * discard_fraction)
    positions = results["positions"][start:].reshape(-1, 2)
    velocities = results["velocities"][start:].reshape(-1, 2)
    return positions, velocities


def main():
    k = 1.0
    temperature = 1.0  # kB = 1, so this is kB*T
    mass = 1.0
    pos_var_theory = temperature / k
    vel_var_theory = temperature / mass

    frictions = [0.5, 1.0, 5.0]
    tol_pos = 0.03  # position sampling should be near-exact for BAOAB
    tol_vel = 0.05  # velocity carries a small O(dt^2) bias

    print("=== BAOAB Langevin verification: 2D harmonic oscillator ===")
    print(f"k={k}, kB*T={temperature}, m={mass}")
    print(f"Theory: Var[x]={pos_var_theory:.4f}, Var[v]={vel_var_theory:.4f}\n")

    samples = {}
    all_pass = True
    for friction in frictions:
        positions, velocities = sample_equilibrium(
            k=k, temperature=temperature, friction=friction, mass=mass
        )
        samples[friction] = (positions, velocities)

        pos_var = positions.var()
        vel_var = velocities.var()
        pos_err = abs(pos_var - pos_var_theory) / pos_var_theory
        vel_err = abs(vel_var - vel_var_theory) / vel_var_theory
        pos_ok = pos_err < tol_pos
        vel_ok = vel_err < tol_vel
        all_pass = all_pass and pos_ok and vel_ok

        print(f"friction={friction}:")
        print(f"  Var[x] = {pos_var:.4f}  (err {pos_err:6.2%})  {'PASS' if pos_ok else 'FAIL'}")
        print(f"  Var[v] = {vel_var:.4f}  (err {vel_err:6.2%})  {'PASS' if vel_ok else 'FAIL'}")

    _plot(samples, pos_var_theory, vel_var_theory)

    print(f"\nOverall: {'PASS — BAOAB samples the canonical distribution.' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


def _plot(samples: dict, pos_var_theory: float, vel_var_theory: float):
    """Overlay sampled position/velocity histograms with the analytic Gaussians."""
    frictions = list(samples)
    fig, axes = plt.subplots(2, len(frictions), figsize=(4 * len(frictions), 7))

    grids = {
        "x": np.linspace(-4 * pos_var_theory**0.5, 4 * pos_var_theory**0.5, 200),
        "v": np.linspace(-4 * vel_var_theory**0.5, 4 * vel_var_theory**0.5, 200),
    }

    def gaussian(grid, var):
        return np.exp(-(grid**2) / (2 * var)) / np.sqrt(2 * np.pi * var)

    for col, friction in enumerate(frictions):
        positions, velocities = samples[friction]
        for row, (label, data, var, grid) in enumerate(
            [
                ("position", positions[:, 0], pos_var_theory, grids["x"]),
                ("velocity", velocities[:, 0], vel_var_theory, grids["v"]),
            ]
        ):
            ax = axes[row, col]
            ax.hist(data, bins=80, density=True, alpha=0.6, color=f"C{row}")
            ax.plot(grid, gaussian(grid, var), "k--", lw=1.5, label="theory")
            ax.set_title(f"{label}, friction={friction}")
            ax.set_xlabel(label)
            if col == 0:
                ax.set_ylabel("density")
            ax.legend()

    fig.suptitle("BAOAB Langevin sampling vs. analytic harmonic-oscillator distribution")
    fig.tight_layout()
    out = "langevin_verification.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved distribution plot to {out}")


if __name__ == "__main__":
    raise SystemExit(main())
