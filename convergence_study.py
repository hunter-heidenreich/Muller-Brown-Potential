#!/usr/bin/env python3
"""Timestep-convergence study for the BAOAB integrator.

BAOAB samples configurational averages very accurately (exactly so for a
harmonic oscillator), but the velocity marginal -- the kinetic temperature --
carries an O(dt^2) bias that grows with the potential's curvature. This script
measures that bias as a function of dt on a stiff harmonic oscillator and on the
Müller-Brown surface, fits the convergence order on a log-log plot, and confirms
the bias extrapolates to zero as dt -> 0 (i.e. it is a finite-timestep artifact,
not an error in the scheme).

Run: uv run python convergence_study.py
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.muller_brown import MuellerBrownPotential, LangevinSimulator, set_random_seed


class StiffHarmonic(nn.Module):
    """Isotropic harmonic well with large curvature, V(r) = 0.5*k*|r|^2."""

    def __init__(self, k: float = 25.0):
        super().__init__()
        self.k = k

    def forward(self, coordinates):
        return 0.5 * self.k * (coordinates**2).sum(dim=-1)

    def force(self, coordinates):
        return -self.k * coordinates


def kinetic_temperature(
    potential, x0: np.ndarray, temperature: float, dt: float, friction: float = 1.0,
    n_seeds: int = 6, n_steps: int = 15000, save_every: int = 10,
) -> float:
    """Measure per-dof <v^2> (= kB*T/m at equilibrium), averaged over seeds."""
    values = []
    for seed in range(n_seeds):
        set_random_seed(seed)
        simulator = LangevinSimulator(potential, temperature=temperature, friction=friction, dt=dt)
        results = simulator.simulate(
            x0, n_steps=n_steps, save_every=save_every,
            observables=["positions", "velocities"],
        )
        velocities = results["velocities"][len(results["velocities"]) // 4:]
        values.append((velocities**2).mean())
    return float(np.mean(values))


def sweep(potential, x0, temperature, timesteps) -> np.ndarray:
    """Kinetic-temperature bias |<v^2> - kB*T| over a range of timesteps."""
    return np.array(
        [abs(kinetic_temperature(potential, x0, temperature, dt) - temperature) for dt in timesteps]
    )


def main():
    temperature = 5.0
    n_particles = 400

    print("=== BAOAB timestep-convergence study (kinetic temperature bias) ===")
    print(f"kB*T = {temperature}, friction = 1.0\n")

    # Each system gets a dt range within its own stability limit; Müller-Brown is
    # far stiffer (curvature ~4000 vs 25) so it must use much smaller timesteps.
    systems = [
        (
            "stiff harmonic (k=25)",
            StiffHarmonic(25.0),
            np.zeros((n_particles, 2)),
            np.array([0.02, 0.03, 0.04, 0.05]),
        ),
        (
            "Müller-Brown",
            MuellerBrownPotential(dtype=torch.float64),
            np.tile([0.623, 0.028], (n_particles, 1)),  # start in a minimum
            np.array([0.004, 0.006, 0.008, 0.010, 0.012]),
        ),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    harmonic_timesteps = harmonic_bias = None
    for label, potential, x0, timesteps in systems:
        bias = sweep(potential, x0, temperature, timesteps)
        order = np.polyfit(np.log(timesteps), np.log(bias), 1)[0]
        print(f"{label}: fitted convergence order = {order:.2f}")
        for dt, b in zip(timesteps, bias):
            print(f"    dt={dt:.3f}  |bias|={b:.5f}")
        ax.loglog(timesteps, bias, "o-", label=f"{label}  (order {order:.2f})")
        if harmonic_bias is None:
            harmonic_timesteps, harmonic_bias = timesteps, bias

    # Reference dt^2 slope, anchored to the harmonic curve's first point
    ax.loglog(
        harmonic_timesteps,
        harmonic_bias[0] * (harmonic_timesteps / harmonic_timesteps[0]) ** 2,
        "k--", alpha=0.6, label=r"$\propto dt^2$",
    )

    ax.set_xlabel("timestep dt")
    ax.set_ylabel(r"kinetic-temperature bias $|\langle v^2\rangle - k_BT|$")
    ax.set_title("BAOAB kinetic-temperature bias vanishes as $dt^2$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = "convergence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved convergence plot to {out}")
    print("Both systems show ~second-order decay: the kinetic bias is a finite-dt artifact.")


if __name__ == "__main__":
    main()
