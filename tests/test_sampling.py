"""Statistical sampling tests: does the thermostat reproduce the canonical
distribution? Seeded, with tolerances chosen well above the sampling error so
the assertions are stable.

Equipartition is checked on a free particle and a *soft* harmonic oscillator
rather than on Müller-Brown: BAOAB is accurate for configurational sampling but
the velocity marginal (kinetic temperature) carries an O(dt^2) bias that grows
with the potential's curvature. On the stiff Müller-Brown surface that bias is
~3% at dt=0.01, which would swamp the statistical signal; on soft systems it is
negligible.
"""

import numpy as np
import torch
import torch.nn as nn
import pytest

from src.muller_brown import LangevinSimulator, set_random_seed


class HarmonicPotential(nn.Module):
    """Isotropic 2D harmonic oscillator V(r) = 0.5 * k * |r|^2."""

    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = k

    def forward(self, coordinates):
        return 0.5 * self.k * (coordinates**2).sum(dim=-1)

    def force(self, coordinates):
        return -self.k * coordinates


class FreeParticle(nn.Module):
    """V(r) = 0; the velocity marginal is then the exact Ornstein-Uhlenbeck one."""

    def forward(self, coordinates):
        return torch.zeros(coordinates.shape[:-1], dtype=coordinates.dtype)

    def force(self, coordinates):
        return torch.zeros_like(coordinates)


def _kinetic_temperature(
    potential, temperature: float, seed: int, n_particles: int = 400,
    n_steps: int = 5000, save_every: int = 10,
) -> float:
    """Per-dof <v^2>, which equals kB*T/m at equilibrium (kB = m = 1 here)."""
    set_random_seed(seed)
    simulator = LangevinSimulator(potential, temperature=temperature, friction=1.0, dt=0.01)
    results = simulator.simulate(
        np.zeros((n_particles, 2)),
        n_steps=n_steps,
        save_every=save_every,
        observables=["positions", "velocities"],
    )
    velocities = results["velocities"][len(results["velocities"]) // 4:]  # drop transient
    return float((velocities**2).mean())


@pytest.mark.statistical
@pytest.mark.parametrize("temperature", [5.0, 15.0, 30.0])
def test_equipartition_free_particle(temperature):
    t_kin = _kinetic_temperature(FreeParticle(), temperature, seed=0)
    assert abs(t_kin - temperature) / temperature < 0.04


@pytest.mark.statistical
@pytest.mark.parametrize("temperature", [5.0, 15.0, 30.0])
def test_equipartition_soft_harmonic(temperature):
    t_kin = _kinetic_temperature(HarmonicPotential(k=1.0), temperature, seed=0)
    assert abs(t_kin - temperature) / temperature < 0.04
