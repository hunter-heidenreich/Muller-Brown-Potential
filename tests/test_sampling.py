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

from src.muller_brown import MuellerBrownPotential, LangevinSimulator, set_random_seed


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


@pytest.mark.statistical
def test_harmonic_position_distribution():
    """Configurational sampling of a harmonic well: Var[x]=kB*T/k, <V>=kB*T (2
    dof), and the position marginal is Gaussian (kurtosis 3)."""
    temperature, k = 10.0, 1.0
    set_random_seed(0)
    simulator = LangevinSimulator(HarmonicPotential(k), temperature=temperature, friction=1.0, dt=0.01)
    results = simulator.simulate(
        np.zeros((400, 2)), n_steps=8000, save_every=10,
        observables=["positions", "potential_energy"],
    )
    start = len(results["positions"]) // 4
    positions = results["positions"][start:].reshape(-1, 2)
    energies = results["potential_energy"][start:].reshape(-1)

    assert abs(positions.var() - temperature / k) / (temperature / k) < 0.03
    assert abs(energies.mean() - temperature) / temperature < 0.03
    kurtosis = (positions[:, 0] ** 4).mean() / (positions[:, 0] ** 2).mean() ** 2
    assert abs(kurtosis - 3.0) < 0.15


def _canonical_mean_energy(
    potential, temperature, x_range=(-1.8, 1.3), y_range=(-0.5, 2.2), n=400
) -> float:
    """Boltzmann average <V> = ∫V e^{-V/T} / ∫e^{-V/T}, by grid quadrature."""
    xs = np.linspace(*x_range, n)
    ys = np.linspace(*y_range, n)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = torch.tensor(np.stack([grid_x.ravel(), grid_y.ravel()], axis=1), dtype=torch.float64)
    with torch.no_grad():
        v = potential(points).numpy()
    weight = np.exp(-(v - v.min()) / temperature)
    return float((v * weight).sum() / weight.sum())


@pytest.mark.statistical
def test_muller_brown_samples_boltzmann_distribution():
    """End-to-end configurational check on the real potential: the sampled mean
    energy at equilibrium matches the grid-integrated canonical average. Run at
    T=15 so barrier crossings make the ensemble ergodic."""
    temperature = 15.0
    potential = MuellerBrownPotential(dtype=torch.float64)
    reference = _canonical_mean_energy(potential, temperature)

    set_random_seed(1)
    simulator = LangevinSimulator(potential, temperature=temperature, friction=1.0, dt=0.01)
    x0 = np.random.uniform([-1.5, -0.2], [1.0, 2.0], (300, 2))
    results = simulator.simulate(
        x0, n_steps=40000, save_every=20, observables=["positions", "potential_energy"]
    )
    energies = results["potential_energy"][len(results["potential_energy"]) // 2:].reshape(-1)
    assert abs(energies.mean() - reference) / abs(reference) < 0.02
