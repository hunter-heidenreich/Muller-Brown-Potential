"""Integrator-level tests that need no statistics.

With friction set to zero the BAOAB step degenerates to velocity-Verlet
(c1 = 1, c2 = 0), which is symplectic: total energy must oscillate with bounded
amplitude and show no secular drift. This exercises the B and A half-steps in
isolation from the thermostat, so a failure here is an integrator bug rather
than a sampling one.
"""

import numpy as np
import torch

from src.muller_brown import MuellerBrownPotential, LangevinSimulator, set_random_seed


def _run(seed: int) -> dict:
    """Seed, then run a short stochastic trajectory from a fixed start."""
    set_random_seed(seed)
    potential = MuellerBrownPotential(dtype=torch.float64)
    simulator = LangevinSimulator(potential, temperature=15.0, friction=1.0, dt=0.01)
    return simulator.simulate(
        np.array([[0.0, 0.0]]),
        n_steps=2000,
        save_every=10,
        observables=["positions", "velocities"],
        progress=False,
    )


def test_energy_conserved_without_friction():
    potential = MuellerBrownPotential(dtype=torch.float64)
    mass = 1.0
    dt = 1e-3
    simulator = LangevinSimulator(
        potential, temperature=1.0, friction=0.0, mass=mass, dt=dt
    )

    # Start at a minimum with a modest kick so PE <-> KE exchange is non-trivial
    x0 = np.array([[0.623, 0.028]])
    v0 = np.array([[0.5, 0.3]])
    n_steps, save_every = 20000, 10
    results = simulator.simulate(
        x0,
        n_steps=n_steps,
        save_every=save_every,
        initial_velocities=v0,
        observables=["positions", "velocities", "potential_energy"],
        progress=False,
    )

    pe = results["potential_energy"][:, 0]
    velocities = results["velocities"][:, 0, :]
    ke = 0.5 * mass * (velocities**2).sum(axis=1)
    energy = ke + pe

    # Bounded oscillation, measured against the energy-exchange (kinetic) scale
    rel_fluctuation = energy.std() / ke.mean()
    assert rel_fluctuation < 1e-3, f"energy fluctuation too large: {rel_fluctuation:.2e}"

    # No secular drift: the linear trend over the run is far below the oscillation
    time = np.arange(len(energy)) * dt * save_every
    slope = np.polyfit(time, energy, 1)[0]
    total_drift = abs(slope * time[-1])
    assert total_drift < 0.1 * energy.std(), (
        f"energy drifts ({total_drift:.2e}) relative to its oscillation ({energy.std():.2e})"
    )


def test_same_seed_reproduces_trajectory():
    """Reseeding before a run reproduces it bitwise (the O-step draws from the
    global torch RNG, so determinism depends on the seed being reset)."""
    first = _run(seed=123)
    second = _run(seed=123)
    assert np.array_equal(first["positions"], second["positions"])
    assert np.array_equal(first["velocities"], second["velocities"])


def test_different_seed_changes_trajectory():
    assert not np.array_equal(_run(seed=1)["positions"], _run(seed=2)["positions"])
