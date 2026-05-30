"""Integrator-level tests that need no statistics.

With friction set to zero the BAOAB step degenerates to velocity-Verlet
(c1 = 1, c2 = 0), which is symplectic: total energy must oscillate with bounded
amplitude and show no secular drift. This exercises the B and A half-steps in
isolation from the thermostat, so a failure here is an integrator bug rather
than a sampling one.
"""

import numpy as np
import torch

from src.muller_brown import MuellerBrownPotential, LangevinSimulator


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
