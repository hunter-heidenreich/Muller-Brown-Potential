"""Langevin dynamics simulation for the Müller-Brown potential."""

import math

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from muller_brown.potential import MuellerBrownPotential
from muller_brown.data import convert_to_tensor
from muller_brown.constants import DEFAULT_DEVICE, DEFAULT_DTYPE


class LangevinSimulator:
    """
    Langevin dynamics simulator for the Müller-Brown potential.

    Implements: m*dv/dt = F(x) - γ*m*v + η(t)
    Integrated with the BAOAB splitting scheme (Leimkuhler & Matthews, 2013),
    which solves the friction + noise part exactly and samples the canonical
    distribution accurately (exact configurational sampling for harmonic systems).
    """

    def __init__(
        self,
        potential: MuellerBrownPotential,
        temperature: float = 15.0,
        friction: float = 1.0,
        mass: float = 1.0,
        dt: float = 0.01,
        device: str | torch.device = DEFAULT_DEVICE,
        dtype: torch.dtype = DEFAULT_DTYPE,
    ):
        """Initialize the Langevin simulator."""
        self.potential = potential
        self.temperature = temperature
        self.friction = friction
        self.mass = mass
        self.dt = dt
        self.device = device
        self.dtype = dtype

        # Pre-compute constants for efficiency
        self.kB = 1.0
        self._half_dt = 0.5 * dt
        self._mass_inv = 1.0 / mass

        # Ornstein-Uhlenbeck (O step) coefficients: the exact solution of the
        # friction + noise sub-dynamics over one timestep. _c1 decays velocity,
        # _c2 is the matching thermal-noise amplitude (fluctuation-dissipation).
        self._c1 = math.exp(-friction * dt)
        self._c2 = math.sqrt(self.kB * temperature / mass * (1.0 - self._c1**2))

    def _baoab_step(
        self, positions: Tensor, velocities: Tensor, forces: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Perform one BAOAB Langevin integration step.

        Splits the step as B-A-O-A-B, integrating the friction/noise (O) part
        exactly via the Ornstein-Uhlenbeck solution. `forces` is the force at
        `positions`; the returned force is the input for the next step, so only
        one force evaluation is needed per step.
        """
        n_particles = positions.shape[0]

        # B: half velocity kick from the current force
        velocities = velocities + self._half_dt * self._mass_inv * forces

        # A: half position drift
        positions = positions + self._half_dt * velocities

        # O: exact friction + thermal noise update
        noise = torch.randn(n_particles, 2, device=self.device, dtype=self.dtype)
        velocities = self._c1 * velocities + self._c2 * noise

        # A: half position drift
        positions = positions + self._half_dt * velocities

        # B: half velocity kick from the force at the new position
        forces = self.potential.force(positions)
        velocities = velocities + self._half_dt * self._mass_inv * forces

        return positions, velocities, forces

    def simulate(
        self,
        initial_positions: Tensor | np.ndarray,
        n_steps: int,
        save_every: int = 1,
        initial_velocities: Tensor | np.ndarray | None = None,
        observables: list[str] | tuple[str, ...] | None = None,
        progress: bool = True,
    ) -> dict:
        """
        Run Langevin dynamics simulation.

        `observables` selects which trajectories to store and return; positions
        are always stored. Pass a subset (e.g. ("positions",)) to avoid
        allocating velocity/force/energy arrays and recomputing energy on long
        runs. Defaults to all four observables. `progress=False` silences the
        startup message and tqdm bar (useful in library/test code). Returns the
        stored trajectories plus run metadata.
        """
        if observables is None:
            observables = ("positions", "velocities", "forces", "potential_energy")
        store_velocities = "velocities" in observables
        store_forces = "forces" in observables
        store_energy = "potential_energy" in observables

        positions = convert_to_tensor(
            initial_positions, device=self.device, dtype=self.dtype
        )
        n_particles = positions.shape[0]

        if initial_velocities is None:
            velocities = torch.zeros_like(positions)
        else:
            velocities = convert_to_tensor(
                initial_velocities, device=self.device, dtype=self.dtype
            )

        # Allocate storage only for the requested observables (positions always)
        n_save_steps = n_steps // save_every + 1
        positions_traj = torch.empty(
            n_save_steps, n_particles, 2, device=self.device, dtype=self.dtype
        )
        velocities_traj = (
            torch.empty(n_save_steps, n_particles, 2, device=self.device, dtype=self.dtype)
            if store_velocities else None
        )
        forces_traj = (
            torch.empty(n_save_steps, n_particles, 2, device=self.device, dtype=self.dtype)
            if store_forces else None
        )
        energies_traj = (
            torch.empty(n_save_steps, n_particles, device=self.device, dtype=self.dtype)
            if store_energy else None
        )

        # Force is always needed to drive the dynamics, even when not stored
        forces = self.potential.force(positions)

        # Store initial state
        positions_traj[0].copy_(positions)
        if store_velocities:
            velocities_traj[0].copy_(velocities)
        if store_forces:
            forces_traj[0].copy_(forces)
        if store_energy:
            energies_traj[0].copy_(self.potential(positions))
        save_idx = 1

        if progress:
            print(f"Starting simulation with {n_particles} particles for {n_steps} steps")

        steps = range(1, n_steps + 1)
        if progress:
            steps = tqdm(steps, desc="Simulation Progress", unit="steps")
        for step in steps:
            positions, velocities, forces = self._baoab_step(
                positions, velocities, forces
            )

            if step % save_every == 0 and save_idx < n_save_steps:
                positions_traj[save_idx].copy_(positions)
                if store_velocities:
                    velocities_traj[save_idx].copy_(velocities)
                if store_forces:
                    forces_traj[save_idx].copy_(forces)
                if store_energy:
                    energies_traj[save_idx].copy_(self.potential(positions))
                save_idx += 1

        results = {
            "positions": positions_traj.detach().cpu().numpy(),
            "dt": self.dt * save_every,
            "n_particles": n_particles,
            "n_steps": n_steps,
            "save_every": save_every,
            "temperature": self.temperature,
            "friction": self.friction,
            "mass": self.mass,
        }
        if store_velocities:
            results["velocities"] = velocities_traj.detach().cpu().numpy()
        if store_forces:
            results["forces"] = forces_traj.detach().cpu().numpy()
        if store_energy:
            results["potential_energy"] = energies_traj.detach().cpu().numpy()
        return results
