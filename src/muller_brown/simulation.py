"""Langevin dynamics simulation for the Müller-Brown potential."""

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from src.muller_brown.potential import MuellerBrownPotential
from src.muller_brown.data import convert_to_tensor
from src.muller_brown.constants import DEFAULT_DEVICE, DEFAULT_DTYPE


class LangevinSimulator:
    """
    Langevin dynamics simulator for the Müller-Brown potential.

    Implements: m*dv/dt = F(x) - γ*m*v + η(t)
    Uses velocity-Verlet integration with proper Langevin thermostat.
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
        noise_coeff = dt * self.kB * temperature * friction / mass
        self._noise_std = torch.sqrt(
            torch.tensor(noise_coeff, device=device, dtype=dtype)
        )
        self._half_dt = 0.5 * dt
        self._mass_inv = 1.0 / mass

    def _compute_noise_term(self, n_particles: int) -> Tensor:
        """Generate Gaussian white noise for Langevin thermostat."""
        return self._noise_std * torch.randn(
            n_particles, 2, device=self.device, dtype=self.dtype
        )

    def _velocity_verlet_step(
        self, positions: Tensor, velocities: Tensor, forces: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Perform one velocity-Verlet integration step with Langevin thermostat."""
        n_particles = positions.shape[0]

        # Generate noise for both half-steps
        noise1 = self._compute_noise_term(n_particles) * 0.5
        noise2 = self._compute_noise_term(n_particles) * 0.5

        # Velocity update (first half)
        vel_half = (
            velocities
            + self._half_dt * (forces * self._mass_inv - self.friction * velocities)
            + noise1
        )

        # Position update
        new_positions = positions + vel_half * self.dt

        # Force update
        new_forces = self.potential.force(new_positions)

        # Velocity update (second half)
        new_velocities = (
            vel_half
            + self._half_dt * (new_forces * self._mass_inv - self.friction * vel_half)
            + noise2
        )

        return new_positions, new_velocities, new_forces

    def simulate(
        self,
        initial_positions: Tensor | np.ndarray,
        n_steps: int,
        save_every: int = 1,
        initial_velocities: Tensor | np.ndarray | None = None,
    ) -> dict:
        """
        Run Langevin dynamics simulation.

        Returns dict with full observables: positions, velocities, forces, potential energy.
        """
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

        # Initialize storage for all observables
        n_save_steps = n_steps // save_every + 1
        positions_traj = torch.empty(
            n_save_steps, n_particles, 2, device=self.device, dtype=self.dtype
        )
        velocities_traj = torch.empty(
            n_save_steps, n_particles, 2, device=self.device, dtype=self.dtype
        )
        forces_traj = torch.empty(
            n_save_steps, n_particles, 2, device=self.device, dtype=self.dtype
        )
        energies_traj = torch.empty(
            n_save_steps, n_particles, device=self.device, dtype=self.dtype
        )

        forces = self.potential.force(positions)
        energies = self.potential(positions)

        # Store initial state
        positions_traj[0].copy_(positions)
        velocities_traj[0].copy_(velocities)
        forces_traj[0].copy_(forces)
        energies_traj[0].copy_(energies)
        save_idx = 1

        print(f"Starting simulation with {n_particles} particles for {n_steps} steps")

        for step in tqdm(
            range(1, n_steps + 1), desc="Simulation Progress", unit="steps"
        ):
            positions, velocities, forces = self._velocity_verlet_step(
                positions, velocities, forces
            )

            if step % save_every == 0 and save_idx < n_save_steps:
                energies = self.potential(positions)
                positions_traj[save_idx].copy_(positions)
                velocities_traj[save_idx].copy_(velocities)
                forces_traj[save_idx].copy_(forces)
                energies_traj[save_idx].copy_(energies)
                save_idx += 1

        return {
            "positions": positions_traj.detach().cpu().numpy(),
            "velocities": velocities_traj.detach().cpu().numpy(),
            "forces": forces_traj.detach().cpu().numpy(),
            "potential_energy": energies_traj.detach().cpu().numpy(),
            "dt": self.dt * save_every,
            "n_particles": n_particles,
            "n_steps": n_steps,
            "save_every": save_every,
            "temperature": self.temperature,
            "friction": self.friction,
            "mass": self.mass,
        }
