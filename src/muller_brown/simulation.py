"""Langevin dynamics simulation for the Müller-Brown potential."""

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from .potential import MuellerBrownPotential


class LangevinSimulator:
    """
    Langevin dynamics simulator for molecular dynamics on potential energy surfaces.
    
    Uses the velocity-Verlet integration scheme with Langevin thermostat.
    """
    
    def __init__(
        self,
        potential: MuellerBrownPotential,
        temperature: float = 15.0,
        friction: float = 1.0,
        mass: float = 1.0,
        dt: float = 0.01,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        """
        Initialize the Langevin simulator.
        
        Args:
            potential: Müller-Brown potential instance
            temperature: Temperature for the thermostat
            friction: Friction coefficient (gamma)
            mass: Particle mass
            dt: Time step
            device: Torch device
            dtype: Torch data type
        """
        self.potential = potential
        self.temperature = temperature
        self.friction = friction
        self.mass = mass
        self.dt = dt
        self.device = device
        self.dtype = dtype
        
        # Constants
        self.kB = 1.0  # Boltzmann constant
        
        # Pre-compute noise variance coefficient (matching original)
        # Original: dt * kB * Temp * gamma * (1/masses)
        self.noise_coeff = dt * self.kB * temperature * friction / mass
        
    def _compute_noise_term(self, n_particles: int) -> Tensor:
        """Compute the noise term for Langevin dynamics matching original."""
        noise_std = torch.sqrt(torch.tensor(self.noise_coeff, device=self.device, dtype=self.dtype))
        noise = torch.randn(n_particles, 2, device=self.device, dtype=self.dtype)
        return noise_std * noise
    
    def _compute_forces_efficient(self, positions: Tensor) -> Tensor:
        """Compute forces for multiple particles efficiently."""
        # Vectorized force computation - potential.force handles batched inputs
        return self.potential.force(positions)
    
    def _velocity_verlet_step(
        self, 
        positions: Tensor, 
        velocities: Tensor, 
        forces: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Perform one velocity-Verlet integration step with Langevin thermostat.
        Fixed to use correct Langevin dynamics formulation.
        
        Args:
            positions: Current positions (n_particles, 2)
            velocities: Current velocities (n_particles, 2)
            forces: Current forces (n_particles, 2)
            
        Returns:
            Updated positions, velocities, and forces
        """
        n_particles = positions.shape[0]
        
        # Compute noise terms
        noise1 = self._compute_noise_term(n_particles)
        noise2 = self._compute_noise_term(n_particles)
        
        # Update velocities (half step) - correct Langevin formulation
        vel_half = velocities + 0.5 * self.dt * (
            forces / self.mass - self.friction * velocities
        ) + 0.5 * noise1
        
        # Update positions
        new_positions = positions + vel_half * self.dt
        
        # Compute new forces
        new_forces = self._compute_forces_efficient(new_positions)
        
        # Update velocities (second half step)
        new_velocities = vel_half + 0.5 * self.dt * (
            new_forces / self.mass - self.friction * vel_half
        ) + 0.5 * noise2
        
        return new_positions, new_velocities, new_forces
    
    def simulate(
        self,
        initial_positions: Tensor | np.ndarray,
        n_steps: int,
        save_every: int = 1,
        initial_velocities: Optional[Tensor | np.ndarray] = None,
    ) -> dict:
        """
        Run Langevin dynamics simulation.
        
        Args:
            initial_positions: Starting positions (n_particles, 2)
            n_steps: Number of simulation steps
            save_every: Save trajectory every N steps
            initial_velocities: Initial velocities (optional)
            
        Returns:
            Dictionary containing trajectory data
        """
        # Convert inputs to tensors
        if isinstance(initial_positions, np.ndarray):
            initial_positions = torch.from_numpy(initial_positions)
        positions = initial_positions.to(device=self.device, dtype=self.dtype)
        
        n_particles = positions.shape[0]
        
        if initial_velocities is None:
            velocities = torch.zeros_like(positions)
        else:
            if isinstance(initial_velocities, np.ndarray):
                initial_velocities = torch.from_numpy(initial_velocities)
            velocities = initial_velocities.to(device=self.device, dtype=self.dtype)
        
        # Initialize trajectory storage
        n_save_steps = n_steps // save_every + 1  # +1 for initial state
        trajectory = torch.zeros(n_save_steps, n_particles, 2, device=self.device, dtype=self.dtype)
        
        # Compute initial forces efficiently
        forces = self._compute_forces_efficient(positions)
        
        # Save initial positions
        trajectory[0] = positions.clone()
        save_idx = 1
        
        print(f"Starting simulation with {n_particles} particles for {n_steps} steps")
        
        # Main simulation loop with tqdm progress bar
        for step in tqdm(range(1, n_steps + 1), desc="Simulation Progress", unit="steps"):
            positions, velocities, forces = self._velocity_verlet_step(
                positions, velocities, forces
            )
            
            # Save trajectory - only when needed
            if step % save_every == 0 and save_idx < n_save_steps:
                trajectory[save_idx] = positions.clone()
                save_idx += 1
        
        # Convert to numpy for easier handling
        trajectory_np = trajectory.detach().cpu().numpy()
        
        return {
            "trajectory": trajectory_np,
            "dt": self.dt * save_every,
            "n_particles": n_particles,
            "n_steps": n_steps,
            "save_every": save_every,
            "temperature": self.temperature,
            "friction": self.friction,
            "mass": self.mass,
        }
    
    def generate_random_initial_positions(
        self, 
        n_particles: int, 
        x_range: Tuple[float, float] = (-1.5, 1.2),
        y_range: Tuple[float, float] = (-0.2, 2.0),
    ) -> Tensor:
        """Generate random initial positions within specified ranges."""
        x_coords = torch.rand(n_particles, device=self.device, dtype=self.dtype)
        x_coords = x_coords * (x_range[1] - x_range[0]) + x_range[0]
        
        y_coords = torch.rand(n_particles, device=self.device, dtype=self.dtype)
        y_coords = y_coords * (y_range[1] - y_range[0]) + y_range[0]
        
        return torch.stack([x_coords, y_coords], dim=1)
