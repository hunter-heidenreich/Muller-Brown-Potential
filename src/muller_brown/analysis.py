"""Trajectory analysis and statistical computations."""

import numpy as np


def calculate_trajectory_statistics(data: dict) -> dict:
    """Calculate comprehensive statistical properties from simulation data."""
    if "positions" not in data:
        raise ValueError("Positions data is required for trajectory statistics")
    
    positions = data["positions"]
    n_steps, n_particles, n_dims = positions.shape

    if n_dims != 2:
        raise ValueError(f"Expected 2D trajectories, got {n_dims} dimensions")
    if n_steps < 1:
        raise ValueError(f"Trajectory must have at least 1 time step, got {n_steps}")
    if n_particles < 1:
        raise ValueError(f"Trajectory must have at least 1 particle, got {n_particles}")

    stats = {
        "n_steps": n_steps,
        "n_particles": n_particles,
        "n_dimensions": n_dims,
    }

    positions_flat = positions.reshape(-1, n_dims)
    stats.update({
        "position_mean": positions_flat.mean(axis=0),
        "position_std": positions_flat.std(axis=0),
        "position_min": positions_flat.min(axis=0),
        "position_max": positions_flat.max(axis=0),
    })

    if "velocities" in data:
        velocities = data["velocities"]
        velocities_flat = velocities.reshape(-1, n_dims)
        stats.update({
            "velocity_mean": velocities_flat.mean(axis=0),
            "velocity_std": velocities_flat.std(axis=0),
            "velocity_magnitude_mean": np.linalg.norm(velocities_flat, axis=1).mean(),
        })

    if "forces" in data:
        forces = data["forces"]
        forces_flat = forces.reshape(-1, n_dims)
        stats.update({
            "force_mean": forces_flat.mean(axis=0),
            "force_std": forces_flat.std(axis=0),
            "force_magnitude_mean": np.linalg.norm(forces_flat, axis=1).mean(),
        })

    if "potential_energy" in data:
        energies = data["potential_energy"]
        energies_flat = energies.reshape(-1)
        stats.update({
            "energy_mean": energies_flat.mean(),
            "energy_std": energies_flat.std(),
            "energy_min": energies_flat.min(),
            "energy_max": energies_flat.max(),
        })

    return stats

