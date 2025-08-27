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

    displacement_per_particle = np.linalg.norm(positions[-1] - positions[0], axis=1)
    valid_displacements = displacement_per_particle[~np.isnan(displacement_per_particle)]
    
    stats["total_displacement"] = displacement_per_particle
    stats["mean_displacement"] = valid_displacements.mean() if len(valid_displacements) > 0 else np.nan

    if n_steps > 1:
        msd_per_particle = np.sum((positions - positions[0]) ** 2, axis=2)
        valid_msd_final = msd_per_particle[-1][~np.isnan(msd_per_particle[-1])]
        stats["msd_final"] = msd_per_particle[-1]
        stats["msd_mean_final"] = valid_msd_final.mean() if len(valid_msd_final) > 0 else np.nan
        stats["msd_time_series"] = np.nanmean(msd_per_particle, axis=1)

    return stats


def compute_batch_statistics(all_results: list) -> dict:
    """Compute summary statistics across multiple simulation results."""
    if not all_results:
        return {}

    displacements = [r["statistics"]["mean_displacement"] for r in all_results]
    n_steps_list = [r["statistics"]["n_steps"] for r in all_results]
    n_particles_list = [r["statistics"]["n_particles"] for r in all_results]

    return {
        "n_simulations": len(all_results),
        "mean_displacement_stats": {
            "mean": np.mean(displacements),
            "std": np.std(displacements),
            "min": np.min(displacements),
            "max": np.max(displacements),
        },
        "trajectory_length_stats": {
            "mean_n_steps": np.mean(n_steps_list),
            "mean_n_particles": np.mean(n_particles_list),
        },
        "all_displacements": displacements,
    }
