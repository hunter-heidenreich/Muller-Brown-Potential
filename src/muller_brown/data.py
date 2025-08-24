"""Data conversion and processing utilities."""

import random
import numpy as np
import torch

from src.muller_brown.constants import DEFAULT_DEVICE, DEFAULT_DTYPE


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print(f"Random seed set to: {seed}")


def convert_to_tensor(
    data: np.ndarray | torch.Tensor,
    device: str | torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    """Convert numpy array or tensor to specified device and dtype."""
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device=device, dtype=dtype)
    else:
        return data.to(device=device, dtype=dtype)


def apply_transient_removal(results: dict, n_transient: int) -> dict:
    """Remove initial equilibration steps from all observable data."""
    if n_transient <= 0:
        return results

    save_every = results.get("save_every", 1)
    n_transient_saved = max(1, n_transient // save_every)

    # Check any observable array to get the total length
    positions = results["positions"]

    if n_transient_saved >= positions.shape[0]:
        print(
            f"Warning: n_transient ({n_transient}) corresponds to {n_transient_saved} saved points, "
            f"but data only has {positions.shape[0]} points. Removing {positions.shape[0] - 1} points instead."
        )
        n_transient_saved = positions.shape[0] - 1

    if n_transient_saved == 0:
        print(
            f"Note: n_transient ({n_transient}) is smaller than save_every ({save_every}), no transient removal applied."
        )
        return results

    # Remove transients from all observables
    results["positions"] = positions[n_transient_saved:]
    results["velocities"] = results["velocities"][n_transient_saved:]
    results["forces"] = results["forces"][n_transient_saved:]
    results["potential_energy"] = results["potential_energy"][n_transient_saved:]

    # Update metadata
    original_n_saved = positions.shape[0]
    results["n_saved_original"] = original_n_saved
    results["n_transient_removed"] = n_transient_saved
    results["n_saved_after_transient"] = results["positions"].shape[0]

    print(
        f"Removed {n_transient_saved} transient data points (corresponding to ~{n_transient_saved * save_every} simulation steps)"
    )
    print(
        f"Saved data length: {original_n_saved} -> {results['n_saved_after_transient']} points"
    )

    return results


def combine_batch_trajectories(all_results: list) -> np.ndarray:
    """Combine position trajectories from multiple simulation results into a single array."""
    all_trajectories = []

    for result in all_results:
        positions = result["positions"]  # (n_steps, n_particles, 2)
        trajectory_reshaped = np.transpose(
            positions, (1, 0, 2)
        )  # (n_particles, n_steps, 2)
        # Add each particle's trajectory to the list
        for particle_traj in trajectory_reshaped:
            all_trajectories.append(particle_traj)

    return np.array(all_trajectories)
