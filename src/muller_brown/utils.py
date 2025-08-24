"""Utility functions for data handling and I/O operations."""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def save_simulation_data(
    data: Dict[str, Any],
    output_dir: str | Path = "simulation_data",
    filename: Optional[str] = None,
) -> Path:
    """
    Save simulation data to a pickle file.
    
    Args:
        data: Dictionary containing simulation results
        output_dir: Directory to save the data
        filename: Custom filename (optional)
        
    Returns:
        Path to the saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        # Generate filename with timestamp
        import time
        timestamp = int(time.time())
        filename = f"trajectory_{timestamp}.pkl"
    
    if not filename.endswith('.pkl'):
        filename += '.pkl'
    
    filepath = output_path / filename
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
    print(f"Simulation data saved to: {filepath}")
    return filepath


def load_simulation_data(filepath: str | Path) -> Dict[str, Any]:
    """
    Load simulation data from a pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Dictionary containing simulation results
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data


def batch_save_trajectories(
    trajectories_list: list,
    output_dir: str | Path = "simulation_data",
    base_filename: str = "trajectory",
) -> list[Path]:
    """
    Save multiple trajectory datasets with indexed filenames.
    
    Args:
        trajectories_list: List of trajectory data dictionaries
        output_dir: Directory to save the data
        base_filename: Base name for files
        
    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for i, data in enumerate(trajectories_list):
        filename = f"{base_filename}_{i:04d}.pkl"
        filepath = output_path / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
        saved_paths.append(filepath)
    
    print(f"Saved {len(trajectories_list)} trajectory files to {output_path}")
    return saved_paths


def calculate_trajectory_statistics(trajectory: np.ndarray) -> Dict[str, Any]:
    """
    Calculate basic statistics for a trajectory.
    
    Args:
        trajectory: Array of shape (n_steps, n_particles, 2)
        
    Returns:
        Dictionary with trajectory statistics
    """
    n_steps, n_particles, n_dims = trajectory.shape
    
    # Position statistics
    positions = trajectory.reshape(-1, n_dims)  # Flatten time and particles
    
    stats = {
        'n_steps': n_steps,
        'n_particles': n_particles,
        'n_dimensions': n_dims,
        'position_mean': positions.mean(axis=0),
        'position_std': positions.std(axis=0),
        'position_min': positions.min(axis=0),
        'position_max': positions.max(axis=0),
        'total_displacement': np.linalg.norm(trajectory[-1] - trajectory[0], axis=1),
        'mean_displacement': np.linalg.norm(trajectory[-1] - trajectory[0], axis=1).mean(),
    }
    
    # Calculate diffusion properties
    if n_steps > 1:
        # Mean squared displacement for each particle
        msd_per_particle = np.sum((trajectory - trajectory[0]) ** 2, axis=2)  # Sum over dimensions
        stats['msd_final'] = msd_per_particle[-1]  # Final MSD for each particle
        stats['msd_mean_final'] = msd_per_particle[-1].mean()  # Average final MSD
        
        # Time-averaged MSD
        stats['msd_time_series'] = msd_per_particle.mean(axis=1)  # Average over particles
    
    return stats


def generate_initial_positions(
    n_particles: int,
    method: str = "random",
    x_range: tuple[float, float] = (-1.5, 1.2),
    y_range: tuple[float, float] = (-0.2, 2.0),
    **kwargs,
) -> np.ndarray:
    """
    Generate initial positions for particles.
    
    Args:
        n_particles: Number of particles
        method: Method for position generation ("random", "grid", "near_minimum")
        x_range: Range for x coordinates
        y_range: Range for y coordinates
        **kwargs: Additional parameters for specific methods
        
    Returns:
        Array of initial positions (n_particles, 2)
    """
    if method == "random":
        x_coords = np.random.uniform(x_range[0], x_range[1], n_particles)
        y_coords = np.random.uniform(y_range[0], y_range[1], n_particles)
        return np.column_stack([x_coords, y_coords])
    
    elif method == "grid":
        # Create a regular grid of positions
        n_x = int(np.ceil(np.sqrt(n_particles)))
        n_y = int(np.ceil(n_particles / n_x))
        
        x_coords = np.linspace(x_range[0], x_range[1], n_x)
        y_coords = np.linspace(y_range[0], y_range[1], n_y)
        
        xx, yy = np.meshgrid(x_coords, y_coords)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        
        return positions[:n_particles]  # Take only the required number
    
    elif method == "near_minimum":
        # Start particles near a specific minimum
        minimum_idx = kwargs.get("minimum_idx", 0)
        sigma = kwargs.get("sigma", 0.1)
        
        # Müller-Brown minima (hardcoded for convenience)
        minima = [(-0.558, 1.442), (0.623, 0.028), (-0.050, 0.467)]
        
        if minimum_idx >= len(minima):
            minimum_idx = 0
        
        center = np.array(minima[minimum_idx])
        
        # Generate Gaussian distributed positions around the minimum
        positions = np.random.normal(center, sigma, (n_particles, 2))
        
        return positions
    
    else:
        raise ValueError(f"Unknown method: {method}")


def create_experiment_config(
    n_particles: int = 1,
    n_steps: int = 100000,
    temperature: float = 15.0,
    friction: float = 1.0,
    dt: float = 0.01,
    save_every: int = 100,
    n_transient: int = 0,
    seed: int = 42,
    initial_position_method: str = "random",
    **kwargs,
) -> Dict[str, Any]:
    """
    Create a configuration dictionary for simulation experiments.
    
    Args:
        n_particles: Number of particles
        n_steps: Number of simulation steps
        temperature: Temperature for thermostat
        friction: Friction coefficient
        dt: Time step
        save_every: Save frequency
        n_transient: Number of initial steps to discard as transient
        seed: Random seed for reproducibility
        initial_position_method: Method for generating initial positions
        **kwargs: Additional configuration parameters
        
    Returns:
        Configuration dictionary
    """
    config = {
        "simulation": {
            "n_particles": n_particles,
            "n_steps": n_steps,
            "temperature": temperature,
            "friction": friction,
            "dt": dt,
            "save_every": save_every,
            "n_transient": n_transient,
            "seed": seed,
        },
        "initial_conditions": {
            "method": initial_position_method,
            **kwargs,
        },
        "output": {
            "save_data": True,
            "save_plots": True,
        }
    }
    
    return config
