"""Configuration and experiment setup utilities."""

import numpy as np

from src.muller_brown.constants import (
    MULLER_BROWN_MINIMA,
    DEFAULT_X_RANGE,
    DEFAULT_Y_RANGE,
)
from src.muller_brown.io import _validate_positive_integer

# Default observables to save and plot
DEFAULT_OBSERVABLES = ["positions"]
AVAILABLE_OBSERVABLES = ["positions", "velocities", "forces", "potential_energy"]


def generate_initial_positions(
    n_particles: int,
    method: str = "random",
    x_range: tuple[float, float] = DEFAULT_X_RANGE,
    y_range: tuple[float, float] = DEFAULT_Y_RANGE,
    **kwargs,
) -> np.ndarray:
    """Generate initial positions using 'random', 'grid', or 'near_minimum' methods."""
    _validate_positive_integer(n_particles, "n_particles")

    if method == "random":
        x_coords = np.random.uniform(x_range[0], x_range[1], n_particles)
        y_coords = np.random.uniform(y_range[0], y_range[1], n_particles)
        return np.column_stack([x_coords, y_coords])

    elif method == "grid":
        n_x = int(np.ceil(np.sqrt(n_particles)))
        n_y = int(np.ceil(n_particles / n_x))

        x_coords = np.linspace(x_range[0], x_range[1], n_x)
        y_coords = np.linspace(y_range[0], y_range[1], n_y)

        xx, yy = np.meshgrid(x_coords, y_coords)
        positions = np.column_stack([xx.ravel(), yy.ravel()])

        return positions[:n_particles]

    elif method == "near_minimum":
        minimum_idx = kwargs.get("minimum_idx", 0)
        sigma = kwargs.get("sigma", 0.1)

        if minimum_idx >= len(MULLER_BROWN_MINIMA):
            minimum_idx = 0

        center = np.array(MULLER_BROWN_MINIMA[minimum_idx])
        positions = np.random.normal(center, sigma, (n_particles, 2))

        return positions

    else:
        raise ValueError(f"Unknown method: {method}")


def validate_observables(observables: list[str]) -> list[str]:
    """Validate and return a list of observables to save.
    
    Args:
        observables: List of observable names to validate
        
    Returns:
        Validated list of observables
        
    Raises:
        ValueError: If any observable is not recognized
    """
    if not isinstance(observables, list):
        raise ValueError("observables must be a list")
    
    invalid_observables = [obs for obs in observables if obs not in AVAILABLE_OBSERVABLES]
    if invalid_observables:
        raise ValueError(
            f"Unknown observables: {invalid_observables}. "
            f"Available observables: {AVAILABLE_OBSERVABLES}"
        )
    
    if len(observables) == 0:
        raise ValueError("At least one observable must be specified")
    
    return observables


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
    observables: list[str] | None = None,
    **kwargs,
) -> dict[str, any]:
    """Create a standardized configuration dictionary for simulation experiments."""
    # Validate inputs
    _validate_positive_integer(n_particles, "n_particles")
    _validate_positive_integer(n_steps, "n_steps")
    _validate_positive_integer(save_every, "save_every")

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if friction <= 0:
        raise ValueError(f"friction must be positive, got {friction}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_transient < 0:
        raise ValueError(f"n_transient must be non-negative, got {n_transient}")

    # Set default observables if none specified
    if observables is None:
        observables = DEFAULT_OBSERVABLES.copy()
    
    # Validate observables
    observables = validate_observables(observables)

    return {
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
            "observables": observables,
        },
    }
