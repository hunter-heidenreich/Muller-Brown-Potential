"""File I/O operations for simulation data using HDF5 format."""

import time
from pathlib import Path
import h5py


def _validate_positive_integer(value: int, name: str) -> None:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")


def _ensure_directory_exists(path: Path) -> None:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)


def _save_hdf5_file(data: dict, filepath: Path, observables: list[str] | None = None) -> None:
    """Save simulation data to HDF5 file.
    
    Args:
        data: Simulation data dictionary
        filepath: Path to save the HDF5 file
        observables: List of observables to save. If None, saves all available.
    """
    # Default to all observables if none specified
    if observables is None:
        from src.muller_brown.config import DEFAULT_OBSERVABLES
        observables = DEFAULT_OBSERVABLES
    
    with h5py.File(filepath, "w") as f:
        # Save observables with higher compression level and shuffle filter
        observables_group = f.create_group("observables")
        
        # Only save observables that are both requested and available in data
        available_observables = ["positions", "velocities", "forces", "potential_energy"]
        
        for obs in observables:
            if obs in available_observables and obs in data:
                observables_group.create_dataset(
                    obs, data=data[obs], compression="gzip", compression_opts=9, shuffle=True
                )
        
        # Store which observables were saved as metadata
        observables_group.attrs["saved_observables"] = [obs for obs in observables if obs in data]

        # Save metadata
        metadata = f.create_group("metadata")
        metadata.attrs["dt"] = data["dt"]
        metadata.attrs["n_particles"] = data["n_particles"]
        metadata.attrs["n_steps"] = data["n_steps"]
        metadata.attrs["save_every"] = data["save_every"]
        metadata.attrs["temperature"] = data["temperature"]
        metadata.attrs["friction"] = data["friction"]
        metadata.attrs["mass"] = data["mass"]

        # Add timestamp and version info
        metadata.attrs["timestamp"] = time.time()
        metadata.attrs["format_version"] = "1.0"


def _load_hdf5_file(filepath: Path) -> dict:
    """Load simulation data from HDF5 file."""
    data = {}
    with h5py.File(filepath, "r") as f:
        # Load available observables
        observables_group = f["observables"]
        
        # Load each observable that exists in the file
        for obs_name in observables_group.keys():
            data[obs_name] = observables_group[obs_name][:]
        
        # Load which observables were saved (if available)
        if "saved_observables" in observables_group.attrs:
            saved_obs = observables_group.attrs["saved_observables"]
            # Handle both string arrays and single strings
            if isinstance(saved_obs, (list, tuple)):
                # If it's already a list/tuple, use it directly
                data["saved_observables"] = [str(obs) for obs in saved_obs]
            elif isinstance(saved_obs, (str, bytes)):
                # If it's a single string or bytes, put it in a list
                if isinstance(saved_obs, bytes):
                    data["saved_observables"] = [saved_obs.decode()]
                else:
                    data["saved_observables"] = [saved_obs]
            else:
                # For numpy arrays of strings
                data["saved_observables"] = [str(obs) for obs in saved_obs.flat]

        # Load metadata
        metadata = f["metadata"]
        data["dt"] = metadata.attrs["dt"]
        data["n_particles"] = metadata.attrs["n_particles"]
        data["n_steps"] = metadata.attrs["n_steps"]
        data["save_every"] = metadata.attrs["save_every"]
        data["temperature"] = metadata.attrs["temperature"]
        data["friction"] = metadata.attrs["friction"]
        data["mass"] = metadata.attrs["mass"]

        # Load additional metadata if available
        if "timestamp" in metadata.attrs:
            data["timestamp"] = metadata.attrs["timestamp"]
        if "format_version" in metadata.attrs:
            data["format_version"] = metadata.attrs["format_version"]

    return data


def generate_timestamp_filename(base_name: str, extension: str = "h5") -> str:
    """Generate a timestamped filename for unique file identification."""
    timestamp = int(time.time())
    return f"{base_name}_{timestamp}.{extension}"


def create_artifact_directory(base_dir: str | Path = "artifacts") -> Path:
    """Create a timestamped artifact directory for storing simulation data and plots."""
    from datetime import datetime

    base_path = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_path = base_path / timestamp
    _ensure_directory_exists(artifact_path)
    return artifact_path


def get_artifact_directories(base_dir: str | Path = "artifacts") -> list[Path]:
    """Get list of all artifact directories sorted by creation time."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    artifact_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    return sorted(artifact_dirs, key=lambda x: x.name)


def save_simulation_data(
    data: dict,
    output_dir: str | Path | None = None,
    filename: str | None = None,
    create_artifact_dir: bool = False,
    observables: list[str] | None = None,
) -> Path:
    """Save simulation data to an HDF5 file with automatic directory creation.

    Args:
        data: Simulation data dictionary
        output_dir: Output directory path. If None and create_artifact_dir is True,
                   creates a new timestamped artifact directory
        filename: Output filename. If None, generates timestamped name
        create_artifact_dir: If True, creates a new timestamped artifact directory
        observables: List of observables to save. If None, saves all available.

    Returns:
        Path to saved file
    """
    if create_artifact_dir:
        output_path = create_artifact_directory()
    elif output_dir is None:
        output_path = Path("simulation_data")
    else:
        output_path = Path(output_dir)

    _ensure_directory_exists(output_path)

    if filename is None:
        filename = generate_timestamp_filename("trajectory")

    if not filename.endswith(".h5"):
        filename += ".h5"

    filepath = output_path / filename
    _save_hdf5_file(data, filepath, observables)

    print(f"Simulation data saved to: {filepath}")
    return filepath


def load_simulation_data(filepath: str | Path) -> dict:
    """Load simulation data from an HDF5 file."""
    return _load_hdf5_file(Path(filepath))


def load_artifact_data(artifact_dir: str | Path) -> dict | None:
    """Load simulation data from an artifact directory.

    Searches for .h5 files in the directory and loads the first one found.
    Returns None if no valid data file is found.
    """
    artifact_path = Path(artifact_dir)
    if not artifact_path.exists() or not artifact_path.is_dir():
        raise ValueError(f"Artifact directory does not exist: {artifact_path}")

    # Find .h5 files in the directory
    h5_files = list(artifact_path.glob("*.h5"))
    if not h5_files:
        return None

    # Load the first .h5 file found (you might want to make this smarter)
    return load_simulation_data(h5_files[0])


def batch_save_trajectories(
    trajectories_list: list,
    output_dir: str | Path = "simulation_data",
    base_filename: str = "trajectory",
    observables: list[str] | None = None,
) -> list[Path]:
    """Save multiple trajectory datasets with indexed filenames."""
    output_path = Path(output_dir)
    _ensure_directory_exists(output_path)

    saved_paths = []

    for i, data in enumerate(trajectories_list):
        filename = f"{base_filename}_{i:04d}.h5"
        filepath = output_path / filename
        _save_hdf5_file(data, filepath, observables)
        saved_paths.append(filepath)

    print(f"Saved {len(trajectories_list)} trajectory files to {output_path}")
    return saved_paths
