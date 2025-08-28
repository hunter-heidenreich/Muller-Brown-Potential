#!/usr/bin/env python3
"""
Modern implementation of Müller-Brown potential simulation.

This script demonstrates the usage of the modular Müller-Brown simulation package
with clean separation of concerns and modern PyTorch practices.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.muller_brown import (
    MuellerBrownPotential,
    LangevinSimulator,
    MuellerBrownVisualizer,
    save_simulation_data,
    load_artifact_data,
    get_artifact_directories,
    generate_initial_positions,
    create_experiment_config,
    set_random_seed,
    apply_transient_removal,
)


def run_single_simulation(config: dict) -> dict:
    """Run a single simulation with the given configuration."""
    # Set random seed using utility function
    seed = config["simulation"]["seed"]
    set_random_seed(seed)

    print("Initializing Müller-Brown potential...")
    device = config["simulation"].get("device", "cpu")
    potential = MuellerBrownPotential(device=device, dtype=torch.float64)

    print("Setting up Langevin simulator...")
    simulator = LangevinSimulator(
        potential=potential,
        temperature=config["simulation"]["temperature"],
        friction=config["simulation"]["friction"],
        dt=config["simulation"]["dt"],
        device=device,
        dtype=torch.float64,
    )

    print("Generating initial positions...")
    initial_positions = generate_initial_positions(
        n_particles=config["simulation"]["n_particles"], **config["initial_conditions"]
    )

    print(f"Starting positions:\n{initial_positions}")

    print("Running simulation...")
    raw_results = simulator.simulate(
        initial_positions=initial_positions,
        n_steps=config["simulation"]["n_steps"],
        save_every=config["simulation"]["save_every"],
    )

    # Apply transient removal if specified using utility function
    n_transient = config["simulation"].get("n_transient", 0)
    if n_transient > 0:
        raw_results = apply_transient_removal(raw_results, n_transient)

    # Filter results to only include requested observables
    requested_observables = config["output"]["observables"]
    results = {}
    
    # Always include metadata
    metadata_keys = ["dt", "n_particles", "n_steps", "save_every", "temperature", "friction", "mass"]
    for key in metadata_keys:
        if key in raw_results:
            results[key] = raw_results[key]
    
    # Include only requested observables
    for obs in requested_observables:
        if obs in raw_results:
            results[obs] = raw_results[obs]

    results["config"] = config

    return results


def run_batch_simulation(n_simulations: int = 10, observables: list[str] | None = None, **sim_kwargs) -> tuple[list, Path]:
    """Run multiple simulations with random initial conditions.

    Returns:
        Tuple of (all_results, batch_artifact_directory)
    """
    print(f"Running {n_simulations} simulations...")

    all_results = []
    base_seed = sim_kwargs["seed"]  # Now always has a value

    # Create a main artifact directory for this batch
    from src.muller_brown import create_artifact_directory

    batch_artifact_dir = create_artifact_directory()
    print(f"Saving batch results to: {batch_artifact_dir}")

    for i in range(n_simulations):
        print(f"\n--- Simulation {i + 1}/{n_simulations} ---")

        # Create configuration for this simulation
        # Use incremented seeds for each simulation
        sim_kwargs_copy = sim_kwargs.copy()
        sim_kwargs_copy["seed"] = base_seed + i

        config = create_experiment_config(observables=observables, **sim_kwargs_copy)

        # Run simulation
        results = run_single_simulation(config)
        all_results.append(results)

        # Save individual result in the batch artifact directory
        if config["output"]["save_data"]:
            observables_to_save = config["output"]["observables"]
            save_simulation_data(
                results,
                output_dir=batch_artifact_dir,
                filename=f"simulation_{i:03d}.h5",
                observables=observables_to_save,
            )

    return all_results, batch_artifact_dir


def _setup_visualizer(config: dict) -> MuellerBrownVisualizer:
    """Create a MuellerBrownVisualizer from simulation configuration."""
    device = config.get("simulation", {}).get("device", "cpu")
    potential = MuellerBrownPotential(device=device, dtype=torch.float64)
    return MuellerBrownVisualizer(potential)


def _save_plot(fig, output_dir: Path, filename: str, dpi: int = 300):
    """Save a matplotlib figure with consistent settings."""
    fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def create_visualizations(results: dict, output_dir: str | Path = "plots", save_animation: bool = False):
    """Create and save all visualizations for simulation results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    visualizer = _setup_visualizer(results["config"])
    
    # Get which observables are available
    available_observables = [obs for obs in ["positions", "velocities", "forces", "potential_energy"] if obs in results]
    print(f"Creating visualizations for available observables: {available_observables}")

    print("Creating potential surface plot...")
    fig, ax = visualizer.plot_potential_surface()
    _save_plot(fig, output_path, "potential_surface.png")

    print("Creating time-independent distribution plots...")

    # Position distributions - aggregate across all particles
    if "positions" in results:
        try:
            fig, axes = visualizer.plot_position_distributions(results)
            _save_plot(fig, output_path, "position_distributions.png")
        except ValueError as e:
            print(f"Skipping position distributions: {e}")

    # Velocity distributions - aggregate across all particles
    if "velocities" in results:
        try:
            fig, axes = visualizer.plot_velocity_distributions(results)
            _save_plot(fig, output_path, "velocity_distributions.png")
        except ValueError as e:
            print(f"Skipping velocity distributions: {e}")

    # Force distributions - aggregate across all particles
    if "forces" in results:
        try:
            fig, axes = visualizer.plot_force_distributions(results)
            _save_plot(fig, output_path, "force_distributions.png")
        except ValueError as e:
            print(f"Skipping force distributions: {e}")

    # Energy distribution - aggregate across all particles
    if "potential_energy" in results:
        try:
            fig, ax = visualizer.plot_energy_distribution(results)
            _save_plot(fig, output_path, "energy_distribution.png")
        except ValueError as e:
            print(f"Skipping energy distribution: {e}")

    print("Creating time-dependent plots...")

    # Trajectory on potential surface
    if "positions" in results:
        try:
            fig, ax = visualizer.plot_trajectory_on_potential(results, sample_idx=0)
            _save_plot(fig, output_path, "0_trajectory_on_potential.png")
        except ValueError as e:
            print(f"Skipping trajectory plot: {e}")

    # Position time series
    if "positions" in results:
        try:
            fig, axes = visualizer.plot_position_time_series(results, sample_idx=0)
            _save_plot(fig, output_path, "0_position_time_series.png")
        except ValueError as e:
            print(f"Skipping position time series: {e}")

    # Animated trajectory
    if "positions" in results and save_animation:
        try:
            print("Creating animated trajectory...")
            # Animation settings - modify these as needed:
            fps = 60 
            desired_duration = 60  # seconds
            max_frames = desired_duration * fps  # 10800
            frame_skip = max(1, len(results["positions"]) // max_frames)
            
            # Account for the save_every parameter in actual frame calculations
            total_sim_steps = results.get("n_steps", len(results["positions"]) * results.get("save_every", 1))
            save_every = results.get("save_every", 1)
            saved_frames = len(results["positions"])
            actual_frames = saved_frames // frame_skip
            duration = actual_frames / fps
            
            print(f"Simulation info: {total_sim_steps:,} total steps, save_every={save_every}")
            print(f"Data available: {saved_frames:,} saved frames")
            print(f"Animation settings: frame_skip={frame_skip}, will create {actual_frames:,} animation frames")
            print(f"Expected video duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            
            animation_path = visualizer.create_animated_trajectory(
                results, 
                sample_idx=0,
                output_path=output_path / "0_trajectory_animation.mp4",
                frames_per_second=fps,
                trail_length=150,
                frame_skip=frame_skip
            )
            print(f"Animation saved: {animation_path}")
        except Exception as e:
            print(f"Skipping trajectory animation: {e}")
            print("Note: Animation requires matplotlib.animation and may need ffmpeg for MP4 or pillow for GIF")

    # Velocity time series
    if "velocities" in results:
        try:
            fig, axes = visualizer.plot_velocity_time_series(results, sample_idx=0)
            _save_plot(fig, output_path, "0_velocity_time_series.png")
        except ValueError as e:
            print(f"Skipping velocity time series: {e}")

    # Force time series
    if "forces" in results:
        try:
            fig, axes = visualizer.plot_force_time_series(results, sample_idx=0)
            _save_plot(fig, output_path, "0_force_time_series.png")
        except ValueError as e:
            print(f"Skipping force time series: {e}")

    # Energy vs time
    if "potential_energy" in results:
        try:
            fig, ax = visualizer.plot_energy_vs_time(results, sample_idx=0)
            _save_plot(fig, output_path, "0_energy_vs_time.png")
        except ValueError as e:
            print(f"Skipping energy time series: {e}")

    print(f"All available plots saved to {output_path}")


def create_batch_visualizations(all_results: list, output_dir: str | Path = "plots", save_animation: bool = False):
    """Create and save visualizations for batch simulation results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Use the first result to set up visualizer
    visualizer = _setup_visualizer(all_results[0]["config"])
    
    # Get which observables are available in the first result
    available_observables = [obs for obs in ["positions", "velocities", "forces", "potential_energy"] if obs in all_results[0]]
    print(f"Creating batch visualizations for available observables: {available_observables}")

    # Plot potential surface
    print("Creating potential surface plot...")
    fig, ax = visualizer.plot_potential_surface()
    _save_plot(fig, output_path, "potential_surface.png")

    # For batch mode, create visualizations from the first simulation for time-dependent plots
    print("Creating visualizations from first simulation...")

    first_result = all_results[0]

    # Position distributions - aggregate across ALL simulations and particles
    if "positions" in first_result and all(("positions" in result) for result in all_results):
        try:
            print("Creating batch position distributions (aggregated across all simulations)...")
            fig, axes = visualizer.plot_batch_position_distributions(all_results)
            _save_plot(fig, output_path, "position_distributions.png")
        except ValueError as e:
            print(f"Skipping batch position distributions: {e}")

    # Velocity distributions - from first simulation across all particles
    if "velocities" in first_result:
        try:
            fig, axes = visualizer.plot_velocity_distributions(first_result)
            _save_plot(fig, output_path, "velocity_distributions.png")
        except ValueError as e:
            print(f"Skipping velocity distributions: {e}")

    # Force distributions - from first simulation across all particles
    if "forces" in first_result:
        try:
            fig, axes = visualizer.plot_force_distributions(first_result)
            _save_plot(fig, output_path, "force_distributions.png")
        except ValueError as e:
            print(f"Skipping force distributions: {e}")

    # Energy distribution - from first simulation across all particles
    if "potential_energy" in first_result:
        try:
            fig, ax = visualizer.plot_energy_distribution(first_result)
            _save_plot(fig, output_path, "energy_distribution.png")
        except ValueError as e:
            print(f"Skipping energy distribution: {e}")

    # Time-dependent plots from first simulation
    if "positions" in first_result:
        try:
            fig, ax = visualizer.plot_trajectory_on_potential(first_result, sample_idx=0)
            _save_plot(fig, output_path, "0_trajectory_on_potential.png")
        except ValueError as e:
            print(f"Skipping trajectory plot: {e}")

    # Position time series from first simulation
    if "positions" in first_result:
        try:
            fig, axes = visualizer.plot_position_time_series(first_result, sample_idx=0)
            _save_plot(fig, output_path, "0_position_time_series.png")
        except ValueError as e:
            print(f"Skipping position time series: {e}")

    # Animated trajectory from first simulation
    if "positions" in first_result and save_animation:
        try:
            print("Creating animated trajectory from first simulation...")
            # Animation settings - modify these as needed:
            fps = 60
            desired_duration = 90  # seconds - set to 90 for a 90-second video
            max_frames = desired_duration * fps  # 5400 frames for 90 seconds at 60fps
            frame_skip = max(1, len(first_result["positions"]) // max_frames)
            
            # Account for the save_every parameter in actual frame calculations
            total_sim_steps = first_result.get("n_steps", len(first_result["positions"]) * first_result.get("save_every", 1))
            save_every = first_result.get("save_every", 1)
            saved_frames = len(first_result["positions"])
            actual_frames = saved_frames // frame_skip
            duration = actual_frames / fps
            
            print(f"Simulation info: {total_sim_steps:,} total steps, save_every={save_every}")
            print(f"Data available: {saved_frames:,} saved frames")
            print(f"Animation settings: frame_skip={frame_skip}, will create {actual_frames:,} animation frames")
            print(f"Expected video duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            
            animation_path = visualizer.create_animated_trajectory(
                first_result, 
                sample_idx=0,
                output_path=output_path / "0_trajectory_animation.mp4",
                frames_per_second=fps,
                trail_length=150,
                frame_skip=frame_skip
            )
            print(f"Animation saved: {animation_path}")
        except Exception as e:
            print(f"Skipping trajectory animation: {e}")
            print("Note: Animation requires matplotlib.animation and may need ffmpeg for MP4 or pillow for GIF")

    if "velocities" in first_result:
        try:
            fig, axes = visualizer.plot_velocity_time_series(first_result, sample_idx=0)
            _save_plot(fig, output_path, "0_velocity_time_series.png")
        except ValueError as e:
            print(f"Skipping velocity time series: {e}")

    if "forces" in first_result:
        try:
            fig, axes = visualizer.plot_force_time_series(first_result, sample_idx=0)
            _save_plot(fig, output_path, "0_force_time_series.png")
        except ValueError as e:
            print(f"Skipping force time series: {e}")

    if "potential_energy" in first_result:
        try:
            fig, ax = visualizer.plot_energy_vs_time(first_result, sample_idx=0)
            _save_plot(fig, output_path, "0_energy_vs_time.png")
        except ValueError as e:
            print(f"Skipping energy time series: {e}")

    print(f"Batch plots saved to {output_path}")


def demo_potential_features():
    """Demonstrate the features of the Müller-Brown potential."""
    print("=== Müller-Brown Potential Demo ===")

    potential = MuellerBrownPotential(dtype=torch.float64)

    print(f"Potential has {potential.ndims} dimensions")
    print(f"Minima locations: {potential.get_minima()}")
    print(f"Saddle point locations: {potential.get_saddle_points()}")

    # Test potential evaluation
    test_coords = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    energies = potential(test_coords)
    print(f"Energies at test coordinates: {energies}")

    # Test force calculation
    test_coords.requires_grad_(True)
    forces = potential.force(test_coords)
    print(f"Forces at test coordinates: {forces}")


def run_plotting_mode(artifact_dir: str | Path):
    """Load data from an artifact directory and regenerate all plots."""
    artifact_path = Path(artifact_dir)

    if not artifact_path.exists() or not artifact_path.is_dir():
        print(f"Error: Artifact directory does not exist: {artifact_path}")
        return

    print(f"Loading data from artifact directory: {artifact_path}")

    # Load simulation data
    data = load_artifact_data(artifact_path)
    if data is None:
        print(f"Error: No simulation data found in {artifact_path}")
        return

    # Reconstruct results dictionary with required structure
    results = {}
    
    # Add metadata
    metadata_keys = ["dt", "n_particles", "n_steps", "save_every", "temperature", "friction", "mass"]
    for key in metadata_keys:
        if key in data:
            results[key] = data[key]
    
    # Add only available observables
    observable_keys = ["positions", "velocities", "forces", "potential_energy"]
    for key in observable_keys:
        if key in data:
            results[key] = data[key]

    # Add basic config structure (needed for visualizer)
    results["config"] = {
        "simulation": {
            "device": "cpu",
            "temperature": data.get("temperature", 15.0),
            "friction": data.get("friction", 1.0),
            "dt": data.get("dt", 0.01),
        }
    }

    print("Regenerating plots...")
    create_visualizations(results, output_dir=artifact_path, save_animation=False)  # Don't create animations by default in plot mode
    print(f"Plots saved to: {artifact_path}")


def list_artifacts():
    """List all available artifact directories."""
    artifacts = get_artifact_directories()
    if not artifacts:
        print("No artifact directories found.")
        return

    print("Available artifact directories:")
    for i, artifact_dir in enumerate(artifacts, 1):
        print(f"  {i:2d}. {artifact_dir.name} ({artifact_dir})")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Müller-Brown potential simulation")
    parser.add_argument(
        "--mode",
        choices=["demo", "single", "batch", "plot", "list"],
        default="demo",
        help="Simulation mode: demo (show potential info), single (one simulation), batch (multiple simulations), plot (regenerate plots from artifact), list (show available artifacts)",
    )
    parser.add_argument(
        "--artifact-dir", type=str, help="Artifact directory for plot mode"
    )
    parser.add_argument(
        "--n-particles", type=int, default=1, help="Number of particles"
    )
    parser.add_argument(
        "--n-steps", type=int, default=1_000_000, help="Number of simulation steps"
    )
    parser.add_argument(
        "--temperature", type=float, default=15.0, help="Temperature for thermostat"
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=10,
        help="Number of simulations for batch mode",
    )
    parser.add_argument(
        "--n-transient",
        type=int,
        default=0,
        help="Number of beginning steps to discard as transient (rounded down to nearest save_every interval)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save data every N simulation steps (sampling rate)",
    )
    parser.add_argument(
        "--save-plots", action="store_true", help="Save visualization plots"
    )
    parser.add_argument(
        "--save-animation", action="store_true", help="Save animated trajectory visualization (MP4/GIF)"
    )
    parser.add_argument(
        "--observables",
        nargs="*",
        default=None,
        help="List of observables to save and plot. Available: positions, velocities, forces, potential_energy. Default: all observables",
    )

    args = parser.parse_args()

    if args.mode == "demo":
        demo_potential_features()

    elif args.mode == "list":
        list_artifacts()

    elif args.mode == "plot":
        if not args.artifact_dir:
            print("Error: --artifact-dir required for plot mode")
            return
        run_plotting_mode(args.artifact_dir)

    elif args.mode == "single":
        config = create_experiment_config(
            n_particles=args.n_particles,
            n_steps=args.n_steps,
            temperature=args.temperature,
            n_transient=args.n_transient,
            seed=args.seed,
            save_every=args.save_every,
            observables=args.observables,
        )

        results = run_single_simulation(config)

        # Save to new artifact directory with specified observables
        observables_to_save = config["output"]["observables"]
        artifact_path = save_simulation_data(
            results, create_artifact_dir=True, observables=observables_to_save
        )

        if args.save_plots:
            create_visualizations(results, output_dir=artifact_path.parent, save_animation=args.save_animation)

        print(f"Results saved to artifact directory: {artifact_path.parent}")

    elif args.mode == "batch":
        all_results, batch_artifact_dir = run_batch_simulation(
            n_simulations=args.n_simulations,
            n_particles=args.n_particles,
            n_steps=args.n_steps,
            temperature=args.temperature,
            n_transient=args.n_transient,
            seed=args.seed,
            save_every=args.save_every,
            observables=args.observables,
        )

        print(f"\nCompleted {len(all_results)} simulations")
        if args.save_plots:
            # For batch mode, create visualizations in the same artifact directory
            print("Creating visualizations from batch results...")
            create_batch_visualizations(all_results, output_dir=batch_artifact_dir, save_animation=args.save_animation)
            print(f"Batch visualizations saved to: {batch_artifact_dir}")


if __name__ == "__main__":
    main()
