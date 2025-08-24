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
    calculate_trajectory_statistics,
    set_random_seed,
    apply_transient_removal,
    compute_batch_statistics,
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

    # Calculate statistics
    stats = calculate_trajectory_statistics(raw_results)  # Use full data for statistics
    results["statistics"] = stats
    results["config"] = config

    print("Simulation completed!")
    print(f"Final mean displacement: {stats['mean_displacement']:.3f}")

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


def create_visualizations(results: dict, output_dir: str | Path = "plots"):
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
            fig, axes = visualizer.plot_trajectory_on_potential(results, sample_idx=0)
            _save_plot(fig, output_path, "0_trajectory_on_potential.png")
        except ValueError as e:
            print(f"Skipping trajectory plot: {e}")

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

    # MSD vs time
    if "positions" in results:
        try:
            fig, ax = visualizer.plot_msd_vs_time(results, sample_idx=0)
            _save_plot(fig, output_path, "0_msd_vs_time.png")
        except ValueError as e:
            print(f"Skipping MSD plot: {e}")

    # Energy vs time
    if "potential_energy" in results:
        try:
            fig, ax = visualizer.plot_energy_vs_time(results, sample_idx=0)
            _save_plot(fig, output_path, "0_energy_vs_time.png")
        except ValueError as e:
            print(f"Skipping energy time series: {e}")

    print(f"All available plots saved to {output_path}")


def create_batch_visualizations(all_results: list, output_dir: str | Path = "plots"):
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
            fig, axes = visualizer.plot_trajectory_on_potential(first_result, sample_idx=0)
            _save_plot(fig, output_path, "0_trajectory_on_potential.png")
        except ValueError as e:
            print(f"Skipping trajectory plot: {e}")

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

    if "positions" in first_result:
        try:
            fig, ax = visualizer.plot_msd_vs_time(first_result, sample_idx=0)
            _save_plot(fig, output_path, "0_msd_vs_time.png")
        except ValueError as e:
            print(f"Skipping MSD plot: {e}")

    if "potential_energy" in first_result:
        try:
            fig, ax = visualizer.plot_energy_vs_time(first_result, sample_idx=0)
            _save_plot(fig, output_path, "0_energy_vs_time.png")
        except ValueError as e:
            print(f"Skipping energy time series: {e}")

    # Batch-specific statistics
    print("Creating batch statistics plots...")
    _create_batch_statistics_plots(all_results, output_path, visualizer)

    print(f"Batch plots saved to {output_path}")


def _create_batch_statistics_plots(
    all_results: list, output_path: Path, visualizer: MuellerBrownVisualizer
):
    """Create batch-specific statistical analysis plots."""
    import numpy as np

    # Extract statistics from all simulations
    displacements = [r["statistics"]["mean_displacement"] for r in all_results]
    
    # Only create final position plots if positions are available
    if "positions" in all_results[0]:
        final_positions = [
            r["positions"][-1] for r in all_results
        ]  # Final positions of all particles
    else:
        final_positions = None

    # 1. Displacement distribution histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(displacements, bins=20, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Mean Displacement")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"Distribution of Mean Displacements\n({len(all_results)} simulations)"
    )
    ax.grid(True, alpha=0.3)
    _save_plot(fig, output_path, "displacement_histogram.png")

    # 2. Final position scatter plot on potential surface (only if positions available)
    if final_positions is not None:
        fig, ax = visualizer.plot_potential_surface()

        # Plot all final positions
        all_final_pos = np.concatenate(final_positions, axis=0)
        ax.scatter(
            all_final_pos[:, 0],
            all_final_pos[:, 1],
            c="white",
            s=30,
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,
        )

        ax.set_title(
            f"Final Positions from {len(all_results)} Simulations\n({all_final_pos.shape[0]} particles total)"
        )
        _save_plot(fig, output_path, "final_positions_scatter.png")

    # 3. Convergence plot - how displacement varies across simulations
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(displacements) + 1), displacements, "bo-", markersize=4)
    ax.set_xlabel("Simulation Number")
    ax.set_ylabel("Mean Displacement")
    ax.set_title("Displacement Convergence Across Simulations")
    ax.grid(True, alpha=0.3)

    # Add mean line
    mean_disp = np.mean(displacements)
    ax.axhline(
        mean_disp, color="red", linestyle="--", label=f"Overall Mean: {mean_disp:.3f}"
    )
    ax.legend()
    _save_plot(fig, output_path, "displacement_convergence.png")


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

    # Calculate statistics if not present (only if positions are available)
    if "statistics" not in results and "positions" in results:
        # For statistics calculation, we need a full results structure
        full_results = results.copy()
        # Use positions for displacement calculation even if other observables are missing
        results["statistics"] = calculate_trajectory_statistics(full_results)

    print("Regenerating plots...")
    create_visualizations(results, output_dir=artifact_path)
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
        "--n-steps", type=int, default=600_000, help="Number of simulation steps"
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
        default=100_000,
        help="Number of beginning steps to discard as transient (rounded down to nearest save_every interval)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save data every N simulation steps (sampling rate)",
    )
    parser.add_argument(
        "--save-plots", action="store_true", help="Save visualization plots"
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
            create_visualizations(results, output_dir=artifact_path.parent)

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

        # Analyze batch results using utility function
        batch_stats = compute_batch_statistics(all_results)
        displacement_stats = batch_stats["mean_displacement_stats"]

        print("Mean displacement statistics:")
        print(f"  Average: {displacement_stats['mean']:.3f}")
        print(f"  Std Dev: {displacement_stats['std']:.3f}")
        print(
            f"  Range: [{displacement_stats['min']:.3f}, {displacement_stats['max']:.3f}]"
        )

        if args.save_plots:
            # For batch mode, create visualizations in the same artifact directory
            print("Creating visualizations from batch results...")
            create_batch_visualizations(all_results, output_dir=batch_artifact_dir)
            print(f"Batch visualizations saved to: {batch_artifact_dir}")


if __name__ == "__main__":
    main()
