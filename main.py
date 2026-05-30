#!/usr/bin/env python3
"""
Command-line interface for Müller-Brown potential simulation.

Entry point over the simulation package: demo, single-run, batch, and
plot-regeneration modes wiring the potential, integrator, and visualizer.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from muller_brown import (
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
    results = simulator.simulate(
        initial_positions=initial_positions,
        n_steps=config["simulation"]["n_steps"],
        save_every=config["simulation"]["save_every"],
        observables=config["output"]["observables"],
    )

    # Apply transient removal if specified using utility function
    n_transient = config["simulation"].get("n_transient", 0)
    if n_transient > 0:
        results = apply_transient_removal(results, n_transient)

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
    from muller_brown import create_artifact_directory

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


def _try_save_plot(plot_fn, output_path: Path, filename: str, enabled: bool = True):
    """Render a plot and save it, skipping (with a message) on missing data."""
    if not enabled:
        return
    try:
        fig = plot_fn()[0]
        _save_plot(fig, output_path, filename)
    except ValueError as e:
        print(f"Skipping {filename}: {e}")


def _save_trajectory_animation(
    visualizer: MuellerBrownVisualizer, result: dict, output_path: Path, desired_duration: int
):
    """Render the animated trajectory for one result at the target duration (seconds)."""
    try:
        fps = 60
        saved_frames = len(result["positions"])
        frame_skip = max(1, saved_frames // (desired_duration * fps))
        save_every = result.get("save_every", 1)
        total_sim_steps = result.get("n_steps", saved_frames * save_every)
        actual_frames = saved_frames // frame_skip
        duration = actual_frames / fps

        print(f"Simulation info: {total_sim_steps:,} total steps, save_every={save_every}")
        print(f"Data available: {saved_frames:,} saved frames")
        print(f"Animation settings: frame_skip={frame_skip}, will create {actual_frames:,} animation frames")
        print(f"Expected video duration: {duration:.1f} seconds ({duration / 60:.1f} minutes)")

        animation_path = visualizer.create_animated_trajectory(
            result,
            sample_idx=0,
            output_path=output_path / "0_trajectory_animation.mp4",
            frames_per_second=fps,
            trail_length=150,
            frame_skip=frame_skip,
        )
        print(f"Animation saved: {animation_path}")
    except Exception as e:
        print(f"Skipping trajectory animation: {e}")
        print("Note: Animation requires matplotlib.animation and may need ffmpeg for MP4 or pillow for GIF")


def _render_visualizations(
    result: dict,
    output_dir: str | Path,
    save_animation: bool,
    animation_duration: int = 60,
    position_distribution_fn=None,
):
    """Render the full plot suite for one representative `result`.

    `position_distribution_fn(visualizer)` overrides how the position
    distribution figure is produced; batch mode aggregates across all
    simulations. Plots whose observables are absent are skipped.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    visualizer = _setup_visualizer(result["config"])
    available = [o for o in ("positions", "velocities", "forces", "potential_energy") if o in result]
    print(f"Creating visualizations for available observables: {available}")

    _save_plot(visualizer.plot_potential_surface()[0], output_path, "potential_surface.png")

    if position_distribution_fn is None:
        def position_distribution_fn(viz):
            return viz.plot_position_distributions(result)
    _try_save_plot(lambda: position_distribution_fn(visualizer), output_path,
                   "position_distributions.png", "positions" in result)
    _try_save_plot(lambda: visualizer.plot_velocity_distributions(result), output_path,
                   "velocity_distributions.png", "velocities" in result)
    _try_save_plot(lambda: visualizer.plot_force_distributions(result), output_path,
                   "force_distributions.png", "forces" in result)
    _try_save_plot(lambda: visualizer.plot_energy_distribution(result), output_path,
                   "energy_distribution.png", "potential_energy" in result)

    _try_save_plot(lambda: visualizer.plot_trajectory_on_potential(result, sample_idx=0), output_path,
                   "0_trajectory_on_potential.png", "positions" in result)
    _try_save_plot(lambda: visualizer.plot_position_time_series(result, sample_idx=0), output_path,
                   "0_position_time_series.png", "positions" in result)

    if save_animation and "positions" in result:
        _save_trajectory_animation(visualizer, result, output_path, animation_duration)

    _try_save_plot(lambda: visualizer.plot_velocity_time_series(result, sample_idx=0), output_path,
                   "0_velocity_time_series.png", "velocities" in result)
    _try_save_plot(lambda: visualizer.plot_force_time_series(result, sample_idx=0), output_path,
                   "0_force_time_series.png", "forces" in result)
    _try_save_plot(lambda: visualizer.plot_energy_vs_time(result, sample_idx=0), output_path,
                   "0_energy_vs_time.png", "potential_energy" in result)

    print(f"All available plots saved to {output_path}")


def create_visualizations(results: dict, output_dir: str | Path = "plots", save_animation: bool = False):
    """Create and save all visualizations for a single simulation's results."""
    _render_visualizations(results, output_dir, save_animation, animation_duration=60)


def create_batch_visualizations(all_results: list, output_dir: str | Path = "plots", save_animation: bool = False):
    """Create and save visualizations for batch results.

    Position distributions are aggregated across all simulations; per-particle
    distributions and time-dependent plots use the first simulation.
    """
    _render_visualizations(
        all_results[0],
        output_dir,
        save_animation,
        animation_duration=90,
        position_distribution_fn=lambda viz: viz.plot_batch_position_distributions(all_results),
    )


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
    parser.add_argument(
        "--device", type=str, default="cpu", help="Torch device for the simulation (e.g. cpu, cuda)"
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
            device=args.device,
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
            device=args.device,
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
