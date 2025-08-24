#!/usr/bin/env python3
"""
Modern implementation of Müller-Brown potential simulation.

This script demonstrates the usage of the modular Müller-Brown simulation package
with clean separation of concerns and modern PyTorch practices.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.muller_brown import MuellerBrownPotential, LangevinSimulator, PotentialVisualizer
from src.muller_brown.utils import (
    save_simulation_data,
    generate_initial_positions,
    create_experiment_config,
    calculate_trajectory_statistics,
)


def set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    print(f"Random seed set to: {seed}")


def apply_transient_removal(results: dict, n_transient: int) -> dict:
    """
    Remove transient steps from simulation results.
    
    Args:
        results: Simulation results dictionary
        n_transient: Number of initial steps to discard (in terms of saved trajectory points)
        
    Returns:
        Modified results with transient steps removed
    """
    if n_transient <= 0:
        return results
    
    trajectory = results["trajectory"]  # Shape: (n_saved_steps, n_particles, 2)
    save_every = results.get("save_every", 1)
    
    # Convert n_transient from simulation steps to saved trajectory points
    n_transient_saved = max(1, n_transient // save_every)
    
    if n_transient_saved >= trajectory.shape[0]:
        print(f"Warning: n_transient ({n_transient}) corresponds to {n_transient_saved} saved points, "
              f"but trajectory only has {trajectory.shape[0]} points. Removing {trajectory.shape[0] - 1} points instead.")
        n_transient_saved = trajectory.shape[0] - 1  # Keep at least one point
    
    if n_transient_saved == 0:
        print(f"Note: n_transient ({n_transient}) is smaller than save_every ({save_every}), no transient removal applied.")
        return results
    
    # Remove transient steps
    results["trajectory"] = trajectory[n_transient_saved:]
    
    # Update metadata
    original_n_saved = trajectory.shape[0]
    results["n_saved_original"] = original_n_saved
    results["n_transient_removed"] = n_transient_saved
    results["n_saved_after_transient"] = results["trajectory"].shape[0]  # Use the actual new shape
    
    print(f"Removed {n_transient_saved} transient trajectory points (corresponding to ~{n_transient_saved * save_every} simulation steps)")
    print(f"Saved trajectory length: {original_n_saved} -> {results['n_saved_after_transient']} points")
    
    return results


def run_single_simulation(config: dict) -> dict:
    """Run a single simulation with the given configuration."""
    # Set random seed
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
        n_particles=config["simulation"]["n_particles"],
        **config["initial_conditions"]
    )
    
    print(f"Starting positions:\n{initial_positions}")
    
    print("Running simulation...")
    results = simulator.simulate(
        initial_positions=initial_positions,
        n_steps=config["simulation"]["n_steps"],
        save_every=config["simulation"]["save_every"],
    )
    
    # Apply transient removal if specified
    n_transient = config["simulation"].get("n_transient", 0)
    if n_transient > 0:
        results = apply_transient_removal(results, n_transient)
    
    # Calculate statistics
    stats = calculate_trajectory_statistics(results["trajectory"])
    results["statistics"] = stats
    results["config"] = config
    
    print("Simulation completed!")
    print(f"Final mean displacement: {stats['mean_displacement']:.3f}")
    
    return results


def run_batch_simulation(n_simulations: int = 10, **sim_kwargs) -> list:
    """Run multiple simulations with random initial conditions."""
    print(f"Running {n_simulations} simulations...")
    
    all_results = []
    base_seed = sim_kwargs["seed"]  # Now always has a value
    
    for i in range(n_simulations):
        print(f"\n--- Simulation {i+1}/{n_simulations} ---")
        
        # Create configuration for this simulation
        # Use incremented seeds for each simulation
        sim_kwargs_copy = sim_kwargs.copy()
        sim_kwargs_copy["seed"] = base_seed + i
            
        config = create_experiment_config(**sim_kwargs_copy)
        
        # Run simulation
        results = run_single_simulation(config)
        all_results.append(results)
        
        # Save individual result
        if config["output"]["save_data"]:
            save_simulation_data(results, filename=f"simulation_{i:03d}.pkl")
    
    return all_results


def create_visualizations(results: dict, output_dir: str = "plots"):
    """Create and save visualizations for simulation results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    device = results.get("config", {}).get("simulation", {}).get("device", "cpu")
    potential = MuellerBrownPotential(device=device, dtype=torch.float64)
    visualizer = PotentialVisualizer(potential)
    
    # Plot potential surface
    print("Creating potential surface plot...")
    fig, ax = visualizer.plot_potential()
    fig.savefig(output_path / "potential_surface.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Plot trajectories
    print("Creating trajectory plot...")
    trajectories = results["trajectory"]  # Shape: (n_steps, n_particles, 2)
    # Reshape to (n_particles, n_steps, 2) for visualization
    trajectories_reshaped = np.transpose(trajectories, (1, 0, 2))
    
    fig, ax = visualizer.plot_trajectories(trajectories_reshaped)
    fig.savefig(output_path / "trajectories.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Plot energy evolution
    print("Creating energy evolution plot...")
    fig, ax = visualizer.plot_energy_vs_time(results)
    fig.savefig(output_path / "energy_evolution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Plots saved to {output_path}")


def create_batch_visualizations(all_results: list, output_dir: str = "plots"):
    """Create and save visualizations for batch simulation results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Use the first result to get device and create potential
    first_result = all_results[0]
    device = first_result.get("config", {}).get("simulation", {}).get("device", "cpu")
    potential = MuellerBrownPotential(device=device, dtype=torch.float64)
    visualizer = PotentialVisualizer(potential)
    
    # Plot potential surface
    print("Creating potential surface plot...")
    fig, ax = visualizer.plot_potential()
    fig.savefig(output_path / "potential_surface.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Combine all trajectories for visualization
    print("Creating combined trajectory plot...")
    all_trajectories = []
    
    for result in all_results:
        trajectory = result["trajectory"]  # Shape: (n_steps, n_particles, 2)
        # Reshape to (n_particles, n_steps, 2) for visualization
        trajectory_reshaped = np.transpose(trajectory, (1, 0, 2))
        
        # Add each particle's trajectory to the list
        for particle_traj in trajectory_reshaped:
            all_trajectories.append(particle_traj)
    
    # Convert to numpy array for plotting
    combined_trajectories = np.array(all_trajectories)  # Shape: (total_particles, n_steps, 2)
    
    fig, ax = visualizer.plot_trajectories(combined_trajectories)
    fig.savefig(output_path / "batch_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Create energy evolution plot for the first simulation as example
    print("Creating energy evolution plot (first simulation)...")
    fig, ax = visualizer.plot_energy_vs_time(first_result)
    fig.savefig(output_path / "energy_evolution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Batch plots saved to {output_path}")
    print(f"Combined {len(all_trajectories)} particle trajectories from {len(all_results)} simulations")


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


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Müller-Brown potential simulation")
    parser.add_argument("--mode", choices=["demo", "single", "batch"], default="demo",
                       help="Simulation mode")
    parser.add_argument("--n-particles", type=int, default=1,
                       help="Number of particles")
    parser.add_argument("--n-steps", type=int, default=60_000,
                       help="Number of simulation steps")
    parser.add_argument("--temperature", type=float, default=15.0,
                       help="Temperature for thermostat")
    parser.add_argument("--n-simulations", type=int, default=10,
                       help="Number of simulations for batch mode")
    parser.add_argument("--n-transient", type=int, default=1000,
                       help="Number of beginning steps to discard as transient (rounded down to nearest save_every interval)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--save-plots", action="store_true",
                       help="Save visualization plots")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        demo_potential_features()
        
    elif args.mode == "single":
        config = create_experiment_config(
            n_particles=args.n_particles,
            n_steps=args.n_steps,
            temperature=args.temperature,
            n_transient=args.n_transient,
            seed=args.seed,
        )
        
        results = run_single_simulation(config)
        
        if args.save_plots:
            create_visualizations(results)
            
        # Save results
        save_simulation_data(results)
        
    elif args.mode == "batch":
        all_results = run_batch_simulation(
            n_simulations=args.n_simulations,
            n_particles=args.n_particles,
            n_steps=args.n_steps,
            temperature=args.temperature,
            n_transient=args.n_transient,
            seed=args.seed,
        )
        
        print(f"\nCompleted {len(all_results)} simulations")
        
        # Analyze batch results
        displacements = [r["statistics"]["mean_displacement"] for r in all_results]
        print("Mean displacement statistics:")
        print(f"  Average: {np.mean(displacements):.3f}")
        print(f"  Std Dev: {np.std(displacements):.3f}")
        print(f"  Range: [{np.min(displacements):.3f}, {np.max(displacements):.3f}]")

        if args.save_plots:
            # For batch mode, create visualizations from the first simulation
            # or combine all trajectories for a comprehensive view
            print("Creating visualizations from batch results...")
            create_batch_visualizations(all_results)

if __name__ == "__main__":
    main()
