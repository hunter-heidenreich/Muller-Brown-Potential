#!/usr/bin/env python3
"""
Simple example script demonstrating the Müller-Brown simulation package.

This script shows the basic usage pattern for running simulations
and creating visualizations programmatically.
"""

import torch

from src.muller_brown import (
    MuellerBrownPotential,
    LangevinSimulator,
    MuellerBrownVisualizer,
    save_simulation_data,
)


def main():
    """Run a simple example simulation."""
    print("=== Müller-Brown Simulation Example ===")
    
    # 1. Create the potential
    print("Setting up Müller-Brown potential...")
    potential = MuellerBrownPotential(dtype=torch.float64)
    
    # Show potential features
    print(f"Potential minima: {potential.get_minima()}")
    print(f"Saddle points: {potential.get_saddle_points()}")
    
    # 2. Set up the simulator
    print("Initializing Langevin simulator...")
    simulator = LangevinSimulator(
        potential=potential,
        temperature=15.0,  # Temperature for thermostat
        friction=1.0,      # Friction coefficient
        dt=0.01           # Time step
    )
    
    # 3. Run a simulation
    print("Running simulation...")
    initial_positions = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    
    results = simulator.simulate(
        initial_positions=initial_positions,
        n_steps=10000,
        save_every=10
    )
    
    print(f"Simulation completed! Generated {len(results['positions'])} data points")
    
    # 4. Save the data
    print("Saving simulation data...")
    save_path = save_simulation_data(results, create_artifact_dir=True)
    print(f"Data saved to: {save_path}")
    
    # 5. Create visualizations
    print("Creating visualizations...")
    visualizer = MuellerBrownVisualizer(potential)
    
    # Plot potential surface
    fig, ax = visualizer.plot_potential_surface()
    fig.savefig(save_path.parent / "potential_surface.png", dpi=300, bbox_inches="tight")
    
    # Plot trajectory
    fig, axes = visualizer.plot_trajectory_on_potential(results, sample_idx=0)
    fig.savefig(save_path.parent / "trajectory.png", dpi=300, bbox_inches="tight")
    
    print(f"Plots saved to: {save_path.parent}")
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
