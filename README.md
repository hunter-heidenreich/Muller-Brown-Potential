# Müller-Brown Potential Simulation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A modern, modular PyTorch implementation of the Müller-Brown potential energy surface with Langevin dynamics simulation.

## Overview

This package provides a clean, well-structured implementation of the famous Müller-Brown potential, a 2D model potential energy surface widely used for testing molecular dynamics algorithms and studying rare events in chemical systems.

The implementation serves as a complete example of:
- **Modern scientific computing** with PyTorch and type hints
- **Modular code architecture** with clear separation of concerns
- **Comprehensive data management** with HDF5 and artifact organization
- **Rich visualization capabilities** for scientific analysis

**Reference**: Müller, K., & Brown, L. D. (1979). Location of saddle points and minimum energy paths by a constrained simplex optimization procedure. *Theoretica Chimica Acta*, 53, 75-93.

## Features

- **Modern PyTorch Implementation**: Uses PyTorch tensors with JIT compilation for efficient force calculations
- **HDF5 Data Storage**: Efficient, portable data format for simulation results
- **Langevin Dynamics**: Proper velocity-Verlet integration with Langevin thermostat

## The Müller-Brown Potential

The Müller-Brown potential is defined as a sum of four Gaussian-like terms:

$$V(x,y) = \sum_{i=1}^{4} A_i \exp\left(a_i(x-x_{0i})^2 + b_i(x-x_{0i})(y-y_{0i}) + c_i(y-y_{0i})^2\right)$$

This creates a complex energy landscape with:
- **3 local minima** at different energy levels
- **2 saddle points** connecting the minima
- **Energy barriers** that control transition rates between states

The potential is ideal for studying:
- Rare event transitions between metastable states
- Temperature effects on barrier crossing
- Path sampling and transition state theory
- Molecular dynamics algorithm performance

## Installation

Install dependencies using uv:
```bash
uv sync
```

## Quick Start

The easiest way to get started is with the demo mode:

```bash
uv run python main.py --mode demo
```

This shows the potential's key features: 3 minima, 2 saddle points, and demonstrates energy/force calculations.

### Command Line Interface

```bash
# Quick demo - see potential features
uv run python main.py --mode demo

# Single simulation with visualization
uv run python main.py --mode single --n-particles 1 --n-steps 50000 --save-plots

# Multiple simulations for statistics
uv run python main.py --mode batch --n-simulations 10 --n-particles 2 --save-plots

# List available artifact directories
uv run python main.py --mode list

# Regenerate plots from existing data
uv run python main.py --mode plot --artifact-dir artifacts/20250824_184610
```

### Common Usage Patterns

```bash
# Explore a single trajectory at moderate temperature
uv run python main.py --mode single --temperature 15.0 --n-steps 100000 --save-plots

# Study rare event transitions at lower temperature  
uv run python main.py --mode single --temperature 5.0 --n-steps 500000 --save-plots

# Generate ensemble statistics
uv run python main.py --mode batch --n-simulations 20 --temperature 15.0 --save-plots

# High-statistics run with multiple particles
uv run python main.py --mode single --n-particles 10 --n-steps 100000 --save-plots
```

### Artifact Directory Structure

All simulation results are organized into timestamped artifact directories under `artifacts/`:

```
artifacts/
├── 20250824_184610/          # Single simulation
│   ├── trajectory_*.h5       # Simulation data (HDF5)
│   ├── potential_surface.png # All generated plots
│   ├── 0_trajectory_on_potential.png
│   └── ...
└── 20250824_185504/          # Batch simulation
    ├── simulation_000.h5     # Individual simulation files
    ├── simulation_001.h5
    ├── simulation_002.h5
    ├── potential_surface.png # Combined visualizations
    ├── displacement_histogram.png
    └── ...
```

This structure ensures:
- **Data provenance**: Each dataset has a unique timestamp
- **Self-contained artifacts**: Data and plots are always together
- **Reproducibility**: Plots can be regenerated from any artifact directory
- **Organization**: Easy to manage multiple experiment runs

### Programmatic Usage

For a complete working example, see `example.py`. Here's the basic pattern:

```python
import torch
from src.muller_brown import MuellerBrownPotential, LangevinSimulator, MuellerBrownVisualizer

# Create the potential
potential = MuellerBrownPotential(dtype=torch.float64)

# Set up the simulator
simulator = LangevinSimulator(
    potential=potential,
    temperature=15.0,
    friction=1.0,
    dt=0.01
)

# Run simulation
results = simulator.simulate(
    initial_positions=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
    n_steps=10000,
    save_every=10
)

# Visualize results
visualizer = MuellerBrownVisualizer(potential)
fig, ax = visualizer.plot_potential_surface()
```

Run the example with:
```bash
uv run python example.py
```

### Managing Artifacts

The repository includes a utility to manage simulation artifacts:

```bash
# List all artifact directories
uv run python manage_artifacts.py list

# Clean artifacts older than 7 days (dry run)
uv run python manage_artifacts.py clean

# Actually delete old artifacts
uv run python manage_artifacts.py clean --delete
```

## Data Format

Simulations save all observables in HDF5 format:
- **Positions**: (n_steps, n_particles, 2) - x,y coordinates over time
- **Velocities**: (n_steps, n_particles, 2) - velocity components over time  
- **Forces**: (n_steps, n_particles, 2) - force components over time
- **Potential Energy**: (n_steps, n_particles) - potential energy over time
- **Metadata**: Temperature, friction, timestep, and simulation parameters

## Visualization Capabilities

### Time-Independent Distributions
- Position distributions (1D marginals + 2D joint with log scaling)
- Velocity component and magnitude distributions
- Force component and magnitude distributions  
- Potential energy distribution

### Time-Dependent Analysis
- Trajectory overlay on potential surface with x(t) and y(t) plots
- Velocity components and magnitude vs time
- Force components and magnitude vs time
- Mean squared displacement evolution
- Potential energy evolution

## Project Structure

```
muller_brown/
├── src/muller_brown/
│   ├── __init__.py          # Package exports
│   ├── potential.py         # Müller-Brown potential with JIT compilation
│   ├── simulation.py        # Langevin dynamics simulator
│   ├── visualization.py     # Comprehensive plotting suite
│   ├── io.py               # HDF5 data I/O operations
│   ├── data.py             # Data processing utilities
│   ├── analysis.py         # Statistical analysis functions
│   ├── config.py           # Experiment configuration
│   └── constants.py        # Physical and computational constants
├── main.py                 # Command-line interface
├── example.py              # Simple usage example
├── manage_artifacts.py     # Artifact management utility
├── pyproject.toml         # Project dependencies and metadata
├── LICENSE                # MIT License
└── README.md              # This file
```

## Key Classes

### `MuellerBrownPotential`
- JIT-compiled PyTorch module for efficient evaluation
- Automatic gradient computation for forces
- Built-in critical point locations (3 minima, 2 saddle points)

### `LangevinSimulator`
- Velocity-Verlet integration with proper Langevin thermostat
- Tracks all observables (positions, velocities, forces, energies)
- Configurable temperature, friction, and time step

### `MuellerBrownVisualizer`
- Comprehensive suite of distribution plots
- Time-series analysis for all observables
- Log-scale visualization for rare events
- Automatic critical point annotation

## Performance

The implementation is optimized for:
- **JIT Compilation**: PyTorch JIT for fast potential and force evaluation
- **GPU Support**: All tensors support CUDA acceleration
- **Efficient Storage**: HDF5 with compression for large datasets
- **Numerical Stability**: Double precision by default
- **Memory Management**: Configurable save frequency for long simulations

## Tips for Research Use

- **Temperature effects**: Lower temperatures (T ~ 5-10) emphasize rare events; higher temperatures (T ~ 20-30) increase barrier crossing
- **Equilibration**: Use `--n-transient` to remove initial transient behavior
- **Statistics**: Run batch simulations for ensemble averages and error estimates
- **Long trajectories**: For rare events, use 10⁶ steps or more with appropriate save frequency
- **Multiple particles**: Study collective behavior with `--n-particles > 1`

## License

This project is available under the MIT License.
