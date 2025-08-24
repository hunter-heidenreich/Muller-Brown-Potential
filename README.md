# Müller-Brown Potential Simulation

A modern, modular PyTorch implementation of the Müller-Brown potential energy surface with Langevin dynamics simulation.

## Overview

This package provides a clean, well-structured implementation of the famous Müller-Brown potential, a 2D model potential energy surface widely used for testing molecular dynamics algorithms and studying rare events in chemical systems.

**Reference**: Müller, K., & Brown, L. D. (1979). Location of saddle points and minimum energy paths by a constrained simplex optimization procedure. *Theoretica Chimica Acta*, 53, 75-93.

## Features

- **Modern PyTorch Implementation**: Uses PyTorch tensors with automatic differentiation for force calculations
- **Modular Design**: Clear separation of concerns with dedicated modules for potential, simulation, and visualization
- **Langevin Dynamics**: Proper implementation of velocity-Verlet integration with Langevin thermostat
- **Rich Visualization**: High-quality plots of potential surfaces, trajectories, and energy evolution
- **Comprehensive Utilities**: Data I/O, statistics calculation, and experiment configuration
- **Type Hints**: Full type annotations for better code clarity and IDE support
- **Testing**: Comprehensive test suite with pytest

## Installation

1. Install dependencies using uv:
```bash
uv sync
```

Or manually install the dependencies:
```bash
pip install torch numpy matplotlib scipy
```

2. For development, install the optional dev dependencies:
```bash
uv sync --dev
```

## Quick Start

### Basic Usage

```python
import torch
from src.muller_brown import MuellerBrownPotential, LangevinSimulator, PotentialVisualizer

# Create the potential
potential = MuellerBrownPotential(dtype=torch.float64)

# Set up the simulator
simulator = LangevinSimulator(
    potential=potential,
    temperature=15.0,
    friction=1.0,
    dt=0.01
)

# Generate initial positions
initial_positions = torch.tensor([[0.0, 0.0]], dtype=torch.float64)

# Run simulation
results = simulator.simulate(
    initial_positions=initial_positions,
    n_steps=10000,
    save_every=10
)

# Visualize results
visualizer = PotentialVisualizer(potential)
fig, ax = visualizer.plot_trajectories(results["trajectory"])
```

### Command Line Interface

Run different modes of simulation:

```bash
# Demo mode - showcase potential features
python main.py --mode demo

# Single simulation
python main.py --mode single --n-particles 5 --n-steps 50000 --save-plots

# Batch simulations
python main.py --mode batch --n-simulations 20 --n-particles 1 --temperature 10.0
```

## Project Structure

```
muller_brown/
├── src/muller_brown/
│   ├── __init__.py          # Package exports
│   ├── potential.py         # Müller-Brown potential implementation
│   ├── simulation.py        # Langevin dynamics simulator  
│   ├── visualization.py     # Plotting and visualization tools
│   └── utils.py            # Utilities for data handling
├── tests/
│   └── test_muller_brown.py # Test suite
├── main.py                 # Main CLI script
├── pyproject.toml         # Project metadata and dependencies
└── README.md              # This file
```

## Key Classes

### `MuellerBrownPotential`
- PyTorch module implementing the 2D potential energy surface
- Automatic gradient computation for forces
- Built-in critical point locations (minima and saddle points)

### `LangevinSimulator`  
- Velocity-Verlet integration with Langevin thermostat
- Configurable temperature, friction, and time step
- Efficient batch simulation capabilities

### `PotentialVisualizer`
- High-quality contour plots of the potential surface
- Trajectory visualization with customizable styling
- Energy evolution plots for dynamics analysis

## Examples

### Exploring the Potential Surface

```python
from src.muller_brown import MuellerBrownPotential, PotentialVisualizer

potential = MuellerBrownPotential()

# Get critical points
minima = potential.get_minima()
saddles = potential.get_saddle_points()

print(f"Minima: {minima}")
print(f"Saddle points: {saddles}")

# Visualize the surface
visualizer = PotentialVisualizer(potential)
fig, ax = visualizer.plot_potential()
```

### Multiple Particle Simulation

```python
import numpy as np
from src.muller_brown.utils import generate_initial_positions

# Generate 10 random starting positions
initial_pos = generate_initial_positions(10, method="random")

# Run simulation
results = simulator.simulate(
    initial_positions=initial_pos,
    n_steps=100000,
    save_every=100
)

# Analyze results
from src.muller_brown.utils import calculate_trajectory_statistics
stats = calculate_trajectory_statistics(results["trajectory"])
print(f"Mean final displacement: {stats['mean_displacement']:.3f}")
```

## Testing

Run the test suite:

```bash
pytest tests/
```

## Performance

The implementation is optimized for:
- **GPU acceleration**: All tensors can be moved to CUDA devices
- **Batch processing**: Efficient simulation of multiple particles
- **Memory efficiency**: Configurable trajectory saving frequency
- **Numerical stability**: Double precision by default

## Contributing

1. Install development dependencies: `uv sync --dev`
2. Run tests: `pytest`
3. Format code: `black .`
4. Lint: `ruff check .`

## License

This project is available under the MIT License.
