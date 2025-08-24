"""Basic tests for the Müller-Brown implementation."""

import numpy as np
import torch
import pytest
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from muller_brown import MuellerBrownPotential, LangevinSimulator, PotentialVisualizer


class TestMuellerBrownPotential:
    """Test the Müller-Brown potential implementation."""
    
    def test_initialization(self):
        """Test potential initialization."""
        potential = MuellerBrownPotential()
        assert potential.ndims == 2
        assert hasattr(potential, 'A')
        assert hasattr(potential, 'a')
        assert hasattr(potential, 'b')
        assert hasattr(potential, 'c')
    
    def test_potential_evaluation(self):
        """Test potential energy evaluation."""
        potential = MuellerBrownPotential(dtype=torch.float64)
        
        # Test single point
        coords = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        energy = potential(coords)
        assert energy.shape == (1,)
        assert torch.isfinite(energy).all()
        
        # Test multiple points
        coords = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        energies = potential(coords)
        assert energies.shape == (2,)
        assert torch.isfinite(energies).all()
    
    def test_force_calculation(self):
        """Test force calculation."""
        potential = MuellerBrownPotential(dtype=torch.float64)
        
        coords = torch.tensor([[0.0, 0.0]], dtype=torch.float64, requires_grad=True)
        forces = potential.force(coords)
        
        assert forces.shape == (1, 2)
        assert torch.isfinite(forces).all()
    
    def test_critical_points(self):
        """Test that critical points are returned correctly."""
        potential = MuellerBrownPotential()
        
        minima = potential.get_minima()
        saddles = potential.get_saddle_points()
        
        assert len(minima) == 3
        assert len(saddles) == 2
        
        # Check that all coordinates are reasonable
        for x, y in minima + saddles:
            assert -2 <= x <= 2
            assert -1 <= y <= 3


class TestLangevinSimulator:
    """Test the Langevin simulator."""
    
    def test_initialization(self):
        """Test simulator initialization."""
        potential = MuellerBrownPotential()
        simulator = LangevinSimulator(potential)
        
        assert simulator.potential is potential
        assert simulator.temperature > 0
        assert simulator.friction > 0
        assert simulator.dt > 0
    
    def test_short_simulation(self):
        """Test a very short simulation."""
        potential = MuellerBrownPotential(dtype=torch.float64)
        simulator = LangevinSimulator(potential, dt=0.01, dtype=torch.float64)
        
        initial_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        
        results = simulator.simulate(
            initial_positions=initial_pos,
            n_steps=10,
            save_every=1
        )
        
        assert "trajectory" in results
        assert "dt" in results
        assert "n_particles" in results
        
        trajectory = results["trajectory"]
        assert trajectory.shape == (10, 1, 2)
        assert np.isfinite(trajectory).all()


class TestPotentialVisualizer:
    """Test the visualization tools."""
    
    def test_initialization(self):
        """Test visualizer initialization."""
        potential = MuellerBrownPotential()
        visualizer = PotentialVisualizer(potential)
        
        assert visualizer.potential is potential
    
    def test_plot_potential(self):
        """Test potential plotting (without showing)."""
        potential = MuellerBrownPotential(dtype=torch.float64)
        visualizer = PotentialVisualizer(potential)
        
        # This should not raise an error
        fig, ax = visualizer.plot_potential()
        
        assert fig is not None
        assert ax is not None
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_integration():
    """Integration test of the full pipeline."""
    # Create potential
    potential = MuellerBrownPotential(dtype=torch.float64)
    
    # Create simulator
    simulator = LangevinSimulator(potential, dt=0.01, dtype=torch.float64)
    
    # Run short simulation
    initial_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    results = simulator.simulate(
        initial_positions=initial_pos,
        n_steps=50,
        save_every=5
    )
    
    # Test visualization
    visualizer = PotentialVisualizer(potential)
    
    trajectory = results["trajectory"]
    trajectory_reshaped = np.transpose(trajectory, (1, 0, 2))
    
    fig, ax = visualizer.plot_trajectories(trajectory_reshaped, plot_potential=False)
    
    import matplotlib.pyplot as plt
    plt.close(fig)
    
    assert True  # If we get here, everything worked


if __name__ == "__main__":
    pytest.main([__file__])
