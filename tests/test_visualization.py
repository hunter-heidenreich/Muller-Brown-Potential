"""Smoke tests for the plotting suite and trajectory statistics.

These don't check pixels; they confirm every plot method runs end to end on a
small simulation and returns a matplotlib Figure (the visualization module is
otherwise untested). Animation is excluded -- it needs an external writer.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from src.muller_brown import (
    MuellerBrownPotential,
    LangevinSimulator,
    MuellerBrownVisualizer,
    set_random_seed,
    calculate_trajectory_statistics,
)


@pytest.fixture(scope="module")
def results() -> dict:
    """A small run with all four observables stored."""
    set_random_seed(0)
    simulator = LangevinSimulator(MuellerBrownPotential(), temperature=15.0)
    return simulator.simulate(np.zeros((3, 2)), n_steps=100, save_every=10, progress=False)


@pytest.fixture(scope="module")
def visualizer() -> MuellerBrownVisualizer:
    return MuellerBrownVisualizer(MuellerBrownPotential())


def _assert_figure(returned):
    fig = returned[0]
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_potential_surface(visualizer):
    _assert_figure(visualizer.plot_potential_surface())


@pytest.mark.parametrize(
    "method",
    [
        "plot_position_distributions",
        "plot_velocity_distributions",
        "plot_force_distributions",
        "plot_energy_distribution",
        "plot_trajectory_on_potential",
        "plot_position_time_series",
        "plot_velocity_time_series",
        "plot_force_time_series",
        "plot_energy_vs_time",
    ],
)
def test_plot_methods_render(visualizer, results, method):
    _assert_figure(getattr(visualizer, method)(results))


def test_plot_batch_position_distributions(visualizer, results):
    _assert_figure(visualizer.plot_batch_position_distributions([results, results]))


def test_calculate_trajectory_statistics(results):
    stats = calculate_trajectory_statistics(results)
    assert stats["n_particles"] == 3
    assert stats["n_dimensions"] == 2
    # all observable families are present because results stored all four
    for key in ("position_mean", "velocity_std", "force_magnitude_mean", "energy_mean"):
        assert key in stats
    assert stats["position_mean"].shape == (2,)
