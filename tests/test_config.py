"""Configuration helpers: initial-position generation, observable validation,
and experiment-config construction (including its input validation)."""

import numpy as np
import pytest

from muller_brown import (
    generate_initial_positions,
    validate_observables,
    create_experiment_config,
    DEFAULT_OBSERVABLES,
    set_random_seed,
)
from muller_brown.constants import MULLER_BROWN_MINIMA


def test_random_positions_have_right_shape():
    assert generate_initial_positions(10, method="random").shape == (10, 2)


def test_grid_positions_return_exactly_n():
    # 7 is not a perfect square; the grid is built then trimmed to n
    assert generate_initial_positions(7, method="grid").shape == (7, 2)


def test_near_minimum_clusters_around_chosen_minimum():
    set_random_seed(0)
    positions = generate_initial_positions(200, method="near_minimum", minimum_idx=1, sigma=0.05)
    assert positions.shape == (200, 2)
    center = np.array(MULLER_BROWN_MINIMA[1])
    assert np.linalg.norm(positions.mean(axis=0) - center) < 0.05


def test_near_minimum_out_of_range_index_falls_back():
    set_random_seed(0)
    positions = generate_initial_positions(50, method="near_minimum", minimum_idx=99, sigma=0.05)
    center = np.array(MULLER_BROWN_MINIMA[0])  # out-of-range index falls back to 0
    assert np.linalg.norm(positions.mean(axis=0) - center) < 0.05


def test_unknown_position_method_raises():
    with pytest.raises(ValueError):
        generate_initial_positions(5, method="bogus")


def test_non_positive_particle_count_raises():
    with pytest.raises(ValueError):
        generate_initial_positions(0)


def test_validate_observables_passes_through_valid():
    assert validate_observables(["positions", "forces"]) == ["positions", "forces"]


@pytest.mark.parametrize("bad", [["positions", "bogus"], [], "positions"])
def test_validate_observables_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        validate_observables(bad)


def test_config_has_expected_structure_and_defaults():
    config = create_experiment_config()
    assert set(config) == {"simulation", "initial_conditions", "output"}
    assert config["output"]["observables"] == DEFAULT_OBSERVABLES
    assert config["simulation"]["device"] == "cpu"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.0},
        {"friction": -1.0},
        {"dt": 0.0},
        {"n_transient": -5},
        {"n_particles": 0},
        {"n_steps": -1},
        {"save_every": 0},
    ],
)
def test_config_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        create_experiment_config(**kwargs)
