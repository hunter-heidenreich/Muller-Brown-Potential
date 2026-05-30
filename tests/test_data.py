"""Data utilities: transient removal, batch-trajectory combination, and tensor
conversion."""

import numpy as np
import torch

from muller_brown import apply_transient_removal, combine_batch_trajectories
from muller_brown.data import convert_to_tensor


def _fake_results(n_saved: int = 10, n_particles: int = 2, save_every: int = 100) -> dict:
    return {
        "positions": np.zeros((n_saved, n_particles, 2)),
        "velocities": np.zeros((n_saved, n_particles, 2)),
        "forces": np.zeros((n_saved, n_particles, 2)),
        "potential_energy": np.zeros((n_saved, n_particles)),
        "save_every": save_every,
    }


def test_transient_removal_drops_expected_frames():
    out = apply_transient_removal(_fake_results(n_saved=10, save_every=100), n_transient=300)
    assert out["positions"].shape[0] == 7
    assert out["n_transient_removed"] == 3
    # all observables are trimmed consistently
    for key in ("velocities", "forces", "potential_energy"):
        assert out[key].shape[0] == 7


def test_transient_removal_is_noop_for_zero():
    assert apply_transient_removal(_fake_results(n_saved=10), 0)["positions"].shape[0] == 10


def test_transient_removal_removes_at_least_one_frame():
    # n_transient < save_every still drops the initial frame (max(1, ...))
    out = apply_transient_removal(_fake_results(n_saved=10, save_every=100), n_transient=50)
    assert out["positions"].shape[0] == 9


def test_transient_removal_caps_at_trajectory_length():
    out = apply_transient_removal(_fake_results(n_saved=5, save_every=100), n_transient=100_000)
    assert out["positions"].shape[0] == 1


def test_combine_batch_trajectories_flattens_particles():
    results = [_fake_results(n_saved=10, n_particles=2) for _ in range(3)]
    combined = combine_batch_trajectories(results)
    # 3 simulations x 2 particles = 6 trajectories, each (n_saved, 2)
    assert combined.shape == (6, 10, 2)


def test_convert_to_tensor_from_numpy():
    tensor = convert_to_tensor(np.zeros((3, 2)), dtype=torch.float64)
    assert tensor.dtype == torch.float64
    assert tuple(tensor.shape) == (3, 2)


def test_convert_to_tensor_recasts_existing_tensor():
    tensor = convert_to_tensor(torch.ones(2, 2, dtype=torch.float64), dtype=torch.float32)
    assert tensor.dtype == torch.float32
