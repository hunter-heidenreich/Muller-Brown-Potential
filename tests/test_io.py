"""HDF5 persistence round-trips arrays and metadata, and honors the observable
subset chosen at save time."""

import numpy as np
import torch

from src.muller_brown import (
    MuellerBrownPotential,
    LangevinSimulator,
    save_simulation_data,
    load_simulation_data,
)

ALL_OBSERVABLES = ["positions", "velocities", "forces", "potential_energy"]
METADATA_KEYS = ["dt", "n_particles", "n_steps", "save_every", "temperature", "friction", "mass"]


def _make_results(observables: list[str]) -> dict:
    potential = MuellerBrownPotential(dtype=torch.float64)
    simulator = LangevinSimulator(potential, temperature=15.0, friction=1.0, dt=0.01)
    return simulator.simulate(
        np.array([[0.0, 0.0]]), n_steps=500, save_every=50, observables=observables, progress=False
    )


def test_roundtrip_preserves_arrays_and_metadata(tmp_path):
    results = _make_results(ALL_OBSERVABLES)
    path = save_simulation_data(
        results, output_dir=tmp_path, filename="run.h5", observables=ALL_OBSERVABLES
    )
    loaded = load_simulation_data(path)

    for obs in ALL_OBSERVABLES:
        assert np.array_equal(loaded[obs], results[obs]), f"{obs} changed across save/load"
    for key in METADATA_KEYS:
        assert loaded[key] == results[key], f"metadata {key} changed across save/load"


def test_save_writes_only_requested_observables(tmp_path):
    results = _make_results(ALL_OBSERVABLES)
    path = save_simulation_data(
        results, output_dir=tmp_path, filename="subset.h5", observables=["positions"]
    )
    loaded = load_simulation_data(path)

    assert "positions" in loaded
    for obs in ["velocities", "forces", "potential_energy"]:
        assert obs not in loaded, f"{obs} was saved despite not being requested"
