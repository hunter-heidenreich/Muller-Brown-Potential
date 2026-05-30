"""Validate the Müller-Brown potential's critical points.

The documented minima and saddle points are only rounded to three decimals, so
the force is not negligible there (the curvature reaches ~4000). Two robust
checks instead: the Hessian signature is correct at the documented point, and a
Newton refinement lands on a true stationary point a tiny distance away.
"""

import torch
from torch.autograd.functional import hessian

from src.muller_brown import MuellerBrownPotential


def _hessian_eigenvalues(potential: MuellerBrownPotential, x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.eigvalsh(hessian(lambda q: potential(q), x))


def _newton_refine(potential: MuellerBrownPotential, x: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """Newton-iterate toward the nearest stationary point of the potential."""
    for _ in range(steps):
        grad_v = -potential.force(x)
        h = hessian(lambda q: potential(q), x)
        x = x - torch.linalg.solve(h, grad_v)
    return x


def test_minima_have_positive_curvature():
    potential = MuellerBrownPotential(dtype=torch.float64)
    for point in potential.get_minima():
        x = torch.tensor(point, dtype=torch.float64)
        eigs = _hessian_eigenvalues(potential, x)
        assert (eigs > 0).all(), f"minimum {point} is not a local min: eigenvalues {eigs.tolist()}"


def test_saddles_have_one_negative_curvature():
    potential = MuellerBrownPotential(dtype=torch.float64)
    for point in potential.get_saddle_points():
        x = torch.tensor(point, dtype=torch.float64)
        eigs = _hessian_eigenvalues(potential, x)
        n_negative = int((eigs < 0).sum())
        assert n_negative == 1, f"saddle {point} has {n_negative} negative eigenvalues: {eigs.tolist()}"


def test_documented_points_are_near_true_stationary_points():
    potential = MuellerBrownPotential(dtype=torch.float64)
    for point in potential.get_minima() + potential.get_saddle_points():
        x = torch.tensor(point, dtype=torch.float64)
        refined = _newton_refine(potential, x)
        assert potential.force(refined).norm() < 1e-6, f"no stationary point found near {point}"
        assert (refined - x).norm() < 1e-2, f"documented point {point} is far from the true one {refined.tolist()}"


def test_force_matches_autograd_gradient():
    """The hand-derived analytical force must equal -∇(energy) from autograd."""
    analytical = MuellerBrownPotential(dtype=torch.float64, use_autograd=False)
    autograd = MuellerBrownPotential(dtype=torch.float64, use_autograd=True)
    coords = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [-0.5, 1.5], [0.6, 0.03]], dtype=torch.float64
    )
    assert torch.allclose(analytical.force(coords), autograd.force(coords.clone()), atol=1e-10)
