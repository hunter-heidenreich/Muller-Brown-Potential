"""Müller-Brown potential energy surface implementation."""

import torch
import torch.nn as nn
from torch import Tensor


@torch.jit.script
def _calculate_potential(
    coordinates: Tensor,
    A: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    x_centers: Tensor,
    y_centers: Tensor,
) -> Tensor:
    """JIT-scripted function to compute the potential energy."""
    original_shape = coordinates.shape[:-1]
    coords = coordinates.view(-1, 2)

    x, y = coords[:, 0], coords[:, 1]

    dx = x.unsqueeze(-1) - x_centers
    dy = y.unsqueeze(-1) - y_centers

    # Compute exponents and potential in one expression to reduce memory
    potential = torch.sum(A * torch.exp(a * dx**2 + b * dx * dy + c * dy**2), dim=-1)

    return potential.view(original_shape)


@torch.jit.script
def _calculate_force(
    coordinates: Tensor,
    A: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    x_centers: Tensor,
    y_centers: Tensor,
) -> Tensor:
    """JIT-scripted function to compute forces (negative gradient)."""
    original_shape = coordinates.shape[:-1]
    coords = coordinates.view(-1, 2)

    x, y = coords[:, 0], coords[:, 1]

    dx = x.unsqueeze(-1) - x_centers  # (N, 4)
    dy = y.unsqueeze(-1) - y_centers  # (N, 4)

    # Compute exponential terms directly, avoiding intermediate storage
    exp_terms = torch.exp(a * dx**2 + b * dx * dy + c * dy**2)  # (N, 4)

    # Compute both gradients simultaneously using vectorized operations
    A_exp = A * exp_terms  # (N, 4)
    grad_x = torch.sum(A_exp * (2 * a * dx + b * dy), dim=-1)  # (N,)
    grad_y = torch.sum(A_exp * (b * dx + 2 * c * dy), dim=-1)  # (N,)

    # Forces are negative gradients - compute directly in final tensor
    forces = torch.stack([-grad_x, -grad_y], dim=-1)  # (N, 2)

    # Reshape back to original batch dimensions
    new_shape = list(original_shape) + [2]
    return forces.view(new_shape)


class MuellerBrownPotential(nn.Module):
    """
    Müller-Brown potential energy surface with JIT-compiled force calculations.

    V(x,y) = ∑ᵢ Aᵢ exp(aᵢ(x-x₀ᵢ)² + bᵢ(x-x₀ᵢ)(y-y₀ᵢ) + cᵢ(y-y₀ᵢ)²)
    """

    def __init__(
        self, 
        device: str | torch.device = "cpu", 
        dtype: torch.dtype = torch.float64,
        use_autograd: bool = False
    ):
        super().__init__()
        
        self.use_autograd = use_autograd

        self.register_buffer(
            "A",
            torch.tensor([-200.0, -100.0, -170.0, 15.0], device=device, dtype=dtype),
        )
        self.register_buffer(
            "a", torch.tensor([-1.0, -1.0, -6.5, 0.7], device=device, dtype=dtype)
        )
        self.register_buffer(
            "b", torch.tensor([0.0, 0.0, 11.0, 0.6], device=device, dtype=dtype)
        )
        self.register_buffer(
            "c", torch.tensor([-10.0, -10.0, -6.5, 0.7], device=device, dtype=dtype)
        )
        self.register_buffer(
            "x_centers",
            torch.tensor([1.0, 0.0, -0.5, -1.0], device=device, dtype=dtype),
        )
        self.register_buffer(
            "y_centers", torch.tensor([0.0, 0.5, 1.5, 1.0], device=device, dtype=dtype)
        )

    @property
    def ndims(self) -> int:
        return 2

    def forward(self, coordinates: Tensor) -> Tensor:
        """Compute potential energy at given coordinates."""
        return _calculate_potential(
            coordinates,
            self.A,
            self.a,
            self.b,
            self.c,
            self.x_centers,
            self.y_centers,
        )

    def force(self, coordinates: Tensor) -> Tensor:
        """Compute forces (negative gradient) at given coordinates."""
        if self.use_autograd:
            coordinates = coordinates.requires_grad_(True)
            potential = self.forward(coordinates)
            grad = torch.autograd.grad(
                potential.sum(), coordinates, create_graph=True
            )[0]
            return -grad
        else:
            return _calculate_force(
                coordinates,
                self.A,
                self.a,
                self.b,
                self.c,
                self.x_centers,
                self.y_centers,
            )

    def get_minima(self) -> list[tuple[float, float]]:
        """Return the approximate locations of the three minima."""
        return [(-0.558, 1.442), (0.623, 0.028), (-0.050, 0.467)]

    def get_saddle_points(self) -> list[tuple[float, float]]:
        """Return the approximate locations of the two saddle points."""
        return [(-0.822, 0.624), (0.212, 0.293)]

    def get_parameters(self) -> dict:
        """Return the Müller-Brown potential parameters."""
        return {
            "A": self.A.detach().cpu().numpy(),
            "a": self.a.detach().cpu().numpy(),
            "b": self.b.detach().cpu().numpy(),
            "c": self.c.detach().cpu().numpy(),
            "x_centers": self.x_centers.detach().cpu().numpy(),
            "y_centers": self.y_centers.detach().cpu().numpy(),
        }
