"""Müller-Brown potential energy surface implementation."""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# This is a static function, independent of any class instance.
# The JIT compiler can easily understand its inputs and operations.
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
    
    exponents = (a * dx**2 + b * dx * dy + c * dy**2)
    
    potential = torch.sum(A * torch.exp(exponents), dim=-1)
        
    return potential.view(original_shape)


class MuellerBrownPotential(nn.Module):
    """
    Müller-Brown potential energy surface.
    """

    def __init__(self, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64):
        super().__init__()
        
        # We use register_buffer so that the tensors are moved to the correct device
        # (e.g., .to('cuda')) when you move the module.
        self.register_buffer("A", torch.tensor([-200.0, -100.0, -170.0, 15.0], device=device, dtype=dtype))
        self.register_buffer("a", torch.tensor([-1.0, -1.0, -6.5, 0.7], device=device, dtype=dtype))
        self.register_buffer("b", torch.tensor([0.0, 0.0, 11.0, 0.6], device=device, dtype=dtype))
        self.register_buffer("c", torch.tensor([-10.0, -10.0, -6.5, 0.7], device=device, dtype=dtype))
        self.register_buffer("x_centers", torch.tensor([1.0, 0.0, -0.5, -1.0], device=device, dtype=dtype))
        self.register_buffer("y_centers", torch.tensor([0.0, 0.5, 1.5, 1.0], device=device, dtype=dtype))
        
    @property
    def ndims(self) -> int:
        return 2

    def forward(self, coordinates: Tensor) -> Tensor:
        """
        Compute the potential energy by calling the JIT-scripted static function.
        """
        # The forward method itself is not scripted. It just calls the scripted helper.
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
        """
        Compute forces (negative gradient) at given coordinates.
        """
        # Ensure requires_grad is set on a new tensor to avoid in-place modification errors.
        coords_with_grad = coordinates.detach().requires_grad_(True)
        potential = self(coords_with_grad)
        
        grad_outputs = torch.ones_like(potential)
        gradients = torch.autograd.grad(
            outputs=potential,
            inputs=coords_with_grad,
            grad_outputs=grad_outputs,
            create_graph=False, # Set to False as we don't need higher-order derivatives
            retain_graph=False,
        )[0]

        return -gradients
    
    def get_minima(self) -> List[Tuple[float, float]]:
        """Return the approximate locations of the three minima."""
        return [(-0.558, 1.442), (0.623, 0.028), (-0.050, 0.467)]
    
    def get_saddle_points(self) -> List[Tuple[float, float]]:
        """Return the approximate locations of the two saddle points."""
        return [(-0.822, 0.624), (0.212, 0.293)]