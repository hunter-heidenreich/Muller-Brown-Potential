"""Müller-Brown potential energy surface implementation."""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class MuellerBrownPotential(nn.Module):
    """
    Müller-Brown potential energy surface.
    
    Reference:
    Müller, K., & Brown, L. D. (1979). Location of saddle points and minimum energy paths
    by a constrained simplex optimization procedure. Theoretica Chimica Acta, 53, 75-93.
    """

    def __init__(self, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64):
        super().__init__()
        
        # Müller-Brown potential parameters
        self._register_parameters(device, dtype)
        
    def _register_parameters(self, device: str | torch.device, dtype: torch.dtype) -> None:
        """Register the Müller-Brown potential parameters as non-trainable buffers."""
        A = torch.tensor([-200.0, -100.0, -170.0, 15.0], device=device, dtype=dtype)
        a = torch.tensor([-1.0, -1.0, -6.5, 0.7], device=device, dtype=dtype)
        b = torch.tensor([0.0, 0.0, 11.0, 0.6], device=device, dtype=dtype)
        c = torch.tensor([-10.0, -10.0, -6.5, 0.7], device=device, dtype=dtype)
        
        # Centers of the Gaussian terms
        x_centers = torch.tensor([1.0, 0.0, -0.5, -1.0], device=device, dtype=dtype)
        y_centers = torch.tensor([0.0, 0.5, 1.5, 1.0], device=device, dtype=dtype)
        
        self.register_buffer("A", A)
        self.register_buffer("a", a)
        self.register_buffer("b", b)
        self.register_buffer("c", c)
        self.register_buffer("x_centers", x_centers)
        self.register_buffer("y_centers", y_centers)
        
    @property
    def ndims(self) -> int:
        """Number of spatial dimensions."""
        return 2
        
    def forward(self, coordinates: Tensor) -> Tensor:
        """
        Compute the potential energy at given coordinates.
        Vectorized implementation for better performance.
        
        Args:
            coordinates: Tensor of shape (..., 2) with x, y coordinates
            
        Returns:
            Potential energy values of shape (...)
        """
        # Ensure coordinates are at least 2D for indexing
        original_shape = coordinates.shape[:-1]
        coords = coordinates.view(-1, 2)
        
        x, y = coords[:, 0], coords[:, 1]  # Shape: (N,)
        
        # Vectorized computation over all 4 Gaussian terms
        # Broadcast coordinates against all centers simultaneously
        dx = x.unsqueeze(-1) - self.x_centers  # Shape: (N, 4)
        dy = y.unsqueeze(-1) - self.y_centers  # Shape: (N, 4)
        
        # Compute exponents for all terms at once
        exponents = (
            self.a * dx**2 +
            self.b * dx * dy +
            self.c * dy**2
        )  # Shape: (N, 4)
        
        # Sum over all Gaussian terms
        potential = torch.sum(self.A * torch.exp(exponents), dim=-1)  # Shape: (N,)
            
        return potential.view(original_shape)
    
    def force(self, coordinates: Tensor) -> Tensor:
        """
        Compute forces (negative gradient) at given coordinates.
        Optimized for batch operations.
        
        Args:
            coordinates: Tensor of shape (..., 2) requiring gradients
            
        Returns:
            Forces of shape (..., 2)
        """
        # Ensure gradients are enabled
        needs_grad = coordinates.requires_grad
        
        if not needs_grad:
            coordinates = coordinates.detach().requires_grad_(True)
            
        potential = self(coordinates)
        
        # Use more efficient gradient computation
        # For single points, no need to sum
        if potential.numel() == 1:
            potential.backward()
        else:
            # For batches, use torch.autograd.grad for better memory efficiency
            grad_outputs = torch.ones_like(potential)
            gradients = torch.autograd.grad(
                outputs=potential,
                inputs=coordinates,
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]
            
            # Force is negative gradient
            return -gradients
        
        # For single point case
        forces = -coordinates.grad.clone()
        
        # Clean up gradients if we created them
        if not needs_grad:
            coordinates.grad = None
        else:
            coordinates.grad.zero_()
        
        return forces
    
    def get_minima(self) -> List[Tuple[float, float]]:
        """Return the approximate locations of the three minima."""
        return [
            (-0.558, 1.442),  # Minimum A
            (0.623, 0.028),   # Minimum B  
            (-0.050, 0.467),  # Minimum C
        ]
    
    def get_saddle_points(self) -> List[Tuple[float, float]]:
        """Return the approximate locations of the two saddle points."""
        return [
            (-0.822, 0.624),  # Saddle 1
            (0.212, 0.293),   # Saddle 2
        ]
