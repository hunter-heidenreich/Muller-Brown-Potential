"""Modern PyTorch implementation of the Müller-Brown potential."""

from .potential import MuellerBrownPotential
from .simulation import LangevinSimulator
from .visualization import PotentialVisualizer

__version__ = "0.1.0"
__all__ = ["MuellerBrownPotential", "LangevinSimulator", "PotentialVisualizer"]
