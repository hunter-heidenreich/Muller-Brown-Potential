"""Constants and core parameters for the Müller-Brown potential system."""

import torch

# Constants for Müller-Brown potential
MULLER_BROWN_MINIMA = [(-0.558, 1.442), (0.623, 0.028), (-0.050, 0.467)]

# Default simulation ranges
DEFAULT_X_RANGE = (-1.5, 1.2)
DEFAULT_Y_RANGE = (-0.2, 2.0)

# Default tensor configuration
DEFAULT_DEVICE = "cpu"
DEFAULT_DTYPE = torch.float64
