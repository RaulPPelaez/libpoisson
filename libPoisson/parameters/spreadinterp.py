from .base import BaseParameters, child_dataclass
from typing import Sequence

@child_dataclass
class SpreadInterpParameters(BaseParameters):
    """
    L: Sequence[float]
        The box size in each dimension (Lx, Ly, Lz).

    gaussian_cutoff: float
        The cutoff distance for the Gaussian spreading kernel. This determines how far the influence of each charge extends when spreading onto the grid.

    n_grid: Sequence[int]
        The number of grid points in each dimension (nx, ny, nz) for the Particle-Mesh method.
    """
    gaussian_cutoff: float
    L: Sequence[float]
    n_grid: Sequence[int]
