from .base import BaseParameters, child_dataclass
from typing import Sequence

@child_dataclass
class EwaldSumParameters(BaseParameters):
    """
    L: Sequence[float]
        The box size in each dimension.

    splitting_ratio: float, optional (default=-1.0)
        The ratio between the original gaussian source and the splitting gaussian source.
        If negative, it will be determined by the Ewald sum method.
        spliiting_ratio = 1.0 means both gaussian sources are the same.
        splitting_ratio < 1.0 means the splitting gaussian source is narrower which leads to faster convergence in the real space but slower convergence in the reciprocal space.
        splitting_ratio > 1.0 means the splitting gaussian source is wider which leads to slower convergence in the real space but faster convergence in the reciprocal space.

    tolerance: float, optional (default=1e-6)
        The tolerance for the Ewald sum. The smaller the tolerance, the more accurate the result but also the more computational cost.
    """
    L: Sequence[float]
    splitting_ratio: float = -1.0
    tolerance: float = 1e-6

@child_dataclass
class EwaldSumSingleWallParameters(EwaldSumParameters):
    """
    bottom_permittivity: float
        Permittivity of the bottom wall (z<-L[2]/2). Particles are confined in the region z>-L[2]/2.
    """
    bottom_permittivity: float

@child_dataclass
class EwaldSumDoubleWallParameters(EwaldSumSingleWallParameters):
    """
    top_permittivity: float
        Permittivity of the top wall (z>L[2]/2). Particles are confined in the region -L[2]/2<z<L[2]/2.
    """
    top_permittivity: float
