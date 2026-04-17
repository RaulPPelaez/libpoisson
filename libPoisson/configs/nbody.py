from .base import BaseConfig, dataclass

@dataclass
class NBodyConfig(BaseConfig):
    pass

@dataclass
class NBodySingleWallConfig(NBodyConfig):
    """
    bottom_pos: float
        Z-position of the wall. Particles are assumed to be above this wall. (i.e. z > bottom_pos)
    bottom_perm: float
        Permittivity of the wall.
    """
    bottom_pos: float
    bottom_perm: float

@dataclass
class NBodyDoubleWallConfig(NBodySingleWallConfig):
    """
    top_pos: float
        Z-position of the wall. Particles are assumed to be below this wall. (i.e. z < top_pos)
    top_perm: float
        Permittivity of the wall.
    tolerance: float, optional (default=1e-5)
        Tolerance for the iterative image generation. The image generation will stop when the maximum image charge is smaller than tolerance.
    """
    top_pos: float
    top_perm: float
    tolerance: float = 1e-5
