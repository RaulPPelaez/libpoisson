from .base import BaseParameters, dataclass

@dataclass
class NBodyParameters(BaseParameters):
    pass

@dataclass
class NBodySingleWallParameters(NBodyParameters):
    """
    bottom_wall_position: float
        Z-position of the wall. Particles are assumed to be above this wall. (i.e. z > bottom_wall_position)
    bottom_permittivity: float
        Permittivity of the wall.
    """
    bottom_wall_position: float
    bottom_permittivity: float

@dataclass
class NBodyDoubleWallParameters(NBodySingleWallParameters):
    """
    top_wall_position: float
        Z-position of the wall. Particles are assumed to be below this wall. (i.e. z < top_wall_position)
    top_permittivity: float
        Permittivity of the wall.
    tolerance: float, optional (default=1e-5)
        Tolerance for the iterative image generation. The image generation will stop when the maximum image charge is smaller than tolerance.
    """
    top_wall_position: float
    top_permittivity: float
    tolerance: float = 1e-5
