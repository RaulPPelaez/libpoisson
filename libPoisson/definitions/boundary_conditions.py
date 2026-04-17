from dataclasses import dataclass
from enum import Enum

class BCtype(Enum):
    OPEN = "open"
    SINGLE_WALL = "single_wall"
    DOUBLE_WALL = "double_wall"
    PERIODIC = "periodic"
    NULL = "null"

@dataclass(frozen=True)
class BoundaryConditions():
    x: BCtype
    y: BCtype
    z: BCtype
