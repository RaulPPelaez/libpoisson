from dataclasses import dataclass
from enum import Enum

class BCType(Enum):
    OPEN = "open"
    SINGLE_WALL = "single_wall"
    DOUBLE_WALL = "double_wall"
    PERIODIC = "periodic"
    NULL = "null"

@dataclass(frozen=True)
class BoundaryConditions():
    x: BCType
    y: BCType
    z: BCType
