from ..definitions.boundary_conditions import BoundaryConditions
from ..definitions.device import DeviceType
from typing import Tuple
from ..base_solver import Solver
from ..registry.core import _SOLVER_REGISTRY
from ..registry.selector import select_solver

def get_solver(bc: BoundaryCondtions | Tuple[str, str, str],
               device: (DeviceType | str) = DeviceType.CUDA,
               **kwargs) -> Solver:
    """
    Get a solver instance based on the specified boundary conditions and device.
    """
    SolverCls = select_solver(bc, device)

    config = SolverCls.config_cls(**kwargs)

    return SolverCls(config)
