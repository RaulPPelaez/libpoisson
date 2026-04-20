from ..definitions.boundary_conditions import BoundaryConditions, BCType
from ..definitions.device import DeviceType
from typing import Tuple
from ..base_solver import Solver
from ..registry.selector import select_solver

def get_solver(boundary_conditions: BoundaryCondtions | Tuple[str, str, str],
               device: (DeviceType | str) = DeviceType.CUDA,
               *, force_implementation: (str | None) = None,
               **kwargs) -> Solver:
    """
    Get a solver instance based on the specified boundary conditions and device.
    """
    if isinstance(boundary_conditions, tuple):
        bcx, bcy, bcz = boundary_conditions
        boundary_conditions = BoundaryConditions(BCType(bcx), BCType(bcy), BCType(bcz))
    if isinstance(device, str):
        device = DeviceType(device)
    if force_implementation is not None:
        SolverEntry = select_solver(boundary_conditions, device, implementation=force_implementation)
    else:
        SolverEntry = select_solver(boundary_conditions, device)

    try:
        parameters = SolverEntry.parameters_cls(**kwargs)
    except TypeError as e:
        if force_implementation is not None:
            raise ValueError(f"Error initializing parameters for solver ({boundary_conditions.x.value}, {boundary_conditions.y.value}, {boundary_conditions.z.value}) on device {device.value} with implementation {force_implementation}: {e}")
        raise ValueError(f"Error initializing parameters for solver ({boundary_conditions.x.value}, {boundary_conditions.y.value}, {boundary_conditions.z.value}) on device {device.value}: {e}")
    solver_cls = SolverEntry.solver_cls
    return solver_cls(parameters)
