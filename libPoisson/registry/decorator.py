from .core import _SOLVER_REGISTRY
from dataclasses import dataclass

@dataclass
class SolverEntry():
    solver_cls: type
    config_cls: type

def register_solver(bc, device, override=False):
    """
    Decorator to register a solver class for a given boundary condition and device.
    """
    def decorator(cls):
        key = (bc, device)

        if key in _SOLVER_REGISTRY:
            if not override:
                raise ValueError(f"Found two solvers for {key}, use override=True to override.")
            raise Warning(f"Overriding existing solver for {key}.")
        _SOLVER_REGISTRY[key] = SolverEntry(solver_cls=cls, config_cls=cls.Config)
        return cls
    return decorator
