from .core import _SOLVER_REGISTRY
from dataclasses import dataclass

@dataclass
class SolverEntry():
    solver_cls: type
    parameters_cls: type

def register(cls, key, override=False):
    if key in _SOLVER_REGISTRY:
        if not override:
            raise ValueError(f"Found two solvers for {key}, use override=True to override.")
        raise Warning(f"Overriding existing solver for {key}.")
    _SOLVER_REGISTRY[key] = SolverEntry(solver_cls=cls, parameters_cls=cls.Parameters)

def register_solver(bc, device, implementation,* ,override=False, default=False):
    """
    Decorator to register a solver class for a given boundary condition and device.
    """
    def decorator(cls):
        if default:
            key = (bc, device)
            register(cls, key, override=override)

        key = (bc, device, implementation)
        register(cls, key, override=override)

        return cls
    return decorator
