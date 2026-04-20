from .core import _SOLVER_REGISTRY

def select_solver(bc, device, implementation=None):
    key = (bc, device)
    if implementation is not None:
        key = (bc, device, implementation)
    if key not in _SOLVER_REGISTRY:
        raise ValueError(f"No solver found for boundary condition '{bc}' and device '{device}'")
    solver_cls = _SOLVER_REGISTRY[key]
    return solver_cls

