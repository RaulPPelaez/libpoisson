from .core import _SOLVER_REGISTRY

def select_solver(bc, device):
    key = (bc, device)
    if key not in _SOLVER_REGISTRY:
        raise ValueError(f"No solver found for boundary condition '{bc}' and device '{device}'")
    solver_cls = _SOLVER_REGISTRY[key]
    return solver_cls

