from ._libpoisson import PoissonSolver as _PoissonSolver
import cupy as cp


class PoissonSolver:

    def __init__(self, *args, **kwargs):
        self.solver = _PoissonSolver(*args, **kwargs)

    def compute_poisson(self, positions, charges):
        forces = cp.zeros_like(positions).astype(cp.float32)
        self.solver.compute_poisson(positions=positions, charges=charges, forces=forces)
        return forces
