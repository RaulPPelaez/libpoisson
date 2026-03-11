from ..solver import Solver
import cupy as cp

class PSE(Solver):
    """
    Positive Split Edwald
    """
    def __init__(self,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _solve_periodic_periodic_periodic(self,
              source_pos: cp.ndarray,
              target_pos: cp.ndarray,
              charges: cp.ndarray,
              compute_potential: bool = True,
              compute_field: bool = True) -> tuple[cp.ndarray, cp.ndarray]:

        raise NotImplementedError("Periodic boundary conditions are not implemented for the N-body solver. Please use a different solver or implement the periodic Green's function for electrostatics.")
