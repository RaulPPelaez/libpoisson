from ...solver import Solver
from ..._uammd import uammd_pse
import cupy as cp
import numpy as np
from numpy.typing import ArrayLike

class uammdPSE(Solver):
    """
    Solver for the Poisson Equation using Poisson Spectral Solver (PSE) implemented in UAMMD.
    This solver is designed for systems with periodic boundary conditions in all three dimensions.

    Parameters:
    -----------
    lbox: float
        The length of the cubic simulation box. The solver assumes periodic boundary conditions in all three dimensions.
    """
    def __init__(self,
                 Lx: float,
                 Ly: float,
                 Lz: float,
                 tolerance: float = 1e-3,
                 split: float = -1.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.split = split
        self.tolerance = tolerance
        self.uammd_solver = uammd_pse(Lx=Lx, Ly=Ly, Lz=Lz, permittivity=self.permittivity, gaussian_width=self.gaussian_width, tolerance=self.tolerance, split=self.split)


    def _solve_periodic_periodic_periodic(self,
                source_pos: ArrayLike,
                target_pos: ArrayLike,
                charges: ArrayLike,
                compute_potential: bool = True,
                compute_field: bool = True) -> tuple[ArrayLike, ArrayLike]:
        eps = self.permittivity
        total_pos = cp.concatenate((source_pos, target_pos), axis=0).reshape(-1, 3).astype(cp.float32)
        total_charges = cp.concatenate((charges, cp.zeros(len(target_pos)//3)), axis=0).astype(cp.float32)
        field = cp.zeros_like(total_pos).astype(cp.float32)
        potential = cp.zeros_like(total_charges).astype(cp.float32)
        assert total_pos.shape[0] == total_charges.shape[0]
        assert field.shape == total_pos.shape
        assert potential.shape == total_charges.shape
        assert total_pos.shape[1] == 3
        self.uammd_solver.compute_poisson(positions=total_pos, charges=total_charges, forces=field, energy=potential)
        print(total_charges.shape, field.shape, potential.shape)
        cpu_potential = cp.asnumpy(potential)
        cpu_field = cp.asnumpy(field)
        return cpu_potential[len(charges):], cpu_field[len(charges):]
