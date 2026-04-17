from ...solver import Solver
from .._uammd import uammd_pse
import cupy as cp
import numpy as np
from numpy.typing import ArrayLike

class uammdPSE(Solver):
    """
    Solver for the Poisson Equation using Poisson Spectral Solver (PSE) implemented in UAMMD.
    This solver is designed for systems with periodic boundary conditions in all three dimensions.

    Parameters:
    -----------
    Lx : float
        Length of the simulation box in the x-direction.
    Ly : float
        Length of the simulation box in the y-direction.
    Lz : float
        Length of the simulation box in the z-direction.
    tolerance : float, optional
        Tolerance for the solver convergence (default is 1e-3).
    splitting_ratio : float, optional
        Ratio for splitting the Gaussian width (default is -1.0, which means no splitting that UAMMD will choose a reasonable value based on the other parameters).
        This parameter controls the width of the Ewald splitting function used in the PSE method.
        The width of the Ewald splitting function is gaussian_width * splitting_ratio. So splitting_ratio > 1 means a
        wider splitting function, and splitting_ratio < 1 means a narrower splitting function. A narrower splitting function
        will lead to a faster convergence of the real-space sum, but a slower convergence of the reciprocal-space sum.
        A wider splitting function will lead to a slower convergence of the real-space sum,
        but a faster convergence of the reciprocal-space sum. The optimal splitting ratio depends on the system size and the desired accuracy,
        and can be determined empirically.
    """
    def __init__(self,
                 Lx: float,
                 Ly: float,
                 Lz: float,
                 tolerance: float = 1e-8,
                 splitting_ratio: float = -1.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.split = 1/(2*self.gaussian_width*splitting_ratio) if splitting_ratio > 0 else -1.0
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
        total_charges = cp.concatenate((charges, target_pos[:,0]*0), axis=0).astype(cp.float32)
        field = cp.zeros_like(total_pos).astype(cp.float32)
        potential = cp.zeros_like(total_charges).astype(cp.float32)
        assert total_pos.shape[0] == total_charges.shape[0]
        assert field.shape == total_pos.shape
        assert potential.shape == total_charges.shape
        assert total_pos.shape[1] == 3
        self.uammd_solver.compute_poisson(positions=total_pos, charges=total_charges, fields=field, potentials=potential)
        return potential[len(charges):], field[len(charges):,:]
