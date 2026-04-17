from ...solver import Solver
from .._uammd import uammd_dpslab
import cupy as cp
import numpy as np
from numpy.typing import ArrayLike

class uammdDPSlab(Solver):
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
    top_permittivity : float, optional
        Permittivity of the top medium (default is None, but needed for double_wall geometry).
    bottom_permittivity : float, optional
        Permittivity of the bottom medium (default is None, but needed for double_wall and single_wall geometry).
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
                 permittivity_top = None,
                 permittivity_bottom = None,
                 tolerance: float = 1e-8,
                 splitting_ratio: float = -1.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.split = 1/(2*self.gaussian_width*splitting_ratio) if splitting_ratio > 0 else -1.0
        self.tolerance = tolerance
        self.permittivity_top = permittivity_top
        self.permittivity_bottom = permittivity_bottom
        self.top_defined = True
        self.bottom_defined = True
        if self.permittivity_top is None:
            self.permittivity_top = self.permittivity
            self.top_defined = False
        if self.permittivity_bottom is None:
            self.permittivity_bottom = self.permittivity
            self.bottom_defined = False
        self.uammd_solver = uammd_dpslab(Lx=Lx, Ly=Ly, Lz=Lz,
                                         permittivity_inside=self.permittivity, permittivity_top=self.permittivity_top, permittivity_bottom=self.permittivity_bottom,
                                         gaussian_width=self.gaussian_width, tolerance=self.tolerance, split=self.split)


    def _solve_periodic_periodic_open(self,
                source_pos: ArrayLike,
                target_pos: ArrayLike,
                charges: ArrayLike,
                compute_potential: bool = True,
                compute_field: bool = True) -> tuple[ArrayLike, ArrayLike]:
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

    def _solve_periodic_periodic_single_wall(self,
                source_pos: ArrayLike,
                target_pos: ArrayLike,
                charges: ArrayLike,
                compute_potential: bool = True,
                compute_field: bool = True) -> tuple[ArrayLike, ArrayLike]:
        if self.bottom_defined:
            return self._solve_periodic_periodic_open(source_pos, target_pos, charges, compute_potential, compute_field)
        raise ValueError("Bottom permittivity must be defined for single wall geometry.")

    def _solve_periodic_periodic_two_walls(self,
                source_pos: ArrayLike,
                target_pos: ArrayLike,
                charges: ArrayLike,
                compute_potential: bool = True,
                compute_field: bool = True) -> tuple[ArrayLike, ArrayLike]:
        if self.bottom_defined and self.top_defined:
            return self._solve_periodic_periodic_open(source_pos, target_pos, charges, compute_potential, compute_field)
        raise ValueError("Both top and bottom permittivity must be defined for double wall geometry.")
