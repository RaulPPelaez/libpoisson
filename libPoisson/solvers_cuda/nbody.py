from ..base_solver import Solver
import cupy as cp
import numpy as np
from numba import cuda, float64
from math import sqrt, erf, exp, pi
from .utils.nbody_kernel import nbody_kernel
from .utils.image_charges import generate_image_charges, two_walls_convergence_criterion
from numpy.typing import ArrayLike

from ..configs.nbody import NBodyConfig, NBodySingleWallConfig, NBodyDoubleWallConfig
from ..registry.decorator import register_solver
from ..definitions.boundary_conditions import BoundaryConditions, BCtype
from ..definitions.device import DeviceType

device = DeviceType.CUDA

def nbody_solve(solver,
                 source_pos: ArrayLike,
                 target_pos: ArrayLike,
                 charges: ArrayLike,
                 compute_potential: bool = True,
                 compute_field: bool = True) -> tuple[ArrayLike, ArrayLike]:
    '''
    Compute the potential and field at target positions due to source charges using the free charge green's function.
    The potential and field are computed using the following formulas:
    '''
    assert len(source_pos.shape) == 2, "Source positions must be a 2D array of shape (N, 3)."
    assert len(target_pos.shape) == 2, "Target positions must be a 2D array of shape (M, 3)."
    assert source_pos.shape[0] == charges.shape[0], "Number of source positions must match number of charges."
    assert source_pos.shape[1] == 3, "Source positions must be 3D."
    assert target_pos.shape[1] == 3, "Target positions must be 3D."
    eps = solver.perm
    eps4pi = 4 * pi * eps
    a = solver.gaussian_width  # Assuming charge_radius is the mean cuadratic radius sqrt(<r^2>) and charge distribution rho=exp(-(r/a)^2) then a = charge_radius * sqrt(2/3).
    M = len(target_pos[:,0])
    N = len(source_pos[:,0])
    field_potential = cp.zeros((M, 4), dtype=cp.float32)
    source_pos_charge = cp.zeros((len(source_pos[:,0]), 4), dtype=cp.float32)
    source_pos_charge[:, :3] = source_pos.reshape(-1, 3).astype(cp.float32)
    source_pos_charge[:, 3] = charges.astype(cp.float32)
    target_pos = cp.asarray(target_pos, dtype=cp.float32).reshape(-1, 3)
    threads_per_block = 256
    blocks_per_grid = (M + threads_per_block - 1) // threads_per_block
    nbody_kernel[blocks_per_grid, threads_per_block](target_pos, source_pos_charge, M, N, field_potential, a)
    field_potential /= eps4pi

    if not compute_potential:
        solver.pot = None
    if not compute_field:
        solver.field = None

    return field_potential[:, 3], field_potential[:, :3]


bc = BoundaryConditions(BCtype.OPEN, BCtype.OPEN, BCtype.OPEN)
@register_solver(bc, device)
class NBody(Solver):
    """
    N-body solver class for electrostatic interactions. Complexity is O(N*M) due to the pairwise interactions between charges.

    Uses the free charge green's function to compute the potential and field at target positions due to source charges.
    Assumes that the charges are gaussian distributed with a given width (chargeRadius) to avoid singularities at zero distance.
    """
    Config = NBodyConfig
    def solve(self,
              source_pos: ArrayLike,
              target_pos: ArrayLike,
              charges: ArrayLike,
              compute_potential: bool = True,
              compute_field: bool = True) -> tuple[ArrayLike, ArrayLike]:

        return nbody_solve(self, source_pos, target_pos, charges, compute_potential, compute_field)

@register_solver(BoundaryConditions(BCtype.OPEN, BCtype.OPEN, BCtype.SINGLE_WALL), device)
class NBodySingleWall(Solver):
    """
    """
    Config = NBodySingleWallConfig
    def solve(self,
              source_pos: ArrayLike,
              target_pos: ArrayLike,
              charges: ArrayLike,
              compute_potential: bool = True,
              compute_field: bool = True) -> tuple[ArrayLike, ArrayLike]:
        r'''
        Compute the potential and field at target positions due to source charges using the free charge green's function with a single wall boundary condition.
        The potential and field are computed using the method of images, where an image charge is placed at the mirrored position of the source charge with respect to the wall,
        and its magnitude is scaled by the reflection coefficient determined by the perm of the wall and the medium.
        The formulas for the potential and field are the same as in the open-open-open case, but with the addition of the contributions from the image charges.

        The image charge for a source charge q at position (x, y, z) is given by:
        $ q' = q \frac{-\epsilon_{wall} + \epsilon}{\epsilon_{wall} + \epsilon} $
        and is located at:
        $ (x', y', z') = (x, y, 2z_{wall} - z) $
        '''

        if self.bottom_pos is None or self.floor_perm is None:
            raise ValueError("Wall position and perm must be provided for single wall boundary condition.")

        eps = self.perm
        eps_wall  = self.floor_perm
        bottom_pos    = self.bottom_pos
        image_charges, image_pos = generate_image_charges(eps, eps_wall, eps, bottom_pos, 0, charges, source_pos, n_rebounds=1)

        if self.need_complex:
            real_charges = image_charges.real
            imag_charges = image_charges.imag
            real_potential, real_field = self.nbody_solve(image_pos, target_pos, real_charges, compute_potential, compute_field)
            imag_potential, imag_field = self.nbody_solve(image_pos, target_pos, imag_charges, compute_potential, compute_field)
            return real_potential + 1j * imag_potential, real_field + 1j * imag_field

        return nbody_solve(image_pos, target_pos, image_charges, compute_potential, compute_field)

@register_solver(BoundaryConditions(BCtype.OPEN, BCtype.OPEN, BCtype.DOUBLE_WALL), device)
class NBodyDoubleWall(Solver):
    Config = NBodyDoubleWallConfig
    def solve(self,
              source_pos: ArrayLike,
              target_pos: ArrayLike,
              charges: ArrayLike,
              compute_potential: bool = True,
              compute_field: bool = True) -> tuple[ArrayLike, ArrayLike]:

        eps       = self.perm
        eps_floor = self.floor_perm
        eps_ceil  = self.ceil_perm
        bottom_pos   = self.bottom_pos
        top_pos    = self.top_pos
        n_rebounds = two_walls_convergence_criterion(eps, eps_floor, eps_ceil, tol=self.two_walls_tolerance)
        image_charges, image_pos = generate_image_charges(eps, eps_floor, eps_ceil, bottom_pos, top_pos, charges, source_pos, n_rebounds)

        if self.need_complex:
            eps_conj = eps.conjugate()
            self.perm = (eps*eps_conj).real
            real_charges = image_charges.real.astype(cp.float64)
            real_pos = image_pos.real.astype(cp.float64)
            imag_charges = image_charges.imag.astype(cp.float64)
            imag_pos = image_pos.imag.astype(cp.float64)
            target_pos = target_pos.astype(cp.float64)
            real_potential, real_field = self.nbody_solve(image_pos, target_pos, real_charges, compute_potential, compute_field)
            imag_potential, imag_field = self.nbody_solve(image_pos, target_pos, imag_charges, compute_potential, compute_field)
            self.perm = eps
            return (real_potential + 1j * imag_potential)*eps_conj, (real_field + 1j * imag_field)*eps_conj

        return self.nbody_solve(image_pos, target_pos, image_charges, compute_potential, compute_field)
