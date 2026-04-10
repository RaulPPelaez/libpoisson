from ...solver import Solver
import cupy as cp
#from scipy.special import erf
import numpy as np
from numba import cuda, float64
from math import sqrt, erf, exp, pi
from .utils.nbody_kernel import nbody_kernel
from .utils.image_charges import generate_image_charges, two_walls_convergence_criterion
from numpy.typing import ArrayLike

class NBody(Solver):
    """
    N-body solver class for electrostatic interactions. Complexity is O(N*M) due to the pairwise interactions between charges.

    Uses the free charge green's function to compute the potential and field at target positions due to source charges.
    Assumes that the charges are gaussian distributed with a given width (chargeRadius) to avoid singularities at zero distance.

    Parameters:
    -----------
    floor_z: float, optional
        Position of the floor wall for single wall or two walls boundary condition. Needed if periodcityZ is set to 'single_wall' or 'two_walls'.
    floor_permittivity: float, optional
        Permittivity of the floor wall for single wall or two walls boundary condition. Needed if periodcityZ is set to 'single_wall' or 'two_walls'.
    ceil_z: float, optional
        Position of the ceiling wall for two walls boundary condition. Needed if periodcityZ is set to 'two_walls'.
    ceil_permittivity: float, optional
        Permittivity of the ceiling wall for two walls boundary condition. Needed if periodcityZ is set to 'two_walls'.
    two_walls_tolerance: float, optional
        Tolerance for the convergence of the image charge method in the two walls boundary condition. Default is 1e-3.
    """
    def __init__(self,
                 floor_z: float = None,
                 ceil_z: float = None,
                 floor_permittivity: float = None,
                 ceil_permittivity: float = None,
                 two_walls_tolerance: float = 1e-3,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.periodicityZ == 'single_wall' and (floor_z is None or floor_permittivity is None):
            raise ValueError("For single wall boundary condition, floor_z and floor_permittivity must be provided.")
        if self.periodicityZ == 'two_walls' and (floor_z is None or ceil_z is None or floor_permittivity is None or ceil_permittivity is None):
            raise ValueError("For two walls boundary condition, floor_z, ceil_z, floor_permittivity and ceil_permittivity must be provided.")
        self.floor_z = floor_z
        self.floor_permittivity = floor_permittivity
        self.ceil_z = ceil_z
        self.ceil_permittivity = ceil_permittivity
        self.two_walls_tolerance = two_walls_tolerance

    def _solve_open_open_open(self,
                source_pos: ArrayLike,
                target_pos: ArrayLike,
                charges: ArrayLike,
                compute_potential: bool = True,
                compute_field: bool = True) -> tuple[ArrayLike, ArrayLike]:
        r'''
        Compute the potential and field at target positions due to source charges using the free charge green's function.
        The potential and field are computed using the following formulas:

        $ \phi(\mathbf{r}) = \sum_{j=1}^N \frac{q_j}{4\pi\epsilon} \frac{\text{erf}(r_{ij}/a)}{r_{ij}} $
        $ \mathbf{E}(\mathbf{r}) = \sum_{j=1}^N \frac{q_j}{4\pi\epsilon} \left( \frac{\text{erf}(r_{ij}/a)}{r_{ij}^3} - \frac{2}{\sqrt{\pi}} \frac{\exp(-(r_{ij}/a)^2)}{a r_{ij}^2} \right) \mathbf{r}_{ij} $
        where $ r_{ij} = |\mathbf{r} - \mathbf{r}_j| $ is the distance between the target position and the source charge, and $ a $ is the width of the Gaussian charge distribution (related to the charge radius).
        '''
        assert len(source_pos.shape) == 2, "Source positions must be a 2D array of shape (N, 3)."
        assert len(target_pos.shape) == 2, "Target positions must be a 2D array of shape (M, 3)."
        assert source_pos.shape[0] == charges.shape[0], "Number of source positions must match number of charges."
        assert source_pos.shape[1] == 3, "Source positions must be 3D."
        assert target_pos.shape[1] == 3, "Target positions must be 3D."
        eps = self.permittivity
        eps4pi = 4 * pi * eps
        a = self.gaussian_width  # Assuming charge_radius is the mean cuadratic radius sqrt(<r^2>) and charge distribution rho=exp(-(r/a)^2) then a = charge_radius * sqrt(2/3).
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
            self.pot = None
        if not compute_field:
            self.field = None

        return field_potential[:, 3], field_potential[:, :3]

    def _solve_open_open_single_wall(self,
              source_pos: ArrayLike,
              target_pos: ArrayLike,
              charges: ArrayLike,
              compute_potential: bool = True,
              compute_field: bool = True) -> tuple[ArrayLike, ArrayLike]:
        r'''
        Compute the potential and field at target positions due to source charges using the free charge green's function with a single wall boundary condition.
        The potential and field are computed using the method of images, where an image charge is placed at the mirrored position of the source charge with respect to the wall,
        and its magnitude is scaled by the reflection coefficient determined by the permittivity of the wall and the medium.
        The formulas for the potential and field are the same as in the open-open-open case, but with the addition of the contributions from the image charges.

        The image charge for a source charge q at position (x, y, z) is given by:
        $ q' = q \frac{-\epsilon_{wall} + \epsilon}{\epsilon_{wall} + \epsilon} $
        and is located at:
        $ (x', y', z') = (x, y, 2z_{wall} - z) $
        '''

        if self.floor_z is None or self.floor_permittivity is None:
            raise ValueError("Wall position and permittivity must be provided for single wall boundary condition.")

        eps = self.permittivity
        eps_wall  = self.floor_permittivity
        floor_z    = self.floor_z
        image_charges, image_pos = generate_image_charges(eps, eps_wall, eps, floor_z, 0, charges, source_pos, n_rebounds=1)
        if self.need_complex:
            real_charges = image_charges.real
            imag_charges = image_charges.imag
            real_potential, real_field = self._solve_open_open_open(image_pos, target_pos, real_charges, compute_potential, compute_field)
            imag_potential, imag_field = self._solve_open_open_open(image_pos, target_pos, imag_charges, compute_potential, compute_field)
            return real_potential + 1j * imag_potential, real_field + 1j * imag_field
        return self._solve_open_open_open(image_pos, target_pos, image_charges, compute_potential, compute_field)

    def _solve_open_open_two_walls(self,
              source_pos: ArrayLike,
              target_pos: ArrayLike,
              charges: ArrayLike,
              compute_potential: bool = True,
              compute_field: bool = True) -> tuple[ArrayLike, ArrayLike]:
        eps       = self.permittivity
        eps_floor = self.floor_permittivity
        eps_ceil  = self.ceil_permittivity
        z_floor   = self.floor_z
        z_ceil    = self.ceil_z
        n_rebounds = two_walls_convergence_criterion(eps, eps_floor, eps_ceil, tol=self.two_walls_tolerance)
        image_charges, image_pos = generate_image_charges(eps, eps_floor, eps_ceil, z_floor, z_ceil, charges, source_pos, n_rebounds)
        if self.need_complex:
            eps_conj = eps.conjugate()
            self.permittivity = (eps*eps_conj).real
            real_charges = image_charges.real.astype(cp.float64)
            real_pos = image_pos.real.astype(cp.float64)
            imag_charges = image_charges.imag.astype(cp.float64)
            imag_pos = image_pos.imag.astype(cp.float64)
            target_pos = target_pos.astype(cp.float64)
            real_potential, real_field = self._solve_open_open_open(image_pos, target_pos, real_charges, compute_potential, compute_field)
            imag_potential, imag_field = self._solve_open_open_open(image_pos, target_pos, imag_charges, compute_potential, compute_field)
            self.permittivity = eps
            return (real_potential + 1j * imag_potential)*eps_conj, (real_field + 1j * imag_field)*eps_conj
        return self._solve_open_open_open(image_pos, target_pos, image_charges, compute_potential, compute_field)
