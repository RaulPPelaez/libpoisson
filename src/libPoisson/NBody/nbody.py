from ..solver import Solver
#import cupy as cp
#from scipy.special import erf
import numpy as np
from numba import cuda, float64
from math import sqrt, erf, exp, pi
from .utils.nbody_kernel import nbody_kernel
from numpy.typing import ArrayLike

class NBody(Solver):
    """
    N-body solver class for electrostatic interactions. Complexity is O(N*M) due to the pairwise interactions between charges.

    Uses the free charge green's function to compute the potential and field at target positions due to source charges.
    Assumes that the charges are gaussian distributed with a given width (chargeRadius) to avoid singularities at zero distance.

    Parameters:
    -----------
    z_wall: float, optional
        Position of the wall for single wall boundary condition. Needed if periodcityZ is set to 'single_wall'.
    wall_permittivity: float, optional
        Permittivity of the wall for single wall boundary condition. Needed if periodcityZ is set to 'single_wall'.
    """
    def __init__(self,
                 z_wall: float = None,
                 wall_permittivity: float = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wall_position = z_wall
        self.wall_permittivity = wall_permittivity


    def _solve_open_open_open(self,
                source_pos: ArrayLike,
                target_pos: ArrayLike,
                charges: ArrayLike,
                compute_potential: bool = True,
                compute_field: bool = True) -> tuple[ArrayLike, ArrayLike]:
        '''
        Compute the potential and field at target positions due to source charges using the free charge green's function.
        The potential and field are computed using the following formulas:

        $ \phi(\mathbf{r}) = \sum_{j=1}^N \frac{q_j}{4\pi\epsilon} \frac{\text{erf}(r_{ij}/a)}{r_{ij}} $
        $ \mathbf{E}(\mathbf{r}) = \sum_{j=1}^N \frac{q_j}{4\pi\epsilon} \left( \frac{\text{erf}(r_{ij}/a)}{r_{ij}^3} - \frac{2}{\sqrt{\pi}} \frac{\exp(-(r_{ij}/a)^2)}{a r_{ij}^2} \right) \mathbf{r}_{ij} $
        where $ r_{ij} = |\mathbf{r} - \mathbf{r}_j| $ is the distance between the target position and the source charge, and $ a $ is the width of the Gaussian charge distribution (related to the charge radius).
        '''
        eps = self.permittivity
        eps4pi = 4 * pi * eps
        a = self.charge_radius * sqrt(2/3)  # Assuming charge_radius is the mean cuadratic radius sqrt(<r^2>) and charge distribution rho=exp(-(r/a)^2) then a = charge_radius * sqrt(2/3).
        self.pot = target_pos[::3] * 0 # conserve target_pos dtype but reduce to 1D array of length M (number of target positions)
        self.field = target_pos * 0
        threads_per_block = 256 # Number of threads per block (can be tuned for performance)
        M = len(target_pos) // 3
        blocks_per_grid = (M + threads_per_block - 1) // threads_per_block # Number of blocks needed to cover all target positions
        nbody_kernel[blocks_per_grid, threads_per_block](source_pos, target_pos, charges, a, eps4pi, self.pot, self.field)

        if not compute_potential:
            self.pot = None
        if not compute_field:
            self.field = None

        return self.pot, self.field

    #def _solve_open_open_open_old(self,
    #          source_pos: cp.ndarray,
    #          target_pos: cp.ndarray,
    #          charges: cp.ndarray,
    #          compute_potential: bool = True,
    #          compute_field: bool = True) -> tuple[cp.ndarray, cp.ndarray]:

    #    pot = None
    #    field = None
    #    eps = self.permittivity
    #    pos_i = target_pos.reshape(-1, 1, 3)  # (M, 1, 3)
    #    pos_j = source_pos.reshape(1, -1, 3) # (1, N, 3)
    #    r_ij = pos_i - pos_j # (M, N, 3)
    #    r = cp.linalg.norm(r_ij, axis=-1) # (M, N)
    #    eps4pi = 4 * cp.pi * eps
    #    a = self.charge_radius * cp.sqrt(2/3)  # Assuming charge_radius is the mean cuadratic radius sqrt(<r^2>) and charge distribution rho=exp(-(r/a)^2) then a = charge_radius * sqrt(2/3).
    #    r_sigma = r/a
    #    if compute_potential:
    #        G = 1 / (eps4pi * r) * erf(r_sigma)
    #        G[r == 0] = 1 / (eps4pi * a) * 2 / cp.sqrt(cp.pi)  # Handle the singularity at r=0 by using the limit of G as r approaches 0 (the potential is finite at r=0 due to the Gaussian charge distribution)
    #        pot = G @ charges
    #    if compute_field:
    #        gradG = (1/(eps4pi * r**3) * (erf(r_sigma) - 2/cp.sqrt(cp.pi) * r_sigma * cp.exp(-r_sigma*r_sigma)))[:,:,cp.newaxis] * r_ij
    #        gradG[r == 0] = 0  # Handle the singularity at r=0 by setting the gradient to zero (the field is finite at r=0 due to the Gaussian charge distribution)
    #        field = cp.sum(gradG * charges[:, cp.newaxis], axis=1)  # (M, 3)
    #    return pot, field.flatten()

    def _solve_open_open_single_wall(self,
              source_pos: ArrayLike,
              target_pos: ArrayLike,
              charges: ArrayLike,
              compute_potential: bool = True,
              compute_field: bool = True) -> tuple[ArrayLike, ArrayLike]:
        '''
        Compute the potential and field at target positions due to source charges using the free charge green's function with a single wall boundary condition.
        The potential and field are computed using the method of images, where an image charge is placed at the mirrored position of the source charge with respect to the wall,
        and its magnitude is scaled by the reflection coefficient determined by the permittivity of the wall and the medium.
        The formulas for the potential and field are the same as in the open-open-open case, but with the addition of the contributions from the image charges.

        The image charge for a source charge q at position (x, y, z) is given by:
        $ q' = q \frac{-\epsilon_{wall} + \epsilon}{\epsilon_{wall} + \epsilon} $
        and is located at:
        $ (x', y', z') = (x, y, 2z_{wall} - z) $
        '''

        if self.wall_position is None or self.wall_permittivity is None:
            raise ValueError("Wall position and permittivity must be provided for single wall boundary condition.")

        eps = self.permittivity
        eps_wall = self.wall_permittivity
        z_wall = self.wall_position
        image_pos = np.copy(source_pos)
        image_pos[2::3] = 2*z_wall - image_pos[2::3]
        source_pos = np.concatenate((source_pos, image_pos), axis=0)
        charges = np.concatenate((charges, charges * (-eps_wall + eps) / (eps_wall + eps)), axis=0)

        return self._solve_open_open_open(source_pos, target_pos, charges, compute_potential, compute_field)

