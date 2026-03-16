from ..solver import Solver
import cupy as cp
import numpy as np
from numpy.typing import ArrayLike
import spreadinterp
from math import pi
from ..PMSimple import PMSimple

class PMEwald(Solver):
    """
    A simple Particle-Mesh solver that uses Gaussian spreading to compute the potential and field at target positions due to source charges.
    This solver assumes periodic boundary conditions in all three dimensions. The source charges are spread onto a grid using a Gaussian kernel,
    the potential is computed in Fourier space, and then the potential and field are interpolated back to the target positions.
    Parameters
    ----------
    gaussian_cutoff : float
        The cutoff distance for the Gaussian spreading kernel. This determines how far the influence of each charge extends when spreading onto the grid.
    L : ArrayLike
        The size of the simulation box in each dimension (Lx, Ly, Lz).
    n_grid : ArrayLike
        The number of grid points in each dimension (nx, ny, nz) for the Particle-Mesh method.
    split_factor : float
        The splitting factor of the ewald sumation, which determines the gaussian width of the sum.
    """
    def __init__(self,
                 gaussian_cutoff: float,
                 L : ArrayLike,
                 n_grid: ArrayLike,
                 split_factor: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_grid = n_grid
        self.L = L
        self.gaussian_cutoff = gaussian_cutoff
        self.kernel = spreadinterp.create_kernel(type='gaussian', width=self.gaussian_width, cutoff=gaussian_cutoff)

    def _solve_periodic_periodic_periodic(self,
                source_pos: ArrayLike,
                target_pos: ArrayLike,
                charges: ArrayLike,
                compute_potential: bool = True,
                compute_field: bool = True) -> tuple[ArrayLike, ArrayLike]:
        '''
        Compute the potential and field at target positions due to source charges using a simple Particle-Mesh method with Gaussian spreading.
        This method assumes periodic boundary conditions in all three dimensions. The source charges are spread onto a grid using a Gaussian kernel,
        the potential is computed in Fourier space, and then the potential and field are interpolated back to the target positions.
        '''
        eps = self.permittivity
        eps4pi = 4 * pi * eps
        self.pot = target_pos[::3] * 0 # conserve target_pos dtype but reduce to 1D array of length M (number of target positions)
        self.field = target_pos * 0 # conserve target_pos dtype but reduce to 3D array of shape (M, 3)

        source_pos = source_pos.reshape(-1, 3)
        target_pos = target_pos.reshape(-1, 3)

        charge_grid = spreadinterp.spread(source_pos, charges, self.L, self.n_grid, kernel=self.kernel)

        charge_hat = cp.fft.fftn(charge_grid, axes=(0, 1, 2))
        kx = cp.fft.fftfreq(self.n_grid[0], self.L[0] / self.n_grid[0]) * 2 * cp.pi
        ky = cp.fft.fftfreq(self.n_grid[1], self.L[1] / self.n_grid[1]) * 2 * cp.pi
        kz = cp.fft.fftfreq(self.n_grid[2], self.L[2] / self.n_grid[2]) * 2 * cp.pi
        k_squared = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
        k_squared[0, 0, 0] = 1.0

        potential_hat = charge_hat / (eps4pi * k_squared[..., None])
        potential_hat[0, 0, 0] = 0.0
        potential_grid = cp.fft.ifftn(potential_hat, axes=(0, 1, 2)).real

        K = cp.stack(cp.meshgrid(kx, ky, kz, indexing='ij'), axis=-1)
        field_hat = 1j * K * potential_hat
        field_grid = cp.fft.ifftn(field_hat, axes=(0, 1, 2)).real

        self.pot = spreadinterp.interpolate(target_pos, potential_grid, self.L, kernel=self.kernel)
        self.field = spreadinterp.interpolate(target_pos, field_grid, self.L, kernel=self.kernel)

        return self.pot, self.field
