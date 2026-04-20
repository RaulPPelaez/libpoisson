from ..base_solver import Solver
from ..parameters.spreadinterp import SpreadInterpParameters
from ..registry.decorator import register_solver
from ..definitions.boundary_conditions import BoundaryConditions, BCType
from ..definitions.device import DeviceType

import cupy as cp
import numpy as np
from numpy.typing import ArrayLike
import spreadinterp
from math import pi


implementation = "spreadinterp"
device = DeviceType.CUDA
bc = BoundaryConditions(BCType.PERIODIC, BCType.PERIODIC, BCType.PERIODIC)

@register_solver(bc, device, implementation)
class spread_interp(Solver):
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
    """
    Parameters = SpreadInterpParameters
    def __init__(self, parameters: SpreadInterpParameters):
        super().__init__(parameters)
        self.kernel = spreadinterp.create_kernel(type='gaussian', width=self.gaussian_width, cutoff=self.gaussian_cutoff)

    def solve(self,
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
        pot = target_pos[:,0] * 0
        field = target_pos * 0

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

        pot = spreadinterp.interpolate(target_pos, potential_grid, self.L, kernel=self.kernel)
        field = spreadinterp.interpolate(target_pos, field_grid, self.L, kernel=self.kernel)

        return pot, field
