from ..solver import Solver
import cupy as cp
import numpy as np
from numpy.typing import ArrayLike
from math import pi

class AnalyticFourier(Solver):
    """
    AnalyticFourier is a solver that computes the potential and field
    using an analytic Fourier transform of the charge distribution.
    It assumes periodic boundary conditions in all dimensions.

    Parameters
    ----------
    L : ArrayLike
        The size of the simulation box in each dimension (Lx, Ly, Lz).
    tol : float, optional
        The tolerance for truncating the Fourier series. The default is 1e-6.
    """
    def __init__(self,
                 L : ArrayLike,
                 tol: float = 1e-6,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.L = np.array(L)
        k_max = np.sqrt(-2*np.log(tol)) / self.gaussian_width
        n_max = np.ceil(k_max * L / (2 * pi)).astype(int)
        kx = np.linspace(0, 2 * pi * n_max[0] / L[0], n_max[0])
        ky = np.linspace(0, 2 * pi * n_max[1] / L[1], n_max[1])
        kz = np.linspace(0, 2 * pi * n_max[2] / L[2], n_max[2])
        self.K = cp.array(np.meshgrid(kx, ky, kz, indexing='ij')).transpose(1, 2, 3, 0) # shape (Nx, Ny, Nz, 3)

    def _solve_periodic_periodic_periodic(self,
                source_pos: ArrayLike,
                target_pos: ArrayLike,
                charges: ArrayLike,
                compute_potential: bool = True,
                compute_field: bool = True) -> tuple[ArrayLike, ArrayLike]:
        '''
        Compute the potential and field at target positions due to source charges
        using an analytic Fourier transform of the charge distribution,
        assuming the distribution is a gaussian with width self.gaussian_width. The Fourier series is truncated
        at k_max, which is determined by the tolerance self.tol and the gaussian width.
        '''
        eps = self.permittivity
        eps4pi = 4 * pi * eps
        target_pos = target_pos.reshape(-1, 3) # shape (M, 3)
        source_pos = source_pos.reshape(-1, 3) # shape (M, 3)
        kr = np.sum(self.K[:,:,:,None,:] * source_pos[None, None, None, :, :], axis=-1) # shape (Nx, Ny, Nz, M, 3) - shape (M, 3) -> shape (Nx, Ny, Nz, M)
        K_squared = cp.sum(self.K**2, axis=-1) # shape (Nx, Ny, Nz)
        exp_ikr  = cp.exp(1j * kr)
        gaussian = cp.exp(-0.5 * K_squared * self.gaussian_width**2) # shape (Nx, Ny, Nz)
        rho_hat = gaussian * cp.sum(charges[None, None, None, :] * cp, axis=-1) # shape (Nx, Ny, Nz)
        K_squared[0,0,0] = 1 # avoid division by zero

        phi_hat = rho_hat /eps4pi * K_squared # shape (Nx, Ny, Nz)
        phi_hat[0,0,0] = 0
        field_hat = 1j * self.K * phi_hat[:,:,:,None]

        exp_ikr_target = cp.exp(1j * np.sum(self.K[:,:,:,None,:] * target_pos[None, None, None, :, :], axis=-1)) # shape (Nx, Ny, Nz, M)
        self.pot = cp.real(cp.sum(phi_hat[:,:,:,None] * exp_ikr_target, axis=(0,1,2)))
        self.field = cp.real(cp.sum(field_hat[:,:,:,:,None] * exp_ikr_target[:,:,:,None,:], axis=(0,1,2)))
        return self.pot, self.field
