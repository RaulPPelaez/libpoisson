from ..base_solver import Solver
from ._uammd import uammd_pse
import cupy as cp
import numpy as np
from numpy.typing import ArrayLike

from ..parameters.ewaldsum import EwaldSumParameters
from ..registry.decorator import register_solver
from ..definitions.boundary_conditions import BoundaryConditions, BCType
from ..definitions.device import DeviceType

implementation = "uammd"
device = DeviceType.CUDA
bc = BoundaryConditions(BCType.PERIODIC, BCType.PERIODIC, BCType.PERIODIC)

@register_solver(bc, device, implementation, default=True)
class UAMMDSplitEwaldPoisson(Solver):
    """
    Solver for the Poisson Equation using the Split Ewald Poisson algorithm implemented in UAMMD.
    This solver is designed for systems with periodic boundary conditions in all three dimensions.
    """
    Parameters = EwaldSumParameters
    def __init__(self, parameters: EwaldSumParameters):
        super().__init__(parameters)
        self.split = 1/(2*self.gaussian_width*self.splitting_ratio) if self.splitting_ratio > 0 else -1.0
        Lx, Ly, Lz = self.L
        self.uammd_solver = uammd_pse(Lx=Lx, Ly=Ly, Lz=Lz, permittivity=self.permittivity, gaussian_width=self.gaussian_width, tolerance=self.tolerance, split=self.split)

    def solve(self,
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
