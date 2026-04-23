from ..base_solver import Solver
from ._uammd import uammd_dpslab
import cupy as cp
import numpy as np
from numpy.typing import ArrayLike

from ..parameters.ewaldsum import EwaldSumParameters, EwaldSumSingleWallParameters, EwaldSumDoubleWallParameters
from ..registry.decorator import register_solver
from ..definitions.boundary_conditions import BoundaryConditions, BCType
from ..definitions.device import DeviceType

def ratio_to_split(gaussian_width, splitting_ratio) -> float:
    if splitting_ratio > 0:
        return 1/(2*gaussian_width*splitting_ratio)
    else:
        return -1.0

def create_uammd_solver(solver):
    return uammd_dpslab(Lx=solver.L[0], Ly=solver.L[1], Lz=solver.L[2],
                        permittivity_inside=solver.permittivity, permittivity_top=solver.permittivity_top, permittivity_bottom=solver.permittivity_bottom,
                        gaussian_width=solver.gaussian_width, tolerance=solver.tolerance, split=solver.split)


class UAMMDPoissonSlab(Solver):
    def __init__(self,
                 parameters: EwaldSumParameters):
        super().__init__(parameters)
        self.split = ratio_to_split(self.gaussian_width, self.splitting_ratio)
        self.permittivity_top = parameters.permittivity
        self.permittivity_bottom = parameters.permittivity
        if hasattr(parameters, 'top_permittivity'):
            self.permittivity_top = parameters.top_permittivity
        if hasattr(parameters, 'bottom_permittivity'):
            self.permittivity_bottom = parameters.bottom_permittivity
        self.uammd_solver = create_uammd_solver(self)

    def solve(self,
              source_pos: ArrayLike,
              target_pos: ArrayLike,
              charges: ArrayLike,
              compute_potential: bool = True,
              compute_field:     bool = True) -> (cp.ndarray, cp.ndarray):
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



implementation = "uammd"
device = DeviceType.CUDA
bc = BoundaryConditions(BCType.PERIODIC, BCType.PERIODIC, BCType.OPEN)
@register_solver(bc, device, implementation, default=True)
class UAMMDPoissonSlabOpen(UAMMDPoissonSlab):
    Parameters = EwaldSumParameters

bc = BoundaryConditions(BCType.PERIODIC, BCType.PERIODIC, BCType.SINGLE_WALL)
@register_solver(bc, device, implementation, default=True)
class UAMMDPoissonSlabSingleWall(UAMMDPoissonSlab):
    Parameters = EwaldSumSingleWallParameters

bc = BoundaryConditions(BCType.PERIODIC, BCType.PERIODIC, BCType.DOUBLE_WALL)
@register_solver(bc, device, implementation, default=True)
class UAMMDPoissonSlabDoubleWall(UAMMDPoissonSlab):
    Parameters = EwaldSumDoubleWallParameters
