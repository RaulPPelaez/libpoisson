from ..solver import Solver
import cupy as cp
from scipy.special import erf

class NBody(Solver):
    """
    N-body solver for electrostatic interactions. Complexity is O(N*M) due to the pairwise interactions between charges.

    Uses the free charge green's function to compute the potential and field at target positions due to source charges.
    Assumes that the charges are gaussian distributed with a given width (chargeRadius) to avoid singularities at zero distance.
    """
    def __init__(self,
                 z_wall: float = None,
                 wall_permittivity: float = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wall_position = z_wall
        self.wall_permittivity = wall_permittivity

    def _solve_open_open_open(self,
              source_pos: cp.ndarray,
              target_pos: cp.ndarray,
              charges: cp.ndarray,
              compute_potential: bool = True,
              compute_field: bool = True) -> tuple[cp.ndarray, cp.ndarray]:

        pot = None
        field = None
        eps = self.permittivity
        pos_i = target_pos.reshape(-1, 1, 3)  # (M, 1, 3)
        pos_j = source_pos.reshape(1, -1, 3) # (1, N, 3)
        r_ij = pos_i - pos_j # (M, N, 3)
        r = cp.linalg.norm(r_ij, axis=-1) # (M, N)
        eps4pi = 4 * cp.pi * eps
        a = self.charge_radius * cp.sqrt(2/3)  # Assuming charge_radius is the mean cuadratic radius sqrt(<r^2>) and charge distribution rho=exp(-(r/a)^2) then a = charge_radius * sqrt(2/3).
        r_sigma = r/a
        if compute_potential:
            G = 1 / (eps4pi * r) * erf(r_sigma)
            G[r == 0] = 1 / (eps4pi * a) * 2 / cp.sqrt(cp.pi)  # Handle the singularity at r=0 by using the limit of G as r approaches 0 (the potential is finite at r=0 due to the Gaussian charge distribution)
            pot = G @ charges
        if compute_field:
            gradG = (1/(eps4pi * r**3) * (erf(r_sigma) - 2/cp.sqrt(cp.pi) * r_sigma * cp.exp(-r_sigma*r_sigma)))[:,:,cp.newaxis] * r_ij
            gradG[r == 0] = 0  # Handle the singularity at r=0 by setting the gradient to zero (the field is finite at r=0 due to the Gaussian charge distribution)
            field = cp.sum(gradG * charges[:, cp.newaxis], axis=1)  # (M, 3)
        print("pot", pot.shape if pot is not None else None)
        print("field", field.shape if field is not None else None)
        return pot, field.flatten()

    def _solve_open_open_single_wall(self,
              source_pos: cp.ndarray,
              target_pos: cp.ndarray,
              charges: cp.ndarray,
              compute_potential: bool = True,
              compute_field: bool = True) -> tuple[cp.ndarray, cp.ndarray]:

        if self.wall_position is None or self.wall_permittivity is None:
            raise ValueError("Wall position and permittivity must be provided for single wall boundary condition.")

        eps = self.permittivity
        eps_wall = self.wall_permittivity
        z_wall = self.wall_position
        image_pos = cp.copy(source_pos)
        image_pos[2::3] = 2*z_wall - image_pos[2::3]
        print("source_pos", source_pos.shape)
        print("image_pos", image_pos.shape)
        source_pos = cp.concatenate((source_pos, image_pos), axis=0)
        print("source_pos", source_pos.shape)
        charges = cp.concatenate((charges, charges * (-eps_wall + eps) / (eps_wall + eps)), axis=0)
        return self._solve_open_open_open(source_pos, target_pos, charges, compute_potential, compute_field)

