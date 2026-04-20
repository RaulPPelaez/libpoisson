# Poisson Solver virtual class, every solver should inherit from this class and implement the solve function
from abc import ABC, abstractmethod
from numpy.typing import ArrayLike
from math import sqrt
from .parameters.base import BaseParameters
class Solver(ABC):
    '''
    A virtual class for solving the Poisson equation under different periodicity conditions.
    Parameters
    ----------
    permittivity: float
        The permittivity of the medium, which affects the strength of the electric field and potential.

    charge_radius: float
        Characteristic length of the charge distribution, used for regularization to avoid singularities in the potential and field calculations.

    Methods
    -------
    solve(source_pos, target_pos, charges, compute_potential=True, compute_field=True)
        Solve the Poisson equation for the given source and target positions and charges in
        the specified periodicity conditions. Returns the computed potential and field at the target points.

    __call__(source_pos, target_pos, charges, **kwargs)
        A convenient wrapper for the solve method, allowing the solver instance to be called directly with the same parameters as solve.

    Notes
    -----
    - The actual implementation of the solve method must be provided in the subclasses that inherit from this.
    '''
    def __init__(self, parameters: BaseParameters):
        self.parameters = parameters
        self.parameters.gaussian_width = self.parameters.charge_radius #For now we use the charge radius as the width of the Gaussian distribution for regularization, but this can be modified in the future if needed.

    def __getattr__(self, name):
        return getattr(self.parameters, name)

    @abstractmethod
    def solve(self,
              source_pos: ArrayLike,
              target_pos: ArrayLike,
              charges: ArrayLike,
              compute_potential: bool = True,
              compute_field: bool = True
              ) -> tuple[ArrayLike, ArrayLike]:
        '''
        Solve the Poisson equation for the given source and target positions and charges in
        the specified periodicity conditions.

        Parameters
        ----------
        source_pos: ArrayLike, shape (N,3)
            Positions of N source charges in the format [[x1, y1, z1], [x2, y2, z2], ..., [xN, yN, zN]]

        target_pos: ArrayLike, shape (M,3)
            Positions of M target points in the format [[x1, y1, z1], [x2, y2, z2], ..., [xM, yM, zM]]

        charges: ArrayLike, shape (N,)
            Charges of the N source charges.

        compute_potential: bool, optional
            Whether to compute the potential at the target points. Default is True.

        compute_field: bool, optional
            Whether to compute the electric field at the target points. Default is True.

        Returns
        -------
        potential: ArrayLike, shape (M,)
            The computed potential at each target point. Returned if compute_potential is True.

        field: ArrayLike, shape (M,3)
            The computed electric field at each target point in the format (Ex1, Ey1, Ez1, Ex2, Ey2, Ez2, ..., ExM, EyM, EzM). Returned if compute_field is True.
        '''


    def __call__(self,
                 source_pos: ArrayLike,
                 target_pos: ArrayLike,
                 charges: ArrayLike,
                 **kwargs
                 ) -> tuple[ArrayLike, ArrayLike]:
        return self.solve(source_pos, target_pos, charges, **kwargs)
    __call__.__doc__ = solve.__doc__
