# Poisson Solver virtual class, every solver should inherit from this class and implement the solve function
from abc import ABC, abstractmethod
from numpy.typing import ArrayLike

class Solver(ABC):
    def __init__(self,
                 permittivity: float,
                 charge_radius: float,
                 periodicityX: str = 'unspecify',
                 periodicityY: str = 'unspecify',
                 periodicityZ: str = 'unspecify',
                 need_complex: bool = False
                 ):
        self.permittivity = permittivity
        self.charge_radius = charge_radius
        self.periodicityX = periodicityX
        self.periodicityY = periodicityY
        self.periodicityZ = periodicityZ
        self.need_complex = need_complex

    @abstractmethod
    def solve(self,
              source_pos: ArrayLike,
              target_pos: ArrayLike,
              charges: ArrayLike,
              compute_potential: bool = True,
              compute_field: bool = True
              ) -> tuple[ArrayLike, ArrayLike]:
        '''
        Solve the Poisson equation for the given source and target positions and charges.

        Parameters
        ----------
        source_pos: ArrayLike, shape (3N)
            Positions of N source charges in the format (x1, y1, z1, x2, y2, z2, ..., xN, yN, zN)

        target_pos: ArrayLike, shape (3M)
            Positions of M target points in the format (x1, y1, z1, x2, y2, z2, ..., xM, yM, zM)

        charges: ArrayLike, shape (N,)
            Charges of the N source charges.

        compute_potential: bool, optional
            Whether to compute the potential at the target points. Default is True.

        compute_field: bool, optional
            Whether to compute the electric field at the target points. Default is True.

        Returns
        '''
        raise NotImplementedError("This method should be implemented by subclasses")

    def __call__(self,
                 source_pos: ArrayLike,
                 target_pos: ArrayLike,
                 charges: ArrayLike,
                 **kwargs
                 ) -> tuple[ArrayLike, ArrayLike]:
        return self.solve(source_pos, target_pos, charges, **kwargs)
    __call__.__doc__ = solve.__doc__
