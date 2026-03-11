# Poisson Solver virtual class, every solver should inherit from this class and implement the solve function
from abc import ABC, abstractmethod
from numpy.typing import ArrayLike


PERIODICITY_OPTIONS = ['periodic', 'open', 'single_wall', 'two_walls','unspecify']

class Solver(ABC):
    '''
    A virtual class for solving the Poisson equation under different periodicity conditions.
    Parameters
    ----------
    permittivity: float
        The permittivity of the medium, which affects the strength of the electric field and potential.

    charge_radius: float
        Characteristic length of the charge distribution, used for regularization to avoid singularities in the potential and field calculations.

    periodicityX: str
        Periodicity condition in the X direction. Must be one of: 'periodic', 'open', 'single_wall', 'two_walls'.

    periodicityY: str
        Periodicity condition in the Y direction. Must be one of: 'periodic', 'open', 'single_wall', 'two_walls'.

    periodicityZ: str
        Periodicity condition in the Z direction. Must be one of: 'periodic', 'open', 'single_wall', 'two_walls'.

    need_complex: bool, optional
        Whether to compute the complex potential and field. Default is False.

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
    - Developers must implement the _solve_{periodicityX}_{periodicityY}_{periodicityZ} method for each combination of periodicity conditions they wish to support.
    '''
    def __init__(self,
                 permittivity: float,
                 charge_radius: float,
                 periodicityX: str,
                 periodicityY: str,
                 periodicityZ: str,
                 need_complex: bool = False
                 ):
        self.permittivity = permittivity
        self.charge_radius = charge_radius
        if periodicityX not in PERIODICITY_OPTIONS or periodicityY not in PERIODICITY_OPTIONS or periodicityZ not in PERIODICITY_OPTIONS:
            raise ValueError("Invalid periodicity option. Must be one of: {}".format(PERIODICITY_OPTIONS))
        self.periodicityX = periodicityX
        self.periodicityY = periodicityY
        self.periodicityZ = periodicityZ
        self.need_complex = need_complex

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
        -------
        potential: ArrayLike, shape (M,)
            The computed potential at each target point. Returned if compute_potential is True.

        field: ArrayLike, shape (3M,)
            The computed electric field at each target point in the format (Ex1, Ey1, Ez1, Ex2, Ey2, Ez2, ..., ExM, EyM, EzM). Returned if compute_field is True.
        '''
        solver_name = f"_solve_{self.periodicityX}_{self.periodicityY}_{self.periodicityZ}"
        solver = getattr(self, solver_name, None)
        if solver:
            return solver(source_pos, target_pos, charges, compute_potential, compute_field)
        raise NotImplementedError("The solver for periodicity (X={}, Y={}, Z={}) is not implemented.".format(self.periodicityX, self.periodicityY, self.periodicityZ))

    def __call__(self,
                 source_pos: ArrayLike,
                 target_pos: ArrayLike,
                 charges: ArrayLike,
                 **kwargs
                 ) -> tuple[ArrayLike, ArrayLike]:
        return self.solve(source_pos, target_pos, charges, **kwargs)
    __call__.__doc__ = solve.__doc__
