from .cuda.NBody.nbody import NBody
from .cuda.PMSimple.pmsimple import PMSimple
from .cuda.UAMMD_PSE.uammd_pse import uammdPSE
from .cuda.UAMMD_DPslab.uammd_dpslab import uammdDPSlab

AVAILABLE_WALLS = ["No_wall", "Single_wall", "Double_wall"]

SOLVER_MAP = {
    'Open': NBody,
    'TriplePeriodic': uammdPSE,
    'DoublePeriodic': uammdDPSlab
}

AVAILABLE_PERIODICITIES = list(SOLVER_MAP.keys())
XY_PERIODICIY_MAP = {
        'Open': 'open',
        'DoublePeriodic': 'periodic',
        'TriplePeriodic': 'periodic'
    }

Z_PERIODICITY_MAP = {
        'No_wall': 'open',
        'Single_wall': 'single_wall',
        'Double_wall': 'two_walls'
    }

def _solve_periodicity(periodicity, wall):
    if periodicity not in AVAILABLE_PERIODICITIES:
        raise ValueError(f"Invalid periodicity. Available options are: {AVAILABLE_PERIODICITIES}")
    if wall not in AVAILABLE_WALLS:
        raise ValueError(f"Invalid wall configuration. Available options are: {AVAILABLE_WALLS}")
    periodicityX = XY_PERIODICIY_MAP[periodicity]
    periodicityY = XY_PERIODICIY_MAP[periodicity]
    periodicityZ = Z_PERIODICITY_MAP[wall]
    # Exception for TriplePeriodic configuration
    if periodicity == 'TriplePeriodic':
        periodicityZ = XY_PERIODICIY_MAP[periodicity]
        if wall != "No_wall":
            raise ValueError("Walls are not compatible with TriplePeriodic configuration.")

    return periodicityX, periodicityY, periodicityZ

def get_solver(periodicity, walls="No_wall",
               *args, **kwargs):
    '''
    Factory function to get the appropriate solver based on periodicity and wall configuration.
    Args:
        periodicity (str): The type of periodicity ('Open', 'DoublePeriodic', 'TriplePeriodic').
        permittivity (float): The permittivity of the medium.
        charge_radius (float): The radius of the charge distribution.
        walls (str, optional): The wall configuration ('No_wall', 'Single_wall', 'Double_wall'). Default is 'No_wall'.
        *args: Additional positional arguments to pass to the solver constructor.
        **kwargs: Additional keyword arguments to pass to the solver constructor.
    '''
    periodicityX, periodicityY, periodicityZ = _solve_periodicity(periodicity, walls)
    solver_class = SOLVER_MAP.get(periodicity, None)
    if solver_class is None:
        raise NotImplementedError(f"Solver for periodicity '{periodicity}' is not implemented.")
    return solver_class(periodicityX=periodicityX, periodicityY=periodicityY, periodicityZ=periodicityZ, *args, **kwargs)

__all__ = ['NBody', 'PMSimple', 'uammdPSE', 'get_solver','AVAILABLE_PERIODICITIES', 'AVAILABLE_WALLS']
