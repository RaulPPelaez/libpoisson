def test_import():
    import libPoisson as lp

import libPoisson as lb
import pytest
import cupy as cp

EXTRA_PARAMETERS = {
        "TriplePeriodic": {
            "No_wall":{"Lx": 1.0, "Ly": 1.0, "Lz": 1.0, "splitting_ratio": -1.0, "tolerance": 1e-1}
            },
        "Open": {
            "Single_wall": {"floor_z": 0.0, "floor_permittivity": 1.0},
            "Double_wall": {"floor_z": 0.0, "floor_permittivity": 1.0, "ceil_z": 1.0, "ceil_permittivity": 1.0}
                  },
        "DoublePeriodic": {
            "No_wall": {"Lx": 3.0, "Ly": 3.0, "Lz":3.0, "splitting_ratio": 2.0, "tolerance": 1e-1},
            "Single_wall": {"Lx": 3.0, "Ly": 3.0, "Lz":3.0, "splitting_ratio": 2.0, "tolerance": 1e-1, "permittivity_bottom": 1.0},
            "Double_wall": {"Lx": 3.0, "Ly": 3.0, "Lz":3.0, "splitting_ratio": 2.0, "tolerance": 1e-1, "permittivity_bottom": 1.0, "permittivity_top": 1.0}
            },
    }

@pytest.mark.parametrize("periodicity", lb.AVAILABLE_PERIODICITIES)
@pytest.mark.parametrize("wall", lb.AVAILABLE_WALLS)
def test_initialization(periodicity, wall):
    if periodicity == "TriplePeriodic" and wall != "No_wall":
        with pytest.raises(ValueError):
            lb.get_solver(periodicity, wall, permittivity=1.0, charge_radius=0.1)
    else:
        try:
            if periodicity in EXTRA_PARAMETERS and wall in EXTRA_PARAMETERS[periodicity]:
                params = EXTRA_PARAMETERS[periodicity][wall]
                solver = lb.get_solver(periodicity, wall, permittivity=1.0, charge_radius=0.1, **params)
            else:
                solver = lb.get_solver(periodicity, wall, permittivity=1.0, charge_radius=0.1)
        except ValueError:
            pytest.fail(f"Unexpected failure for {periodicity}, {wall}")

@pytest.mark.parametrize("periodicity", lb.AVAILABLE_PERIODICITIES)
@pytest.mark.parametrize("wall", lb.AVAILABLE_WALLS)
def test_solve(periodicity, wall):
    solver = None
    if periodicity == "TriplePeriodic" and wall != "No_wall":
        with pytest.raises(ValueError):
            lb.get_solver(periodicity, wall, permittivity=1.0, charge_radius=0.1)
    else:
        try:
            if periodicity in EXTRA_PARAMETERS and wall in EXTRA_PARAMETERS[periodicity]:
                params = EXTRA_PARAMETERS[periodicity][wall]
                solver = lb.get_solver(periodicity, wall, permittivity=1.0, charge_radius=0.1, **params)
            else:
                solver = lb.get_solver(periodicity, wall, permittivity=1.0, charge_radius=0.1)
        except ValueError as e:
            pytest.fail(f"Unexpected failure for {periodicity}, {wall}: {e}")

    if solver is not None:
        source_pos = cp.zeros((2, 3))
        source_pos[1] = cp.array([0.5, 0.5, 0.5])
        charge = cp.array([1.0, -1.0])
        target_pos = cp.ones((1, 3))
        potential, field = solver.solve(source_pos, target_pos, charge)
        assert potential.shape == (1,)
        assert field.shape == (1, 3)
        assert potential[0] != 0.0
        assert cp.all(field[0] != 0.0)
