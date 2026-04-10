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

def get_test_solver(periodicity, wall):
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
    return solver


@pytest.mark.parametrize("periodicity", lb.AVAILABLE_PERIODICITIES)
@pytest.mark.parametrize("wall", lb.AVAILABLE_WALLS)
def test_solve(periodicity, wall):
    solver = get_test_solver(periodicity, wall)
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


@pytest.mark.parametrize("periodicity", lb.AVAILABLE_PERIODICITIES)
@pytest.mark.parametrize("wall", lb.AVAILABLE_WALLS)
def test_Nsanity(periodicity,wall):
    solver = get_test_solver(periodicity, wall)
    if solver is not None:
        source_pos = cp.zeros((1, 3))
        source_charge = cp.array([1.0])
        target_pos = cp.ones((1, 3))
        target_pos[0, 0] = cp.random.rand() * 1.0 + 1.0
        target_pos[0, 1] = cp.random.rand() * 1.0 + 1.0
        target_pos[0, 2] = cp.random.rand() * 1.0 + 1.0
        potential_test, field_test = solver.solve(source_pos, target_pos, source_charge)
        Ntest = cp.random.randint(1e3, 1e4, size=4)
        for n in Ntest.get():
            source_pos = cp.zeros((n, 3))
            source_charge = cp.ones(n)
            target_pos_n = cp.ones((n, 3))
            target_pos_n[:,0] = target_pos[0, 0] + cp.zeros(n)
            target_pos_n[:,1] = target_pos[0, 1] + cp.zeros(n)
            target_pos_n[:,2] = target_pos[0, 2] + cp.zeros(n)
            potential_n, field_n = solver.solve(source_pos, target_pos_n, source_charge)
            assert cp.allclose(potential_n/n, potential_test, rtol=1e-5, atol=1e-5), "Linear N check failed for potential"
            assert cp.allclose(field_n/n, field_test, rtol=1e-5, atol=1e-5), "Linear N check failed for field"

