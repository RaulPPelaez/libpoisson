def test_import():
    import libPoisson as lp

import libPoisson as lp
import pytest
import cupy as cp

bctype = lp.BCType
bc_cls = lp.BoundaryConditions

Open = bctype.OPEN
Single_wall = bctype.SINGLE_WALL
Double_wall = bctype.DOUBLE_WALL
Periodic = bctype.PERIODIC
NBodySingleWall = bc_cls(Open, Open, Single_wall)
NBodyDoubleWall = bc_cls(Open, Open, Double_wall)
TriplePeriodic = bc_cls(Periodic, Periodic, Periodic)
DoublePeriodic = bc_cls(Periodic, Periodic, Open)
DoublePeriodic_SingleWall = bc_cls(Periodic, Periodic, Single_wall)
DoublePeriodic_DoubleWall = bc_cls(Periodic, Periodic, Double_wall)
EXTRA_PARAMETERS = {
        TriplePeriodic: {"L":[10.0, 10.0, 10.0], "splitting_ratio": -1.0, "tolerance": 1e-1},
        NBodySingleWall: {"bottom_wall_position": 0.0, "bottom_permittivity": 1.0},
        NBodyDoubleWall: {"bottom_wall_position": 0.0, "bottom_permittivity": 1.0, "top_wall_position": 1.0, "top_permittivity": 1.0},
        DoublePeriodic: {"L":(3.0, 3.0, 3.0), "splitting_ratio": 2.0, "tolerance": 1e-1},
        DoublePeriodic_SingleWall: {"L":(3.0, 3.0, 3.0), "splitting_ratio": 2.0, "tolerance": 1e-1, "bottom_permittivity": 1.0},
        DoublePeriodic_DoubleWall: {"L":(3.0, 3.0, 3.0), "splitting_ratio": 2.0, "tolerance": 1e-1, "bottom_permittivity": 1.0, "top_permittivity": 1.0}
    }

def get_test_solver(key):
    try:
        bc, device = key
    except ValueError:
        bc, device, impl = key
    if bc in EXTRA_PARAMETERS:
        params = EXTRA_PARAMETERS[bc]
        solver = lp.get_solver(bc, device, permittivity=1.0, charge_radius=0.1, **params)
    else:
        solver = lp.get_solver(bc, device, permittivity=1.0, charge_radius=0.1)
    return solver

keys = lp.registry.core._SOLVER_REGISTRY.keys()

@pytest.mark.parametrize("key", keys)
def test_initialization(key):
    get_test_solver(key)

def test_tuple_initialization():
    solver_tuple  = lp.get_solver(("open","open","open"),"cuda", charge_radius=0.1, permittivity=1.0)
    solver_normal = lp.get_solver(bc_cls(Open, Open, Open), lp.DeviceType.CUDA, charge_radius=0.1, permittivity=1.0)
    assert type(solver_tuple) == type(solver_normal), "Tuple initialization did not return the same type of solver as normal initialization"

@pytest.mark.parametrize("key", keys)
def test_solve(key):
    solver = get_test_solver(key)
    if solver is not None:
        source_pos = cp.zeros((2, 3))
        source_pos[1] = cp.array([0.5, 0.5, 0.5])
        charge = cp.array([1.0, -1.0])
        target_pos = cp.ones((1, 3))
        assert(len(target_pos.shape) == 2)
        potential, field = solver.solve(source_pos, target_pos, charge)
        assert potential.shape == (1,)
        assert field.shape == (1, 3)
        assert potential[0] != 0.0
        assert cp.all(field[0] != 0.0)

@pytest.mark.parametrize("key", keys)
def test_Nsanity(key):
    solver = get_test_solver(key)
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

