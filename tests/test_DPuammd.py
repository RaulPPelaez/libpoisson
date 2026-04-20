import libPoisson as lp
import cupy as cp


def test_totalchargefailure():
    solver = lp.get_solver(("periodic", "periodic", "open"), "cuda",
                           L = (20.0, 20.0, 20.0), permittivity=1.0, tolerance=1e-6, splitting_ratio=5.0, charge_radius=0.2)

    target_pos = cp.zeros((1, 3))
    target_pos[:, 0] = 1.0
    source_charge = cp.array([1.0])

    field, pot = solver(target_pos, target_pos, source_charge)
    error = None
    try:
        _,_ = solver(target_pos, target_pos, source_charge+1.0)
    except ValueError as e:
        error = e
    assert error is not None, "Changed total charge but no error was raised"
