import libPoisson as lb
import cupy as cp
import numpy as np
Lx, Ly, Lz = 10.0, 7.0, 5.0
charge_radius = 0.3
splitting_ratio = 0.3
tolerance = 1e-8
permittivity = 1.0
solver = lb.get_solver("TriplePeriodic",
                       Lx=Lx, Ly=Ly, Lz=Lz,
                       charge_radius=charge_radius, permittivity=permittivity,
                       tolerance=tolerance,
                       splitting_ratio=splitting_ratio)
def test_periodicity():
    source_pos = cp.ones((1,3))
    target_pos = cp.zeros((100,3))
    charge = cp.ones((1,))
    target_pos[:,0] = cp.arange(100) * Lx
    target_pos[:,1] = cp.arange(100) * Ly*0
    target_pos[:,2] = cp.arange(100) * Lz*0
    potential, field = solver.solve(source_pos, target_pos, charge)
    pot_deviation = cp.abs(potential - potential[0])/cp.abs(potential[0])
    assert cp.all(pot_deviation < 1e-4), f"Periodicity test failed: max deviation = {cp.max(pot_deviation)}"
    field_deviation = cp.linalg.norm(field - field[0,:], axis=1)/cp.linalg.norm(field[0,:])
    assert cp.all(field_deviation < 1e-4), f"Periodicity test failed: max field deviation = {cp.max(field_deviation)}"

def test_high_simetry_points():
    source_pos = cp.array([[0.0, 0.0, 0.0]])
    target_pos = cp.array([[Lx/2, 0.0, 0.0],
                           [0.0, Ly/2, 0.0],
                           [0.0, 0.0, Lz/2],
                           [Lx/2, Ly/2, 0.0],
                           [Lx/2, 0.0, Lz/2],
                           [0.0, Ly/2, Lz/2],
                           [Lx/2, Ly/2, Lz/2],
                           [0.0, 0.0, 0.0]]).reshape(-1,3)
    charge = cp.array([1.0])
    potential, field = solver.solve(source_pos, target_pos, charge)
    print(field)
