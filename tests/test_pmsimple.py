import libPoisson as lp
import cupy as np
from scipy.special import erf

def test_sanity():
    solver = lp.PMSimple(permittivity=1.0, charge_radius=0.1,periodicityX="periodic",periodicityY="periodic",periodicityZ="periodic", gaussian_cutoff=0.1, L=[1.0, 1.0, 1.0], n_grid=[10, 10, 10])
    pos = np.zeros((3, 3))  # 3 particles at the origin
    print(pos.shape)
    charges = np.array([0.0, 0.0, 0.0])
    phi,E = solver(pos, pos, charges)
    phi,_ = solver(pos, pos, charges, compute_field=False)
    _, E = solver(pos, pos, charges, compute_potential=False)
    assert phi.shape[0] == 3, "Expected 3 potentials, got "+str(phi.shape[0])
    assert phi.shape[1] == 1, "Expected 3 potentials, got "+str(phi.shape[1])
    assert E.shape[0] == 3, "Expected 3 electric field components, got "+str(E.shape[0])
    assert E.shape[1] == 3, "Expected 3 electric field components, got "+str(E.shape[1])
    assert np.allclose(phi, np.zeros_like(phi))
    assert np.allclose(E, np.zeros_like(E))
