import libPoisson as lp
import cupy as np
from scipy.special import erf

def test_sanity():
    solver = lp.NBody(permittivity=1.0, charge_radius=0.1,periodicityX="open",periodicityY="open",periodicityZ="open")
    pos = np.zeros((3, 3))  # 3 particles at the origin
    print(pos.shape)
    charges = np.array([0.0, 0.0, 0.0])
    phi,E = solver(pos, pos, charges)
    phi,_ = solver(pos, pos, charges, compute_field=False)
    _, E = solver(pos, pos, charges, compute_potential=False)
    assert phi.shape[0] == 3, "Expected 3 potentials, got "+str(phi.shape[0])
    assert E.shape[0] == 3 and E.shape[1] == 3, "Expected electric field shape (3,3), got "+str(E.shape)
    assert np.allclose(phi, np.zeros_like(phi))
    assert np.allclose(E, np.zeros_like(E))


def phi_th(r, q, eps=1.0, charge_radius=0.1):
    a = charge_radius*np.sqrt(2/3)
    phi = q / (4 * np.pi * eps * r) * erf(r / a)
    phi[r == 0] = q / (4 * np.pi * eps * a)* 2 / np.sqrt(np.pi)
    return phi

def E_th(x, q, eps=1.0, charge_radius=0.1):
    a = charge_radius*np.sqrt(2/3)
    r = np.abs(x)
    Er = q / (4 * np.pi * eps * r**2) * ((erf(r / a)) - (r / a) * 2/np.sqrt(np.pi) * np.exp(-(r/a)**2 ))
    Er[r == 0] = 0.0
    return Er*np.sign(x)

def test_green_tensor():
    solver = lp.NBody(permittivity=1.0, charge_radius=0.1,periodicityX="open",periodicityY="open",periodicityZ="open")
    source_pos = np.array([[0.0, 0.0, 0.0]])
    target_x = np.linspace(-1.0, 1.0, 10)
    target_y = np.zeros_like(target_x)
    target_z = np.zeros_like(target_x)
    target_pos = np.stack((target_x, target_y, target_z), axis=-1)
    charges = np.array([1.0])
    phi,E = solver(source_pos, target_pos, charges)
    phi_test = phi_th(target_x, 1.0, eps=1.0, charge_radius=0.1)
    assert np.allclose(phi, phi_test, atol=1e-3), "Potential does not match theoretical values get:\n "+str(phi)+"\n expected: "+str(phi_test)
    E_test = E_th(target_x, 1.0, eps=1.0, charge_radius=0.1)
    assert np.allclose(E.reshape(-1,3)[:, 0], E_test, atol=1e-3), "Electric field does not match theoretical values get:\n "+str(E.reshape(-1,3)[:, 0])+"\n expected: "+str(E_test)

def test_two_charges():
    solver = lp.NBody(permittivity=1.0, charge_radius=0.1,periodicityX="open",periodicityY="open",periodicityZ="open")
    source_pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    target_x = np.linspace(-1.0, 1.0, 10)
    target_y = np.zeros_like(target_x)
    target_z = np.zeros_like(target_x)
    target_pos = np.stack((target_x, target_y, target_z), axis=-1)
    charges = np.array([1.0, -1.0])
    phi,E = solver(source_pos, target_pos, charges)
    phi_test = phi_th(target_x, 1.0, eps=1.0, charge_radius=0.1) + phi_th(target_x-0.5, -1.0, eps=1.0, charge_radius=0.1)
    assert np.allclose(phi, phi_test, atol=1e-3), "Potential does not match theoretical values get:\n "+str(phi)+"\n expected: "+str(phi_test)

def test_punctual_behaviour():
    solver = lp.NBody(permittivity=1.0, charge_radius=0.1,periodicityX="open",periodicityY="open",periodicityZ="open")
    source_pos = np.array([[0.0, 0.0, 0.0]])
    target_x = np.linspace(10.0, 100.0, 10)
    target_y = np.zeros_like(target_x)
    target_z = np.zeros_like(target_x)
    target_pos = np.stack((target_x, target_y, target_z), axis=-1)
    charges = np.array([1.0])
    phi,E = solver(source_pos, target_pos, charges)
    phi_test = 1/(4 * np.pi * 1.0 * target_x)
    assert np.allclose(phi, phi_test, atol=1e-3), "Potential does not match theoretical values get:\n "+str(phi)+"\n expected: "+str(phi_test)
    E_test = 1/(4 * np.pi * 1.0 * target_x**2)
    assert np.allclose(E.reshape(-1,3)[:, 0], E_test, atol=1e-3), "Electric field does not match theoretical values get:\n "+str(E.reshape(-1,3)[:, 0])+"\n expected: "+str(E_test)

def test_zero_eps_wa(floor_z: float=None, ceil_z: float=None, floor_permittivity: float=None, ceil_permittivity: float=None, need_complex: bool=False):
    solver = lp.NBody(permittivity=1.0, charge_radius=0.1,periodicityX="open",periodicityY="open",periodicityZ="single_wall", floor_z=0.0, floor_permittivity=0.0)
    source_pos = np.array([[0.0, 0.0, 1.0]])
    target_z = np.linspace(0.0, 5.0, 100)
    target_x = np.zeros_like(target_z)
    target_y = np.zeros_like(target_z)
    target_pos = np.stack((target_x, target_y, target_z), axis=-1)
    charges = np.array([1.0])
    print("shapes", source_pos.shape, target_pos.shape, charges.shape)
    phi,E = solver(source_pos, target_pos, charges)
    # Image charge at z=-1.0 with same sign
    phi_test = phi_th(target_z-1.0, 1.0, eps=1.0, charge_radius=0.1) + phi_th(target_z+1.0, 1.0, eps=1.0, charge_radius=0.1)
    assert np.allclose(phi, phi_test, atol=1e-3), "Potential does not match theoretical values get:\n "+str(phi)+"\n expected: "+str(phi_test)

def test_inf_eps_wall():
    solver = lp.NBody(permittivity=1.0, charge_radius=0.1,periodicityX="open",periodicityY="open",periodicityZ="single_wall", floor_z=0.0, floor_permittivity=1000000.0)
    source_pos = np.array([[0.0, 0.0, 1.0]])
    target_z = np.linspace(0.0, 5.0, 100)
    target_x = np.zeros_like(target_z)
    target_y = np.zeros_like(target_z)
    target_pos = np.stack((target_x, target_y, target_z), axis=-1)
    charges = np.array([1.0])
    phi,E = solver(source_pos, target_pos, charges)
    # Image charge at z=-1.0 with opposite sign
    phi_test = phi_th(target_z-1.0, 1.0, eps=1.0, charge_radius=0.1) - phi_th(target_z+1.0, 1.0, eps=1.0, charge_radius=0.1)
    assert np.allclose(phi, phi_test, atol=1e-3), "Potential does not match theoretical values get:\n "+str(phi)+"\n expected: "+str(phi_test)

