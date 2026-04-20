import cupy as np
from typing import ArrayLike
def generate_image_charges(eps0: float,
                           eps1: float,
                           eps2: float,
                           z1: float,
                           z2: float,
                           charges: ArrayLike,
                           postions: ArrayLike,
                           n_rebounds: int):
    """
    Generates all image charges up to n_rebounds for 3D positions (N,3),
    using the multiple reflection method between two planes.
    Parameters:
    - eps1, eps0, eps2: permittivities
    - z1, z2: positions of the planes
    - ccharges: array (N,)
    - positions: array (N,3)
    - n_rebounds: number of reflections to generate

    Returns:
    - charges_ext: array of charges including images
    - postions_ext: array of positions (M,3) including images
    """
    charges = np.atleast_1d(charges)
    postions = np.atleast_2d(postions)
    N = charges.shape[0]

    r10 = (eps0 - eps1) / (eps0 + eps1)
    r20 = (eps0 - eps2) / (eps0 + eps2)

    charges_ext = charges.copy()
    postions_ext = postions.copy()

    if n_rebounds == 0:
        return charges_ext, postions_ext

    n_array = np.arange(1, n_rebounds + 1)[:, np.newaxis]  # (n_rebounds,1)

    n_array_even = np.ceil(n_array/2)
    n_array_odd = n_array-n_array_even
    r10_pot = r10 ** n_array_even * r20 ** n_array_odd  # (n_rebounds,1)
    r20_pot = r20 ** n_array_even * r10 ** n_array_odd

    charges_rep = charges[np.newaxis, :]  # (1, N)
    z_orig = postions[:, 2][np.newaxis, :]  # (1,N)
    xy = postions[:, :2]  # (N,2) -> se mantiene constante

    q_im1 = charges_rep * r10_pot  # (n_rebounds, N)
    z_im1 = 2*n_array_even*z1 - 2*n_array_odd*z2 + (-1)**n_array * z_orig

    q_im2 = charges_rep * r20_pot
    z_im2 = 2*n_array_even*z2 - 2*n_array_odd*z1 + (-1)**n_array * z_orig

    xy_repeat = np.tile(xy, (n_rebounds, 1))  # (n_rebounds*N,2)
    z_im1_flat = z_im1.ravel()
    z_im2_flat = z_im2.ravel()

    pos_im1 = np.hstack([xy_repeat, z_im1_flat[:, np.newaxis]])  # (n_rebounds*N,3)
    pos_im2 = np.hstack([xy_repeat, z_im2_flat[:, np.newaxis]])

    q_im1_flat = q_im1.ravel()
    q_im2_flat = q_im2.ravel()

    charges_ext = np.concatenate([charges_ext, q_im1_flat, q_im2_flat])
    postions_ext = np.concatenate([postions_ext, pos_im1, pos_im2])

    assert charges_ext.shape[0] == postions_ext.shape[0], "Mismatch in number of charges and positions"
    assert len(charges_ext.shape) == 1, "Charges should be a 1D array"
    assert len(postions_ext.shape) == 2 and postions_ext.shape[1] == 3, "Positions should be a (M,3) array"
    return charges_ext, postions_ext

def two_walls_convergence_criterion(eps0: float, eps1: float, eps2: float, tol: float = 1e-3):
    r10 = (eps0 - eps1) / (eps0 + eps1)
    r20 = (eps0 - eps2) / (eps0 + eps2)
    r = np.sqrt(np.real(np.abs(r10*r20)))
    if r == 0:
        return 0
    if r == 1:
        return int(1/(2*tol))
    lntol = np.log(1/tol)
    lnr = np.log(1/r)
    N = int(min(np.ceil(lntol/lnr), 1/(2*tol)))
    return N
