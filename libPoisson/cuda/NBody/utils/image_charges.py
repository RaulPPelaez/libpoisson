import cupy as np
def generate_image_charges(eps0, eps1, eps2, z1, z2, cargas, posiciones, n_rebounds):
    """
    Genera todas las imágenes hasta n_rebounds para posiciones 3D (N,3),
    usando el método de reflexiones múltiples entre dos planos.
    Parámetros:
    - eps1, eps0, eps2: permitividades
    - z1, z2: posiciones de los planos
    - cargas: array (N,)
    - posiciones: array (N,3)
    - n_rebounds: número de reflexiones a generar

    Devuelve:
    - cargas_ext: array de cargas incluyendo imágenes
    - posiciones_ext: array de posiciones (M,3) incluyendo imágenes
    """
    cargas = np.atleast_1d(cargas)
    posiciones = np.atleast_2d(posiciones)
    N = cargas.shape[0]

    # Coeficientes de reflexión
    r10 = (eps0 - eps1) / (eps0 + eps1)
    r20 = (eps0 - eps2) / (eps0 + eps2)

    # Cargas y posiciones originales
    cargas_ext = cargas.copy()
    posiciones_ext = posiciones.copy()

    if n_rebounds == 0:
        return cargas_ext, posiciones_ext

    # Array de rebotes: 1..n
    n_array = np.arange(1, n_rebounds + 1)[:, np.newaxis]  # (n_rebounds,1)

    # Potencias de los coeficientes
    n_array_even = np.ceil(n_array/2)
    n_array_odd = n_array-n_array_even
    r10_pot = r10 ** n_array_even * r20 ** n_array_odd  # (n_rebounds,1)
    r20_pot = r20 ** n_array_even * r10 ** n_array_odd

    # Replicar cargas y z originales para broadcasting
    cargas_rep = cargas[np.newaxis, :]  # (1, N)
    z_orig = posiciones[:, 2][np.newaxis, :]  # (1,N)
    xy = posiciones[:, :2]  # (N,2) -> se mantiene constante

    # Reflejo en z1
    q_im1 = cargas_rep * r10_pot  # (n_rebounds, N)
    z_im1 = 2*n_array_even*z1 - 2*n_array_odd*z2 + (-1)**n_array * z_orig

    # Reflejo en z2
    q_im2 = cargas_rep * r20_pot
    z_im2 = 2*n_array_even*z2 - 2*n_array_odd*z1 + (-1)**n_array * z_orig

    # Construir posiciones 3D
    xy_repeat = np.tile(xy, (n_rebounds, 1))  # (n_rebounds*N,2)
    z_im1_flat = z_im1.ravel()
    z_im2_flat = z_im2.ravel()

    pos_im1 = np.hstack([xy_repeat, z_im1_flat[:, np.newaxis]])  # (n_rebounds*N,3)
    pos_im2 = np.hstack([xy_repeat, z_im2_flat[:, np.newaxis]])

    # Flatten cargas
    q_im1_flat = q_im1.ravel()
    q_im2_flat = q_im2.ravel()

    # Concatenar con originales
    cargas_ext = np.concatenate([cargas_ext, q_im1_flat, q_im2_flat])
    posiciones_ext = np.concatenate([posiciones_ext, pos_im1, pos_im2])

    return cargas_ext, posiciones_ext

def two_walls_convergence_criterion(eps0, eps1, eps2, tol=1e-3):
    """
    https://chatgpt.com/c/69c2733b-5610-8332-a53a-f9560a27a8e6
    """
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
