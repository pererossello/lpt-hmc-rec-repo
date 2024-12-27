import jax
import jax.numpy as jnp

from .fourier import get_k_1D, get_k_rfft_1D, my_fft, my_ifft
from .calculus import gradient_hat
from .utils import symmetric_gaussian_nonorm


def from_psi_to_delta(psi_x, psi_y, psi_z, N, L, pm_func, psi_has_hat=True):
    """
    Apply displacement vector to points distributed on a regular grid
    """

    if psi_has_hat:
        psi_x = my_ifft(psi_x, L)
        psi_y = my_ifft(psi_y, L)
        psi_z = my_ifft(psi_z, L)

    q = jnp.linspace(0, L, N, endpoint=False)
    X, Y, Z = jnp.meshgrid(q, q, q, indexing="ij")
    # del q

    X = (X + psi_x).ravel()
    Y = (Y + psi_y).ravel()
    Z = (Z + psi_z).ravel()

    positions = jnp.array([X, Y, Z]).T
    # del X, Y, Z, psi_x, psi_y, psi_z

    density = pm_func(positions, N, L)

    return density - 1


def get_psi_lpt1(phi_1, N, L):
    """
    Compute first order displacement field
    """
    return gradient_hat(-phi_1, N, L)


def get_psi_o2(psi1_x, psi1_y, psi1_z, one_over_k_sq, N, L):
    """
    Compute second order term of displacement field
    """

    k_1D = get_k_1D(N, L)
    k_1D_r = get_k_rfft_1D(N, L)

    phi1_dxx = 1j * k_1D[:, None, None] * (-psi1_x)
    phi1_dxx = my_ifft(phi1_dxx, L)
    phi1_dyy = 1j * k_1D[None, :, None] * (-psi1_y)
    phi1_dyy = my_ifft(phi1_dyy, L)
    phi1_dxy = 1j * k_1D[None, :, None] * (-psi1_x)
    phi1_dxy = my_ifft(phi1_dxy, L)
    arr1 = phi1_dxx * phi1_dyy - jnp.square(phi1_dxy)
    del phi1_dxy

    phi1_dzz = 1j * k_1D_r[None, None, :] * (-psi1_z)
    phi1_dzz = my_ifft(phi1_dzz, L)
    phi1_dxz = 1j * k_1D_r[None, None, :] * (-psi1_x)
    phi1_dxz = my_ifft(phi1_dxz, L)
    arr2 = phi1_dxx * phi1_dzz - jnp.square(phi1_dxz)
    del phi1_dxx, phi1_dxz

    phi1_dyz = 1j * k_1D_r[None, None, :] * (-psi1_y)
    phi1_dyz = my_ifft(phi1_dyz, L)
    arr3 = phi1_dyy * phi1_dzz - jnp.square(phi1_dyz)
    del phi1_dyy, phi1_dzz, phi1_dyz

    arr = arr1 + arr2 + arr3
    del arr1, arr2, arr3
    arr = my_fft(arr, L)

    phi2 = -arr * one_over_k_sq

    psi2_x = 1j * k_1D[:, None, None] * phi2
    psi2_y = 1j * k_1D[None, :, None] * phi2
    psi2_z = 1j * k_1D_r[None, None, :] * phi2

    return psi2_x, psi2_y, psi2_z


def get_psi_lpt2(phi1, D1, D2, one_over_k_sq, N, L):
    """
    Compute displacement field up to second order
    """

    psi1_x, psi1_y, psi1_z = get_psi_lpt1(phi1, N, L)
    psi2_x, psi2_y, psi2_z = get_psi_o2(psi1_x, psi1_y, psi1_z, one_over_k_sq, N, L)

    psi_x = D1 * psi1_x + D2 * psi2_x
    psi_y = D1 * psi1_y + D2 * psi2_y
    psi_z = D1 * psi1_z + D2 * psi2_z

    return psi_x, psi_y, psi_z


def get_psi_sc(delta_in, D1, k_sq, N, L, MUSCLE, MUSCLE_ITERS, SC_CORRECTION):
    """
    Compute displacement field for Spherical Collapse aproximation

    :delta_in: Initial density field in configuration space
    """

    one_over_k_sq = jnp.where(k_sq > 0, 1 / k_sq, 0.0)
    delta_in *= D1

    threshold = 3/2
    transition_width = 0.25

    if MUSCLE:
        RES = L / N
        bool_arr = delta_in < 3 / 2
        delta_in_hat = my_fft(delta_in, L)

        if MUSCLE_ITERS == "MANY":

            raise ValueError("Don't use MANY, not differentiable yet!")
            # Many iterations
            def cond_fun(state):
                n, perc, bool_arr = state
                return perc < 0.99

            def body_fun(state):
                n, perc, bool_arr = state
                r = (2**n) * RES
                one_over_var_k = r**2
                filt_k = symmetric_gaussian_nonorm(k_sq, one_over_var_k)
                delta_smooth = my_ifft(filt_k * delta_in_hat, L).real
                bool_arr_smooth = delta_smooth < 3 / 2
                bool_arr = bool_arr & bool_arr_smooth
                perc = jnp.mean(bool_arr_smooth)
                n += 1
                return n, perc, bool_arr

            init_state = (0, 0.0, bool_arr)
            _, _, bool_arr = jax.lax.while_loop(cond_fun, body_fun, init_state)

        elif MUSCLE_ITERS == "ONE":
            # Single iteratoin
            r = (2**0) * RES  # Could be changed
            one_over_var_k = r**2
            filt_k = symmetric_gaussian_nonorm(k_sq, one_over_var_k)
            delta_smooth = my_ifft(filt_k * delta_in_hat, L)

        smooth_factor_1 = jax.nn.sigmoid((threshold - delta_smooth) / transition_width)
        smooth_factor_2 = jax.nn.sigmoid((threshold - delta_in) / transition_width)
        mask = smooth_factor_1 * smooth_factor_2

        smooth_value = jnp.log1p(jnp.exp(1 - 2 / 3 * delta_in))
        div_psi = 3*mask * jnp.sqrt(smooth_value)-3

        
        # mask = jax.nn.sigmoid(
        #     (threshold - jnp.minimum(delta_smooth, delta_in))
        #     / transition_width
        # )

        # smooth_value = jnp.log1p(jnp.exp(1 - (2/3)*delta_in))
        # div_psi = 3 * mask * jnp.sqrt(smooth_value) - 3



        if SC_CORRECTION:
            sum_div_psi = jnp.sum(div_psi)
            offset = sum_div_psi / jnp.sum(mask)
            div_psi = div_psi - offset * mask



        # delta_in = delta_in.astype(complex)  # so no nans in sqrt root
        # div_psi = jnp.where(bool_arr, 3 * jnp.sqrt(1 - 2 / 3 * delta_in) - 3, -3).real

    else:


        smooth_factor = jax.nn.sigmoid((threshold - delta_in) / transition_width)
        smooth_value = jnp.log1p(jnp.exp(1 - 2 / 3 * delta_in))
        div_psi = 3*smooth_factor * jnp.sqrt(smooth_value)-3

        if SC_CORRECTION:
            sum_div_psi = jnp.sum(div_psi)
            offset = sum_div_psi / jnp.sum(smooth_factor)
            div_psi = div_psi - offset * smooth_factor



    div_psi = my_fft(div_psi, L)
    phi_sc = -div_psi * one_over_k_sq

    psi_x, psi_y, psi_z = gradient_hat(phi_sc, N, L)

    return psi_x, psi_y, psi_z


def get_psi_alpt(
    delta_in,
    D1,
    D2,
    k_sq,
    N,
    L,
    R_S,
    MUSCLE,
    MUSCLE_ITERS,
    SC_CORRECTION,
):
    # delta_in has hat
    one_over_k_sq = jnp.where(k_sq > 0, 1 / k_sq, 0.0)
    phi1 = -delta_in * one_over_k_sq

    psi_lpt2_x, psi_lpt2_y, psi_lpt2_z = get_psi_lpt2(
        phi1,
        D1,
        D2,
        one_over_k_sq,
        N,
        L,
    )
    del phi1

    delta_in = my_ifft(delta_in, L)
    psi_sc_x, psi_sc_y, psi_sc_z = get_psi_sc(
        delta_in, D1, k_sq, N, L, MUSCLE, MUSCLE_ITERS, SC_CORRECTION
    )
    del delta_in

    one_over_var_k = R_S**2
    filt_k = symmetric_gaussian_nonorm(k_sq, one_over_var_k)

    psi_x = filt_k * psi_lpt2_x + (1 - filt_k) * psi_sc_x
    del psi_lpt2_x, psi_sc_x
    psi_y = filt_k * psi_lpt2_y + (1 - filt_k) * psi_sc_y
    del psi_lpt2_y, psi_sc_y
    psi_z = filt_k * psi_lpt2_z + (1 - filt_k) * psi_sc_z
    del psi_lpt2_z, psi_sc_z

    return psi_x, psi_y, psi_z
