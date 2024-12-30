from functools import partial

import jax
import jax.numpy as jnp

from .fourier import my_fft, my_ifft, get_k, get_one_over_k_sq
from .utils import compute_or_load_D1_D2, compute_or_load_pow_spec
from .particle_mesh import cic
from .utils import symmetric_gaussian_nonorm
from .ics import get_delta
from .lpt_displacements import (
    from_psi_to_delta,
    get_psi_lpt1,
    get_psi_lpt2,
    get_psi_sc,
    get_psi_alpt,
)


def get_forward_lpt(
    N,
    L,
    Z_I,
    Z_F,
    LPT_METHOD,
    PM_METHOD="CIC",
    IC_KIND="U",
    R_S=4,
    MUSCLE=True,
    MUSCLE_ITERS="ONE",
    SC_CORRECTION=True,
    PARTICLE_RIDGE=False,
    K_TH_PR=0.4,
    D_PR=0.2,
):

    D1, D2 = compute_or_load_D1_D2(Z_I, Z_F)

    if PM_METHOD == "CIC":
        # set particle mesh function
        pm_func = cic

    # depending on input array, we need to compute pow_spec or not
    if IC_KIND in ["U", "FSK_U"]:
        pow_spec = compute_or_load_pow_spec(N, L, Z_I)
    else:
        pow_spec = None

    # pre-setting arguments such that forward_lpt depends only on input array
    if LPT_METHOD == "LPT1":
        forward_lpt = partial(forward_lpt1, D1=D1)
    elif LPT_METHOD == "LPT2":
        forward_lpt = partial(forward_lpt2, D1=D1, D2=D2)
    elif LPT_METHOD == "SC":
        forward_lpt = partial(
            forward_sc,
            D1=D1,
            MUSCLE=MUSCLE,
            MUSCLE_ITERS=MUSCLE_ITERS,
            SC_CORRECTION=SC_CORRECTION,
        )
    elif LPT_METHOD == "ALPT":
        forward_lpt = partial(
            forward_alpt,
            D1=D1,
            D2=D2,
            R_S=R_S,
            MUSCLE=MUSCLE,
            MUSCLE_ITERS=MUSCLE_ITERS,
            SC_CORRECTION=SC_CORRECTION,
            PARTICLE_RIDGE=PARTICLE_RIDGE,
            K_TH_PR=K_TH_PR,
            D_PR=D_PR,
        )

    forward_lpt = partial(
        forward_lpt,
        N=N,
        L=L,
        INPUT_ARG=IC_KIND,
        pm_func=pm_func,
        pow_spec=pow_spec,
    )

    return forward_lpt


def forward_lpt1(x, N, L, D1, INPUT_ARG, pm_func, pow_spec):

    delta_in_hat = get_delta(x, N, L, INPUT_ARG, pow_spec, return_hat=True)
    one_over_k_sq = get_one_over_k_sq(N, L)

    phi_1_hat = -delta_in_hat * one_over_k_sq * D1
    psi_x_hat, psi_y_hat, psi_z_hat = get_psi_lpt1(phi_1_hat, N, L)

    delta = from_psi_to_delta(
        psi_x_hat, psi_y_hat, psi_z_hat, N, L, pm_func, psi_has_hat=True
    )

    return delta


def forward_lpt2(x, N, L, D1, D2, INPUT_ARG, pm_func, pow_spec):

    delta_in_hat = get_delta(x, N, L, INPUT_ARG, pow_spec, return_hat=True)
    one_over_k_sq = get_one_over_k_sq(N, L)

    phi_1_hat = -delta_in_hat * one_over_k_sq
    psi_x_hat, psi_y_hat, psi_z_hat = get_psi_lpt2(
        phi_1_hat,
        D1,
        D2,
        one_over_k_sq,
        N,
        L,
    )

    delta = from_psi_to_delta(
        psi_x_hat, psi_y_hat, psi_z_hat, N, L, pm_func, psi_has_hat=True
    )

    return delta


def forward_sc(
    x,
    N,
    L,
    D1,
    INPUT_ARG,
    pm_func,
    pow_spec,
    MUSCLE,
    MUSCLE_ITERS,
    SC_CORRECTION,
):

    delta_in = get_delta(x, N, L, INPUT_ARG, pow_spec, return_hat=False)
    k_sq = jnp.square(get_k(N, L))

    psi_x_hat, psi_y_hat, psi_z_hat = get_psi_sc(
        delta_in, D1, k_sq, N, L, MUSCLE, MUSCLE_ITERS, SC_CORRECTION
    )

    delta = from_psi_to_delta(
        psi_x_hat, psi_y_hat, psi_z_hat, N, L, pm_func, psi_has_hat=True
    )

    return delta


def forward_alpt(
    x,
    N,
    L,
    D1,
    D2,
    INPUT_ARG,
    pm_func,
    pow_spec,
    R_S,
    MUSCLE,
    MUSCLE_ITERS,
    SC_CORRECTION,
    PARTICLE_RIDGE=False,
    K_TH_PR=0.4,
    D_PR=0.2,
):

    delta_in_hat = get_delta(x, N, L, INPUT_ARG, pow_spec, return_hat=True)
    k_sq = jnp.square(get_k(N, L))

    psi_x_hat, psi_y_hat, psi_z_hat = get_psi_alpt(
        delta_in_hat,
        D1,
        D2,
        k_sq,
        N,
        L,
        R_S,
        MUSCLE,
        MUSCLE_ITERS,
        SC_CORRECTION,
    )

    delta = from_psi_to_delta(
        psi_x_hat, psi_y_hat, psi_z_hat, N, L, pm_func, psi_has_hat=True
    )

    if PARTICLE_RIDGE:
        k = get_k(N, L)
        sigma = 0.05
        u = (k - K_TH_PR) / (jnp.sqrt(2) * sigma)
        filt_k = jax.lax.erfc(u)*0.5

        # one_over_var_k = K_TH_PR**2
        # filt_k = symmetric_gaussian_nonorm(k_sq, one_over_var_k)

        delta *= D_PR
        delta = jnp.array(delta, dtype=complex)

        # threshold = 3 / 2
        # transition_width = 0.25
        # smooth_factor = jax.nn.sigmoid((threshold - delta) / transition_width)
        # smooth_value = jnp.log1p(jnp.exp(1 - 2 / 3 * delta))
        # div_psi = 3 * smooth_factor * jnp.sqrt(smooth_value) - 3
        div_psi = 3 * jnp.real(1-2/3 * delta) - 3
        div_psi *= 1

        div_psi_hat = my_fft(div_psi, L)
        div_psi_hat_residual = (1 - filt_k) * div_psi_hat

        one_over_k_sq = jnp.where(k > 0, 1 / k**2, 0.0)
        phi_residual_hat = -div_psi_hat_residual * one_over_k_sq

        phi_hat = -div_psi_hat * one_over_k_sq

        psi_res_x_hat, psi_res_y_hat, psi_res_z_hat = get_psi_lpt1(
            -phi_hat, N, L
        )

        psi_x_hat += psi_res_x_hat
        psi_y_hat += psi_res_y_hat
        psi_z_hat += psi_res_z_hat

        delta = from_psi_to_delta(
            psi_x_hat, psi_y_hat, psi_z_hat, N, L, pm_func, psi_has_hat=True
        )

    return delta
