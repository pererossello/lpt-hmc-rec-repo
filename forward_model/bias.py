from functools import partial

import jax
import jax.numpy as jnp

from .cosmic_web import get_phi_delta_web_classification


def get_forward_bias(N, L, N_TR, BIAS_MODEL, BIAS_PARAMS, SAMPLE):

    if BIAS_MODEL == "POWER_LAW":
        ALPHA = BIAS_PARAMS["ALPHA"]
        bias_func = partial(
            bias_power_law,
            N_TR=N_TR,
            ALPHA=ALPHA,
        )

    if BIAS_MODEL == "HIERARCHICAL_POWER_LAW":
        ALPHAS = jnp.array(BIAS_PARAMS["ALPHA"])
        LAMBDA_TH = BIAS_PARAMS["LAMBDA_TH"]
        bias_func = partial(
            hierarchical_bias_power_law,
            N=N,
            L=L,
            N_TR=N_TR,
            LAMBDA_TH=LAMBDA_TH,
            ALPHAS=ALPHAS,
        )

    if BIAS_MODEL == "HIERARCHICAL_POWER_LAW_AND_THRESHOLDS":
        LAMBDA_TH = BIAS_PARAMS["LAMBDA_TH"]
        ALPHAS = jnp.array(BIAS_PARAMS["ALPHAS"])
        E_LOW = jnp.array(BIAS_PARAMS["E_LOW"])
        RHO_LOW = jnp.array(BIAS_PARAMS["RHO_LOW"])
        E_HIGH = jnp.array(BIAS_PARAMS["E_HIGH"])
        RHO_HIGH = jnp.array(BIAS_PARAMS["RHO_HIGH"])
        bias_func = partial(
            hierarchical_bias_power_law_and_thresholds,
            N=N,
            L=L,
            N_TR=N_TR,
            LAMBDA_TH=LAMBDA_TH,
            ALPHAS=ALPHAS,
            E_LOW=E_LOW,
            RHO_LOW=RHO_LOW,
            E_HIGH=E_HIGH,
            RHO_HIGH=RHO_HIGH,
        )

    if SAMPLE is not False:
        # Then sample is an integer for defining the PRNG seed
        key = jax.random.PRNGKey(SAMPLE)

        bias_func_ = bias_func

        def bias_func(delta_dm):
            n_tr_mean = bias_func_(delta_dm=delta_dm)
            return jax.random.poisson(key, n_tr_mean)

    return bias_func


def bias_power_law(delta_dm, N_TR, ALPHA):

    n_tr = (1 + delta_dm) ** ALPHA
    n_tr *= N_TR / jnp.sum(n_tr)

    return n_tr


def hierarchical_bias_power_law(delta_dm, N, L, N_TR, LAMBDA_TH, ALPHAS):
    cosmic_web = get_phi_delta_web_classification(delta_dm, N, L, LAMBDA_TH)

    alpha = ALPHAS[cosmic_web]  # JAX advanced indexing

    n_tr = (1 + delta_dm) ** alpha
    n_tr *= N_TR / jnp.sum(n_tr)

    return n_tr


def hierarchical_bias_power_law_and_thresholds(
    delta_dm, N, L, N_TR, LAMBDA_TH, ALPHAS, E_LOW, RHO_LOW, E_HIGH, RHO_HIGH
):
    """
    E_LOW > 0
    E_HIGH < 0
    RHO_LOW & RHO_HIGH > 0
    """

    def exp_filter(y, e, rho):
        return jnp.exp(-jnp.power(y / rho, e))

    cosmic_web = get_phi_delta_web_classification(delta_dm, N, L, LAMBDA_TH)

    alpha = ALPHAS[cosmic_web]  # JAX advanced indexing
    e_low = E_LOW[cosmic_web]
    rho_low = RHO_LOW[cosmic_web]
    e_high = E_HIGH[cosmic_web]
    rho_high = RHO_HIGH[cosmic_web]

    rho_dm = 1 + delta_dm
    n_tr = rho_dm**alpha

    n_tr = n_tr * exp_filter(rho_dm, e_low, rho_low)  # low pass filter
    n_tr = n_tr * exp_filter(rho_dm + 1e-10, e_high, rho_high)  # high pass filter

    n_tr *= N_TR / jnp.sum(n_tr)

    # note: for some reason teh addition of a low pass filter makes nans emerge in the derviatives.

    return n_tr
