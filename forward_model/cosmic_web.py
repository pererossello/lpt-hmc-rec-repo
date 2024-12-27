import jax.numpy as jnp

from .fourier import my_fft, get_one_over_k_sq
from .calculus import hessian_fourier
from .algebra import eigenvals

def get_phi_delta_web_classification(delta_dm, N, L, lambda_th):

    def get_web(field):
        hessian = hessian_fourier(field, N, L)
        lambdas = eigenvals(hessian)
        web = lambda_cassify(lambdas, N, lambda_th)
        return web
    
    # put hat to delta_dm
    delta_dm = my_fft(delta_dm, L)
    
    # delta_web 
    delta_web = get_web(-delta_dm)

    #phi_web
    kernel = get_one_over_k_sq(N, L)
    phi_web = get_web(- delta_dm * kernel)

    cosmic_web = 4*phi_web + delta_web

    return cosmic_web


def lambda_cassify(lambdas, N, lambda_th):

    cw_class = jnp.empty((N, N, N), dtype=jnp.int16)

    knot_bool = (
        (lambdas[..., 0] >= lambda_th)
        & (lambdas[..., 1] >= lambda_th)
        & (lambdas[..., 2] > lambda_th)
    )
    void_bool = (
        (lambdas[..., 0] < lambda_th)
        & (lambdas[..., 1] <= lambda_th)
        & (lambdas[..., 2] <= lambda_th)
    )

    filament_bool = (
        (lambdas[..., 0] >= lambda_th)
        & (lambdas[..., 1] >= lambda_th)
        & (lambdas[..., 2] <= lambda_th)
    )
    sheet_bool = (
        (lambdas[..., 0] >= lambda_th)
        & (lambdas[..., 1] < lambda_th)
        & (lambdas[..., 2] <= lambda_th)
    )

    cw_class = jnp.where(void_bool, 0, cw_class)
    cw_class = jnp.where(sheet_bool, 1, cw_class)
    cw_class = jnp.where(filament_bool, 2, cw_class)
    cw_class = jnp.where(knot_bool, 3, cw_class)

    return cw_class

