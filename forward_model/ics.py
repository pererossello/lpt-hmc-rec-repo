import jax.numpy as jnp

from .fourier import my_fft, my_ifft, make_rfft3_arr_from_N_cube_numbers
from .utils import compute_or_load_pow_spec

def get_delta(x, N, L, INPUT_ARG, pow_spec, return_hat):
    """
    Returns delta hat

    :x: input array
    """

    # GAUSSIAN_NCUBE_NUMBERS_FLAT
    if INPUT_ARG == "U":
        # x should have shape (N**3)
        val = make_rfft3_arr_from_N_cube_numbers(x, N)
        fact = 0.25 * L**3
        delta_in = val * jnp.sqrt(fact * pow_spec)
        if not return_hat:
            delta_in = my_ifft(delta_in, L)

    elif INPUT_ARG == "DELTA":
        delta_in = x
        if return_hat:
            delta_in = my_fft(delta_in, L)
            
    elif INPUT_ARG == 'FSK_U':
        # x should have shape (N,)*3
        fact = N**3 / L**3 * 0.5
        delta_in = my_fft(x, L) * jnp.sqrt(pow_spec*fact)
        if not return_hat:
            delta_in = my_ifft(delta_in, L)
        
        

    return delta_in


