import sys

import jax
import jax.numpy as jnp


from .fourier import my_fft, my_ifft, get_k

def get_pdf(n_tr, bin_edges):
    hist, _ = jnp.histogram(n_tr, bins=bin_edges)
    return hist

def get_pow_spec_1D(delta, L, n_bins, sphere_only=True):

    N = delta.shape[0]

    fact = 1 / jnp.sqrt(3) if sphere_only else 1.0

    delta_hat = my_fft(delta, L)
    pk = jnp.abs(delta_hat)**2 / L**3

    k = get_k(N, L)

    pk = pk.flatten() 
    k = k.flatten()

    k_bins = jnp.linspace(k.min(), k.max()*fact, n_bins+1)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])

    pk_1D, _ = jnp.histogram(k, bins=k_bins, weights=pk)
    N_k, _ = jnp.histogram(k, bins=k_bins)
    pk_1D /= N_k  # Average over modes

    return k_centers, pk_1D

def get_cross_power_spec_1D(delta_1, delta_2, L, n_bins, sphere_only=True):

    N = delta_1.shape[0]

    fact = 1 / jnp.sqrt(3) if sphere_only else 1.0

    k = get_k(N, L)
    k = k.flatten()
    k_bins = jnp.linspace(k.min(), k.max()*fact, n_bins+1)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    N_k, _ = jnp.histogram(k, bins=k_bins)

    def get_k_norm(arr):
        arr = arr.flatten() 
        arr_1D, _ = jnp.histogram(k, bins=k_bins, weights=arr)
        arr_1D /= N_k  # Average over modes
        return arr_1D

    delta_1_hat = my_fft(delta_1, L)
    delta_2_hat = my_fft(delta_2, L)

    numerator = (delta_1_hat * jnp.conj(delta_2_hat)) / L**3
    numerator = get_k_norm(numerator)
    numerator = numerator.real


    pk1 = get_k_norm(jnp.abs(delta_1_hat)**2/L**3)
    pk2 = get_k_norm(jnp.abs(delta_2_hat)**2/L**3)

    cross = numerator/jnp.sqrt(pk1*pk2)

    return k_centers, cross

def get_reduced_bispectrum(delta, L, k1, k2, thetas):

    delta_hat = my_fft(delta, L)

    N = delta_hat.shape[0]
    NTHETA = thetas.shape[0]

    k3s = jnp.sqrt((k2 * jnp.sin(thetas)) ** 2 + (k2 * jnp.cos(thetas) + k1) ** 2)

    pow_spec = (delta_hat * jnp.conj(delta_hat)).real / L**3
    k = get_k(N, L)
    k_half_width = 2 * jnp.pi / L

    def pk_at_k(k_val):
        k_weights = jnp.where((k < k_val + k_half_width) & (k > k_val - k_half_width), 1.0, 0.0)
        total_weight = jnp.sum(k_weights)
        pow_spec_val = jnp.sum(pow_spec * k_weights) / total_weight
        return pow_spec_val

    pk1 = pk_at_k(k1)
    pk2 = pk_at_k(k2)

    # Precompute pk3s
    pk3s = jax.vmap(pk_at_k)(k3s)

    k1_mask = (k < k1 + k_half_width) & (k > k1 - k_half_width)
    k2_mask = (k < k2 + k_half_width) & (k > k2 - k_half_width)

    delta_hat_k1 = delta_hat * k1_mask
    delta_k1 = my_ifft(delta_hat_k1, L)
    i_k1 = my_ifft(k1_mask, L=L)

    delta_hat_k2 = delta_hat * k2_mask
    delta_k2 = my_ifft(delta_hat_k2, L)
    i_k2 = my_ifft(k2_mask, L=L)

    def body_func(carry, i):
        red_bispec = carry

        k3 = k3s[i]
        pk3 = pk3s[i]

        k3_mask = (k < k3 + k_half_width) & (k > k3 - k_half_width)
        delta_hat_k3 = delta_hat * k3_mask
        delta_k3 = my_ifft(delta_hat_k3, L)
        i_k3 = my_ifft(k3_mask, L=L)

        bispec_val = jnp.nansum(delta_k1 * delta_k2 * delta_k3)
        denominator = pk1 * pk2 + pk1 * pk3 + pk2 * pk3
        NTRIANG = jnp.sum(i_k1 * i_k2 * i_k3)

        red_bispec_val = bispec_val / denominator 
        red_bispec_val /= NTRIANG
        red_bispec_val /= L**3  

        red_bispec = red_bispec.at[i].set(red_bispec_val)

        return red_bispec, None  # No need to accumulate outputs

    # Initialize the carry
    red_bispec_init = jnp.zeros(NTHETA)

    # Set up the indices over which to loop
    indices = jnp.arange(NTHETA)

    # Use lax.scan to perform the loop
    red_bispec_final, _ = jax.lax.scan(body_func, red_bispec_init, indices)

    return red_bispec_final




