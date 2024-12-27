import os
import struct

import h5py
import numpy as np
import jax
import jax.numpy as jnp

from .fourier import get_one_over_k_sq


directory = os.path.dirname(os.path.abspath(__file__))
dir_cache = directory + '/cache/' 

if not os.path.exists(dir_cache):
    os.makedirs(dir_cache)

def compute_or_load_pow_spec(N, L, Z_I):
    pow_spec_path = dir_cache + f'pow_spec_{N}_{L:0.0f}_{Z_I:0.0f}.npy'
    if os.path.exists(pow_spec_path):
        pow_spec = jnp.array(jnp.load(pow_spec_path))
    else:
        from .jax_cosmo_utils import get_linear_power
        pow_spec = get_linear_power(N, L, Z_I)
        jnp.save(pow_spec_path, pow_spec)
    return pow_spec

def compute_or_load_D1_D2(Z_I, Z_F):
    D1_path = dir_cache + f'D1_{Z_I:0.0f}_{Z_F:0.0f}.npy'
    D2_path = dir_cache + f'D2_{Z_I:0.0f}_{Z_F:0.0f}.npy'

    if os.path.exists(D1_path) and os.path.exists(D2_path):
        D1 = jnp.array(jnp.load(D1_path))
        D2 = jnp.array(jnp.load(D2_path))
    else:
        from .jax_cosmo_utils import get_D1, Omega_m
        D1 = get_D1(Z_I, Z_F)
        D2 = -3 / 7 * Omega_m ** (-1 / 143) * D1**2
        jnp.save(D1_path, jnp.array(D1))
        jnp.save(D2_path, jnp.array(D2))

    return D1, D2

def compute_or_load_one_over_k_sq(N, L):
    one_over_k_sq_path = dir_cache + f'inv_k_sq_{N}_{L:0.0f}.npy'
    if os.path.exists(one_over_k_sq_path):
        one_over_k_sq = jnp.array(jnp.load(one_over_k_sq_path))
    else:
        one_over_k_sq = get_one_over_k_sq(N, L)
        jnp.save(one_over_k_sq_path, one_over_k_sq)
    return one_over_k_sq


def serialize_dic(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item() 
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  
    elif isinstance(obj, jax.Array): 
        return np.array(obj).tolist() 
    elif isinstance(obj, dict):
        return {k: serialize_dic(v) for k, v in obj.items()} 
    elif isinstance(obj, list):
        return [serialize_dic(v) for v in obj] 
    else:
        return obj

def symmetric_gaussian_nonorm(r_sq, one_over_var):
    return jnp.exp(- 0.5 * r_sq * one_over_var)

def err_func_filter():

    return 

