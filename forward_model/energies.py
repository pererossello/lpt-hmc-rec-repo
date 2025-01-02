from functools import partial
import json
import os
import sys

import h5py
import jax
import jax.numpy as jnp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from forward_model.fourier import my_fft, my_ifft


def nll(q, n_tr_ref, forward_model, epsilon=1e-10):
    n_tr = forward_model(q)
    n_tr = jnp.where(n_tr == 0.0, epsilon, n_tr)
    nll = -jnp.sum(n_tr_ref * jnp.log(n_tr))
    return nll


def get_nlp(IC_KIND, L, INV_L3, inv_pow_spec):
    if IC_KIND in ["U", "FSK_U"]:
        nlp = nlp_u
    elif IC_KIND in ["DELTA"]:
        nlp = partial(nlp_delta, L=L, INV_L3=INV_L3, inv_pow_spec=inv_pow_spec)
    return nlp

def get_kinetic(IC_KIND, L, INV_L3, pow_spec):
    if IC_KIND in ["U", "FSK_U"]:
        kinetic = kinetic_u
    elif IC_KIND in ["DELTA"]:
        kinetic = partial(kinetic_delta, L=L, INV_L3=INV_L3, pow_spec=pow_spec)
    return kinetic


#####
def nlp_u(q):
    return 0.5 * jnp.sum(jnp.abs(q) ** 2)


def kinetic_u(p):
    return 0.5 * jnp.sum(jnp.abs(p) ** 2)


def sample_momenta_u(key, N):
    p = jax.random.normal(key, (N,) * 3)
    return p


def make_is_u(key, N):
    q = jax.random.normal(key, (N,) * 3)
    return q


#####


#####
def nlp_delta(q, L, INV_L3, inv_pow_spec):
    fact = 0.5 * INV_L3
    q = my_fft(q, L)
    q = jnp.abs(q) ** 2
    q = q * inv_pow_spec
    return fact * jnp.sum(q)


def kinetic_delta(p, L, INV_L3, pow_spec):
    fact = 0.5 * INV_L3
    p = my_fft(p, L)
    p = jnp.abs(p) ** 2
    p = p * pow_spec
    return fact * jnp.sum(p)


def sample_momenta_delta(key, N, L, N3, INV_L3, inv_pow_spec):
    p = jax.random.normal(key, (N,) * 3)
    fact = N3 * INV_L3 * 0.5
    p = my_fft(p, L) * jnp.sqrt(fact * inv_pow_spec)
    return my_ifft(p, L)


def make_is_delta(key, N, L, pow_spec):
    q = jax.random.normal(key, (N,) * 3)
    fact = 0.5 * N**3 / L**3
    q = my_fft(q, L) * jnp.sqrt(fact * pow_spec)
    return my_ifft(q, L)


#####
