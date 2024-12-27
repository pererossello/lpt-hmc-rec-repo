import jax
import jax.numpy as jnp
import jax_cosmo

from forward_model.fourier import get_k

h = 0.673
Omega_c = 0.1200 / h**2
Omega_b = 0.02237 / h**2
Omega_m = Omega_c + Omega_b

Cosmology = jax_cosmo.Cosmology(
    Omega_c=Omega_c,
    Omega_b=Omega_b,
    h=h,
    sigma8=0.807952,
    n_s=0.9649,
    Omega_k=0.0,
    w0=-1.0,
    wa=0.0,
)

def get_D1(z_i, z_f):

    a_i = jnp.atleast_1d(1.0 / (1 + z_i))
    a_f = jnp.atleast_1d(1.0 / (1 + z_f))

    D1_i = jax_cosmo.background.growth_factor(Cosmology, a_i)
    D1_f = jax_cosmo.background.growth_factor(Cosmology, a_f)

    D1 = ((D1_f/D1_i)[0])

    return D1

def get_linear_power(N, L, z_i):
    a = 1 / (1+z_i)
    k = get_k(N, L)
    power_spectra = jax_cosmo.power.linear_matter_power(Cosmology, k, a=a)
    return power_spectra

