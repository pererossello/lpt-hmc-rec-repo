import jax
import jax.numpy as jnp

def cic(pos, N, L):
    
    """
    Differentiable Cloud in Cell interpolation
    """

    # Scale positions to grid indices
    pos = (pos * N / L) % N  # Positions in [0, N)

    # Continuous indices
    ix_f = pos[:, 0]
    iy_f = pos[:, 1]
    iz_f = pos[:, 2]

    # Integer parts (floors)
    ix0 = jnp.floor(ix_f)
    iy0 = jnp.floor(iy_f)
    iz0 = jnp.floor(iz_f)

    # Fractional parts
    dx = ix_f - ix0
    dy = iy_f - iy0
    dz = iz_f - iz0

    # Wrap around for periodic boundary conditions
    ix0 = ix0 % N
    iy0 = iy0 % N
    iz0 = iz0 % N

    ix1 = (ix0 + 1) % N
    iy1 = (iy0 + 1) % N
    iz1 = (iz0 + 1) % N

    # Convert to integers
    ix0 = ix0.astype(int)
    iy0 = iy0.astype(int)
    iz0 = iz0.astype(int)
    ix1 = ix1.astype(int)
    iy1 = iy1.astype(int)
    iz1 = iz1.astype(int)

    # Weights
    w000 = (1 - dx) * (1 - dy) * (1 - dz)
    w100 = dx * (1 - dy) * (1 - dz)
    w010 = (1 - dx) * dy * (1 - dz)
    w001 = (1 - dx) * (1 - dy) * dz
    w110 = dx * dy * (1 - dz)
    w101 = dx * (1 - dy) * dz
    w011 = (1 - dx) * dy * dz
    w111 = dx * dy * dz

    # Initialize density grid
    density = jnp.zeros((N,) * 3)

    # Accumulate contributions
    density = density.at[ix0, iy0, iz0].add(w000)
    density = density.at[ix1, iy0, iz0].add(w100)
    density = density.at[ix0, iy1, iz0].add(w010)
    density = density.at[ix0, iy0, iz1].add(w001)
    density = density.at[ix1, iy1, iz0].add(w110)
    density = density.at[ix1, iy0, iz1].add(w101)
    density = density.at[ix0, iy1, iz1].add(w011)
    density = density.at[ix1, iy1, iz1].add(w111)

    return density
