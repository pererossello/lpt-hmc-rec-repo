import jax
import jax.numpy as jnp

def eigenvals(matrix):

    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m12 = matrix[..., 1, 2]

    shape = m00.shape

    a = -1 * jnp.ones(shape)

    # trace
    b = m00 + m11 + m22

    # Sum of the products of 2x2 minors
    min01 = m00 * m11 - m01**2
    min02 = m00 * m22 - m02**2
    min12 = m11 * m22 - m12**2
    c = -(min01 + min02 + min12)

    # determinant
    d = (
        m00 * (m11 * m22 - m12**2)
        - m01 * (m01 * m22 - m12 * m02)
        + m02 * (m01 * m12 - m11 * m02)
    )

    roots = roots_cubic_equation(a, b, c, d)

    roots = jnp.sort(roots, axis=-1)[..., ::-1]
    roots = roots.real

    return roots


def roots_cubic_equation(a, b, c, d):

    Delta0 = b**2 - 3 * a * c
    Delta1 = 2 * b**3 - 9 * a * b * c + 27 * a**2 * d

    discriminant = (Delta1**2 - 4 * Delta0**3).astype(jnp.complex64)

    C = ((Delta1 + discriminant ** (1 / 2)) / 2) ** (1 / 3)

    xis = jnp.array([1, (-1 + jnp.sqrt(3) * 1j) / 2, (-1 - jnp.sqrt(3) * 1j) / 2])

    C_xis = C[..., None] * xis[..., :]

    roots = (
        -1
        / (3 * a[..., None] * C_xis)
        * (b[..., None] * C_xis + C_xis**2 + Delta0[..., None])
    )

    return roots
