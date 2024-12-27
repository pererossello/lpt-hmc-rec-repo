import jax
import jax.numpy as jnp

from .fourier import get_k_1D, get_k_rfft_1D, my_fft, my_ifft


def der_i_hat(array_hat, axis, k_1D):
    if axis == 0:
        k_ = k_1D[:, None, None]
    elif axis == 1:
        k_ = k_1D[None, :, None]
    elif axis == 2:
        k_ = k_1D[None, None, :]
    der_hat = 1j * k_ * array_hat
    return der_hat

def gradient_hat(array_hat, N, L):
    k_1D = get_k_1D(N, L)
    k_1D_r = get_k_rfft_1D(N, L)

    arr_hat_dx = 1j * k_1D[:, None, None] * array_hat
    arr_hat_dy = 1j * k_1D[None, :, None] * array_hat
    arr_hat_dz = 1j * k_1D_r[None, None, :] * array_hat
    return arr_hat_dx, arr_hat_dy, arr_hat_dz

def divergence_hat(arr_hat_x, arr_hat_y, arr_hat_z, N, L):
    k_1D = get_k_1D(N, L)
    k_1D_r = get_k_rfft_1D(N, L)

    k_x = k_1D[:, None, None]
    k_y = k_1D[None, :, None]
    k_z = k_1D_r[None, None, :]

    div_arr_hat = arr_hat_x*k_x+arr_hat_y*k_y+arr_hat_z*k_z
    div_arr_hat *= 1j

    return div_arr_hat



def hessian_fourier(array_hat, N, L):

    hessian = jnp.empty((N, N, N, 3, 3))

    k_1D = get_k_1D(N, L)
    k_1D_r = get_k_rfft_1D(N, L)
    k_0 = k_1D[:, None, None]
    k_1 = k_1D[None, :, None]
    k_2 = k_1D_r[None, None, :]

    val_00 = my_ifft(-array_hat*k_0**2, L).real
    hessian = hessian.at[..., 0,0].set(val_00)
    del val_00

    val_11 = my_ifft(-array_hat*k_1**2, L).real
    hessian = hessian.at[..., 1,1].set(val_11)
    del val_11

    val_22 = my_ifft(-array_hat*k_2**2, L).real
    hessian = hessian.at[..., 2,2].set(val_22)
    del val_22

    val_01 = my_ifft(-array_hat*k_0*k_1, L).real
    hessian = hessian.at[..., 1,0].set(val_01)
    hessian = hessian.at[..., 0,1].set(val_01)
    del val_01

    val_02 = my_ifft(-array_hat*k_0*k_2, L).real
    hessian = hessian.at[..., 2,0].set(val_02)
    hessian = hessian.at[..., 0,2].set(val_02)
    del val_02

    val_12 = my_ifft(-array_hat*k_1*k_2, L).real
    hessian = hessian.at[..., 1,2].set(val_12)
    hessian = hessian.at[..., 2,1].set(val_12)
    del val_12

    return hessian

def hessian_fourier_nohat_input(array, N, L):
    array_hat = my_fft(array, L)
    return hessian_fourier(array_hat, N, L)

