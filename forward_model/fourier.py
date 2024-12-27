import jax
import jax.numpy as jnp


fourier_norm = "forward"


def my_fft(f, L):
    f_k = L**3 * jnp.fft.rfftn(f, norm=fourier_norm)
    return f_k

def my_ifft(f_k, L):
    f = (1 / L**3) * jnp.fft.irfftn(f_k, norm=fourier_norm)
    return f

def get_k_1D(N, L):
    k_1D = jnp.fft.fftfreq(N) * N * 2 * jnp.pi / L
    return k_1D

def get_k_rfft_1D(N, L):
    k_1D = jnp.fft.rfftfreq(N) * N * 2 * jnp.pi / L
    return k_1D

def get_k(N, L):
    k_1D = get_k_1D(N, L)
    k_r_1D = get_k_rfft_1D(N, L)
    k_list = [k_1D] * 2 + [k_r_1D]
    k = jnp.linalg.norm(jnp.array(jnp.meshgrid(*k_list, indexing="ij")), axis=0)
    return k

def get_one_over_k_sq(N, L):
    k_sq = jnp.square(get_k(N, L))
    one_over_k_sq = jnp.where(k_sq > 0, 1 / k_sq, 0.0)
    return one_over_k_sq

def get_one_over_k(N, L):
    k = get_k(N, L)
    inv_k = jnp.where(k == 0, 1.0, 1 / k)
    return inv_k

def get_sinc_sq(N):
    x_1D = jnp.fft.rfftfreq(N)
    x = jnp.linalg.norm(jnp.array(jnp.meshgrid(*[x_1D] * 3, indexing="ij")), axis=0)
    return jnp.sinc(x) ** 2


##########    
# The rather cryptic functions below are oriented towards understanding and
# generating a valid fourier transformed array of a real array, i.e., 
# of the type `arr_hat = jnp.fft.rfft(some_3d_real_arr)`,  
# using any N^3 numbers
# I.e. give me N^3 numbers, and I give you a valid 3d array in fourier space 
# (which will have shape (N,N,N//2+1) )
# whose fourier transform is a real array. 
##########


def from_rfft2_arr_to_fft2_arr(arr_hat_r, N):

    arr_hat_rec = jnp.zeros((N, N), dtype=complex)
    if arr_hat_r.shape != (N, N // 2 + 1):
        raise ValueError("Wrong shape! Should be (N,N//2+1)")

    arr_hat_rec = arr_hat_rec.at[:, : N // 2 + 1].set(arr_hat_r)
    arr_hat_rec = arr_hat_rec.at[0, -N // 2 + 1 :].set(
        jnp.conj(arr_hat_r[0, 1 : N // 2][::-1])
    )
    arr_hat_rec = arr_hat_rec.at[1:, -N // 2 + 1 :].set(
        jnp.conj(arr_hat_r[1:, 1 : N // 2][::-1, ::-1])
    )

    return arr_hat_rec


def make_rfft2_arr_from_N_sq_numbers(numbers, N):

    arr_hat_r = jnp.zeros((N, N // 2 + 1), dtype=complex)

    if numbers.shape != (N**2,):
        raise ValueError("Wrong shape! Should be (N^2,)")

    # make adequate shape arrs

    NYQ_vals = numbers[:4]
    numbers = numbers[4:]

    yellow_vals = numbers[: N - 2]
    yellow_vals = yellow_vals.reshape((N // 2 - 1, 2))
    numbers = numbers[N - 2 :]

    green_vals = numbers[: 2 * (N - 2)]
    green_vals = green_vals.reshape((N // 2 - 1, 2, 2))
    numbers = numbers[2 * (N - 2) :]

    blue_vals = numbers
    blue_vals = blue_vals.reshape((N - 1, N // 2 - 1, 2))

    # ASSIGN VALUES

    # place nyq modes (four of them and all real)
    nyq_idxs = jnp.array([(0, 0), (N // 2, 0), (0, N // 2), (N // 2, N // 2)])
    arr_hat_r = arr_hat_r.at[tuple(nyq_idxs.T)].set(NYQ_vals)

    # place yellow modes (complex)
    arr_hat_r = arr_hat_r.at[0, 1 : N // 2].set(
        yellow_vals[:, 0] + 1j * yellow_vals[:, 1]
    )

    # place green modes (complex)
    arr_hat_r = arr_hat_r.at[1 : N // 2, 0].set(
        green_vals[:, 0, 0] + 1j * green_vals[:, 1, 0]
    )
    arr_hat_r = arr_hat_r.at[1 : N // 2, N // 2].set(
        green_vals[:, 0, 1] + 1j * green_vals[:, 1, 1]
    )

    # place pink modes (copmlex conjugates of green modes)
    arr_hat_r = arr_hat_r.at[-N // 2 + 1 :, 0].set(
        jnp.conj(arr_hat_r[1 : N // 2, 0][::-1])
    )
    arr_hat_r = arr_hat_r.at[-N // 2 + 1 :, N // 2].set(
        jnp.conj(arr_hat_r[1 : N // 2, N // 2][::-1])
    )

    # blue modes
    arr_hat_r = arr_hat_r.at[1:, 1 : N // 2].set(
        blue_vals[:, :, 0] + 1j * blue_vals[:, :, 1]
    )

    return arr_hat_r

def make_rfft3_arr_from_N_cube_numbers(numbers, N):

    numbers = numbers.reshape(-1)

    arr_hat_r = jnp.zeros((N, N, N // 2 + 1), dtype=complex)

    if numbers.shape != (N**3,):
        # now this is redundant because of reshape above
        raise ValueError("Wrong shape! Should be (N^3,)")

    M_main = N**3 - 2 * N**2
    blue_vals = numbers[:M_main]
    blue_vals = blue_vals.reshape((N, N, N // 2 - 1, 2))
    numbers = numbers[M_main:]

    plane_1_vals = numbers[: N**2]
    numbers = numbers[N**2 :]

    plane_2_vals = numbers[: N**2]

    # ASSIGN
    arr_hat_r = arr_hat_r.at[:, :, 1 : N // 2].set(
        blue_vals[..., 0] + 1j * blue_vals[..., 1]
    )

    plane_1 = make_rfft2_arr_from_N_sq_numbers(plane_1_vals, N)
    plane_1 = from_rfft2_arr_to_fft2_arr(plane_1, N)
    arr_hat_r = arr_hat_r.at[:, :, 0].set(plane_1)

    plane_2 = make_rfft2_arr_from_N_sq_numbers(plane_2_vals, N)
    plane_2 = from_rfft2_arr_to_fft2_arr(plane_2, N)
    arr_hat_r = arr_hat_r.at[:, :, N // 2].set(plane_2)

    return arr_hat_r



