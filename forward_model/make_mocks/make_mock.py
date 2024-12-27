# run it with
# prossello@geronimo:~/Documents/TFM/code_v2/lpt-hmc-recs-repo$
# python -m forward_model.make_mocks.make_mock

import os
import sys
from functools import partial
import json
import shutil
from pathlib import Path
import datetime

import h5py
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from ..main import get_forward_model
from ..fourier import my_fft, my_ifft
from ..utils import serialize_dic
from ..plot_utils import plot_cubes
from ..utils import compute_or_load_pow_spec
from ..ics import get_delta

directory = str(Path(__file__).resolve().parent.parent) + "/"

N, L, Z_I, Z_F = 128, 600, 99, 0
N_TR = 1e7
SEED_INT = 1

BIAS_MODEL = "POWER_LAW"
ALPHA = 1.5
BIAS_PARAMS = {"ALPHA": ALPHA}

BIAS_MODEL = 'HIERARCHICAL_POWER_LAW'
key_ = jax.random.PRNGKey(1)
alph_min, alph_max = 1, 2
ALPHAS = jax.random.uniform(key_, (16,))*(alph_max-alph_min)+alph_min
BIAS_PARAMS = {"ALPHA": ALPHAS, "LAMBDA_TH": 0.05}

IC_KIND = "FSK_U"
LPT_METHOD = "ALPT"
R_S = 1
argdic = {
    "N": N,
    "L": L,
    "Z_I": Z_I,
    "Z_F": Z_F,
    "LPT_METHOD": LPT_METHOD,
    "PM_METHOD": "CIC",
    "IC_KIND": IC_KIND,
    "MUSCLE": True,
    "MUSCLE_ITERS": "ONE",
    "SC_CORRECTION": True,
    "R_S": R_S,
    "BIAS_MODEL": BIAS_MODEL,
    "BIAS_PARAMS": BIAS_PARAMS,
    "N_TR": N_TR,
    "SAMPLE": 1,
}

forward_model = get_forward_model(argdic)
forward_model = jax.jit(forward_model)

key = jax.random.PRNGKey(SEED_INT)

SHAPE = (N,) * 3

if IC_KIND in ["U", "FSK_U"]:
    u = jax.random.normal(key, shape=SHAPE)
elif IC_KIND == "DELTA":
    pow_spec = compute_or_load_pow_spec(N, L, Z_I)
    u = jax.random.normal(key, shape=SHAPE)
    u = get_delta(u, N, L, "FSK_U", pow_spec, return_hat=False)


n_tr = forward_model(u)

print("n_tr min", n_tr.min())
print("n_tr max", n_tr.max())
print("n_tr sum", n_tr.sum())

num_zeros = jnp.sum(n_tr == 0.0)
print(f"% zeros {100*num_zeros/N**3:0.2f}")


if BIAS_MODEL == "POWER_LAW":
    bias_tag = "PL"
elif BIAS_MODEL == "HIERARCHICAL_POWER_LAW":
    bias_tag = "HPL"

savefold = directory + f"results/"
if not os.path.exists(savefold):
    os.makedirs(savefold)

# Save `n_tr` as HDF5
filename = f"crime_{IC_KIND}_{N}_{L}_{LPT_METHOD}_{bias_tag}.hdf5"
with h5py.File(savefold + filename, "w") as h5_file:

    header = h5_file.create_group("Header")

    header.attrs["N"] = N
    header.attrs["L"] = L
    header.attrs["Z_I"] = Z_I
    header.attrs["Z_F"] = Z_F
    header.attrs["IC_KIND"] = IC_KIND

    header.attrs["DATA_KIND"] = "REF_DATA"
    header.attrs["INVERSE_CRIME"] = jnp.bool(1)


    h5_file.attrs["FM_ARGDIC"] = json.dumps(serialize_dic(argdic))
    h5_file.create_dataset("input", data=u)
    h5_file.create_dataset("n_tr", data=n_tr)


# Save `argdic` as JSON
# with open(savefold + "argdic.json", "w") as json_file:
#     json.dump(serialize_dic(argdic), json_file, indent=4)

idx = N // 2
axis = 2
width = 10

plot_cubes(
    [n_tr], cmap="gnuplot2", vlim=(0, n_tr.max() / 10), idx=idx, axis=axis, width=width
)
plt.show()
