import sys
import os
from functools import partial

import jax
import jax.numpy as jnp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from forward_model.energies import nll
from forward_model.energies import nlp_u, kinetic_u, sample_momenta_u, make_is_u
from forward_model.energies import nlp_delta, kinetic_delta, sample_momenta_delta, make_is_delta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from forward_model.main import get_forward_model as get_fm
from forward_model.utils import compute_or_load_pow_spec
from forward_model.utils import compute_or_load_D1_D2
from hmc.metropolis import metropolis_hastings
from hmc.leapfrog import (
    leapfrog_high_order,
    leapfrog_second_order,
    leapfrog_second_order_rev,
)

