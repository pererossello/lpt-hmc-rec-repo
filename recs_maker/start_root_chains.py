import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from recs.utils import *
from recs.main_root import HMCRootSampler
from recs.hmc_config import HMCRootConfig
from recs.hmc_aux_config import HMCAuxConfig

FOLD_PATH = "./the_samples"
REC_NAME = "64_FSK_U_LPT1_PL"
EXP_NAME = "EXP_A"

DATAPATH = "./forward_model/results/crime_FSK_U_64_200_LPT1_PL.hdf5"
fm_config = None  # specified in datapath (inverse crime)

N_CHAINS = 3
SAMPLES = [2000, 1000, 1000]
TEMP = [64**3, 64**3/10, 1]

LF_METHOD = "STANDARD_REV"
LF_STEPS = [25, 100, 150]
lf_step_fact = [0.05, 0.25, 0.5]
LF_STEP_MIN = 1e-4
LF_STEP_MAX = 1e-1

hmc_config = HMCRootConfig(
    n_chains=N_CHAINS,
    initial_state="RANDOM",
    seed_int_is=2,
    seed_int_chain=1,
    samples=SAMPLES,
    temperature=TEMP,
    lf_method=LF_METHOD,
    lf_step_min=[a * LF_STEP_MIN for a in lf_step_fact],
    lf_step_max=[a * LF_STEP_MAX for a in lf_step_fact],
    lf_steps=LF_STEPS,
)
hmc_aux_config = HMCAuxConfig(
    preset="default", save_n_last=100, print_step=5, plot_step=250
)

sampler = HMCRootSampler(
    REC_NAME, EXP_NAME, hmc_config, hmc_aux_config, fold_path=FOLD_PATH
)
sampler.set_data(DATAPATH)
sampler.set_fm_argdic(fm_config)  # in this case does nothing
sampler.set_up_hmc()
sampler.read_or_make_initial_states()
sampler.run_hmc()
# sampler.postprocess()
