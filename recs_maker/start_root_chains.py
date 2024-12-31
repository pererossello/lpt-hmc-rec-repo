import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


FOLD_PATH = "./the_samples"
REC_NAME = "128_FSK_U_ALPT_HPL"
GROUND_NAME = "EXP_A"

DATAPATH = "./forward_model/results/crime_FSK_U_128_500_ALPT_HPL.hdf5"
fm_config = None  # dic, path to json , or str specifying 'INVERSE_CRIME' which will read from data





# N_CHAINS = 3
# SAMPLES = [1500,2500, 1000]
# TEMP = [128**3, 128**3/100, 1]

# LF_METHOD = "STANDARD_REV"
# LF_STEPS = [7, 50, 50]
# lf_step_fact = [0.5, 0.25, 0.25]
# LF_STEP_MIN = 1e-4
# LF_STEP_MAX = 1e-2

# hmc_config = HMCRootConfig(
#     n_chains=N_CHAINS,
#     initial_state="RANDOM",
#     seed_int_is=1,
#     seed_int_chain=1,
#     samples=SAMPLES,
#     temperature=TEMP,
#     lf_method=LF_METHOD,
#     lf_step_min=[a * LF_STEP_MIN for a in lf_step_fact],
#     lf_step_max=[a * LF_STEP_MAX for a in lf_step_fact],
#     lf_steps=LF_STEPS,
# )
# hmc_aux_config = HMCAuxConfig(
#     preset="default", save_n_last=100, print_step=5, plot_step=50
# )

# sampler = HMCRootSampler(
#     REC_NAME, EXP_NAME, hmc_config, hmc_aux_config, fold_path=FOLD_PATH
# )
# sampler.set_data(DATAPATH)
# sampler.set_fm_argdic(fm_config)  # in this case does nothing
# sampler.set_up_hmc()
# sampler.read_or_make_initial_states()
# sampler.run_hmc()

# EXP_DIR = os.path.join(sampler.dir_manager.working_directory, EXP_NAME)
# plot_root_interchain(EXP_DIR, n_last=5, n_bins=50, show=True)
# save_statistics_to_spreadsheets(EXP_DIR)
