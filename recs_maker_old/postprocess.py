import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from recs.plot_utils import plot_root_interchain
from recs.stats import save_statistics_to_spreadsheets

EXP_DIR = "./the_samples/64_FSK_U_ALPT_HPL/EXP_A"
# EXP_DIR = "./the_samples/128_FSK_U_LPT1_PL/EXP_AR"

plot_root_interchain(EXP_DIR, show=True)
# plot_statistics_tables(EXP_DIR)
# save_statistics_to_spreadsheets(EXP_DIR)
