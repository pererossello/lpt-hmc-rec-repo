import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FOLD_PATH = "./the_samples"
REC_NAME = "128_FSK_U_ALPT_HPL"
EXP_NAME = "EXP_A"

SAMPLES = [1500,2500, 1000]
TEMP = [128**3, 128**3/100, 1]