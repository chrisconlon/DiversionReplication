"""
Goal: Run many Nevo cases, save results to a dict
to access these results later
"""

import pyblp
import numpy as np
import pandas as pd
import pathlib

main_dir = pathlib.Path.cwd().parent
data_dir = main_dir / 'data'

dict_dir = data_dir / 'dict'
raw_dir = data_dir / 'raw'

pyblp.options.digits = 2
pyblp.options.verbose = False

from aux_nevo_cases import get_nevo_base, get_nevo_nocons, get_nevo_noalpha, get_nevo_triple, get_nevo_logit, get_nevo_nested
from aux_blp_cases import get_blp_base, get_blp_nocons, get_blp_noalpha, get_blp_triple, get_blp_logit, get_blp_nested

# %%
# %%
# Base
# %%
# Comment this out if you want
results_nevo_nested = get_nevo_nested()
results_nevo_logit  = get_nevo_logit()
results_nevo_triple = get_nevo_triple()
results_nevo_nocons = get_nevo_nocons()
results_nevo_noalpha = get_nevo_noalpha()
results_nevo_base   = get_nevo_base()

# %%
# Comment this out after running it once
results_blp_nested = get_blp_nested()
results_blp_logit  = get_blp_logit()
results_blp_triple = get_blp_triple()
results_blp_nocons = get_blp_nocons()
results_blp_noalpha = get_blp_noalpha()
results_blp_base   = get_blp_base()