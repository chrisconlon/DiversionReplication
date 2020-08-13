#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 11:47:09 2020

@author: chitra
"""
import pandas as pd
import numpy as np
import pathlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import pyblp

from scipy import  ndimage
from scipy.stats import  percentileofscore

## Header: Import Directories and Define Pats
main_dir = pathlib.Path.cwd().parent
data_dir = main_dir / 'data'

dict_dir = data_dir / 'dict'
raw_dir = data_dir / 'raw'

tab_dir = main_dir / 'tables'
fig_dir = main_dir / 'figures'
pyblp.options.verbose = False

from aux_plot_config import *
from aux_double_plot import plot_double_mpc

from aux_table_functions import draw_blp_agents, load_blp_base


# %% 
# run the best practices
# using the original loading function works for base anyways...

blp_products = pd.read_parquet(raw_dir / 'blp_product_data_opt.parquet')
blp_products['product_ids']=range(0,2217)


# Set draws here
blp_agents=draw_blp_agents(500)
blp_agents['draw_ids']=np.tile(range(0,500),20) 


problem_options = dict(
    product_formulations=(
        pyblp.Formulation('1 + hpwt + air + mpd + space'),
        pyblp.Formulation('1 + prices + hpwt + air + mpd + space'),
        pyblp.Formulation(f'1 + log(hpwt) + air + log(mpg) + log(space) + trend'),
    ),
    agent_formulation=pyblp.Formulation('0 + I(1 / income)'),
    costs_type='log',
    agent_data=blp_agents
)


#sigma_base=np.diag([3.612, 0, 4.628, 1.818, 1.050, 2.056])
#pi_base=np.c_[[0, -43.501, 0, 0, 0, 0]].astype(float)
#beta_base=np.c_[[-6.67862016, -5, 2.7741249, 0.57237907, 0.34009843,3.91954976]]
blp_problem=pyblp.Problem(add_exogenous=False,product_data=blp_products, **problem_options)

filename_blp_base = dict_dir / 'blp_results_base.npy'
# this function only works for base
results_blp = load_blp_base(blp_problem,filename_blp_base)


# %% PLOT THE DOUBLE GRAPHS
## Define some "global"
SIGMA = 10 # larger sigma is smoother curve
N = 200 # number of simulated individuals # not used anywhere
B = 50 # number of bins for the histogram
MO = 'nearest'
MPC_TEXT = 'Marginal Price Change: $ {s_{ij}}\\times {(1- s_{ij})} \\times \\alpha_i$'
QC_TEXT = 'Quality Change: $ {s_{ij}}\\times{(1- s_{ij})} $'
SC_TEXT = 'Second Choice: $ {s_{ij}} $'

PARAMS_INT = [SIGMA, N, B]
PARAMS_STR = [MO, MPC_TEXT, QC_TEXT, SC_TEXT]

# %%
# keep only the most interesting plots for now

plot_double_mpc(results_blp, blp_products, 1990, 53, 113, 19, PARAMS_INT, PARAMS_STR, fig_dir)
plot_double_mpc(results_blp, blp_products, 1990, 9, 8, 75, PARAMS_INT, PARAMS_STR, fig_dir)
plot_double_mpc(results_blp, blp_products, 1990, 8, 9, 113, PARAMS_INT, PARAMS_STR, fig_dir)
