#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  30 10:08:52 2020

@author: chitra
"""
import numpy as np
import pyblp
import pandas as pd
import matplotlib.pyplot as plt
import pathlib


# %%
main_dir = pathlib.Path.cwd().parent
data_dir = main_dir / 'data'

dict_dir = data_dir / 'dict'
raw_dir = data_dir / 'raw'

tab_dir = main_dir / 'tables'
fig_dir = main_dir / 'figures'

from aux_plot_config import *
from aux_late import plot_late_single
from aux_table_functions import draw_blp_agents, load_blp_base
pyblp.options.digits = 3
pyblp.options.verbose = False

# some global variables
TOL = 0.01
POINTS = 1000
NU = 3

PARAMS = [TOL, POINTS, NU]

# %%
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


blp_problem=pyblp.Problem(add_exogenous=False,product_data=blp_products, **problem_options)

filename_blp_base = dict_dir / 'blp_results_base.npy'
# this function only works for base
results_blp = load_blp_base(blp_problem,filename_blp_base)

# %%
# CALL the method
# define some preliminaries on which market you want to analyze
t = -1 ## not sure why this corresponds to 1990? maybe "last one"?
j = 53 # toyota camry
k = 113 # honda accord -- no ranges here!

market_id = blp_problem.unique_market_ids[t]
mkt = market_id

#my_list = np.append(my_list) 
my_list = [26,53,113,75,8,9,19, 49]



plot_late_single(results_blp, blp_products, market_id, 53, 113, fig_dir, 'late_53_113.pdf')
plot_late_single(results_blp, blp_products, market_id, 53, 19, fig_dir, 'late_53_19.pdf')

plot_late_single(results_blp, blp_products, market_id, 9, 8, fig_dir, 'late_9_8.pdf')
plot_late_single(results_blp, blp_products, market_id, 9, 75, fig_dir, 'late_9_75.pdf')


#plot_late_single(results_blp, blp_products, market_id, 8, 53, fig_dir, 'late_8_53.pdf')
#plot_late_single(results_blp, blp_products, market_id, 8, 9, fig_dir, 'late_8_9.pdf')
#plot_late_single(results_blp, blp_products, market_id, 8, 75, fig_dir, 'late_8_75.pdf')

#plot_late_single(results_blp, blp_products, market_id, 8, 113, fig_dir, 'late_8_113.pdf')






