"""
Goal: Run many BLP cases, save results to a dict
to access these results later
"""

import pyblp
import numpy as np
import pandas as pd
import pathlib
from scipy.stats import zscore

main_dir = pathlib.Path.cwd().parent
data_dir = main_dir / 'data'
dict_dir = data_dir / 'dict'
raw_dir = data_dir / 'raw'

pyblp.options.verbose = False

from aux_table_functions import *

# %%

# product data doesn't need to be constructed
product_data = pd.read_parquet(raw_dir / 'blp_product_data_opt.parquet')

# Triple inside share
product_data_adj = product_data.copy()
product_data_adj['shares']=product_data_adj['shares']*3.0

# Set draws here
agent_data=draw_blp_agents(500)

# define common options for initializing and solving the problem
problem_options = dict(
    product_formulations=(
        pyblp.Formulation('1 + hpwt + air + mpd + space'),
        pyblp.Formulation('1 + prices + hpwt + air + mpd + space'),
        pyblp.Formulation(f'1 + log(hpwt) + air + log(mpg) + log(space) + trend'),
    ),
    agent_formulation=pyblp.Formulation('0 + I(1 / income)'),
    costs_type='log',
    agent_data=agent_data
)

# define common options for initializing and solving the problem
# - eliminate price from X2 and remove demographic formula
# - Using same draws to keep comparisons similar -- could do quadrature instead
problem_no_alpha = dict(
    product_formulations=(
        pyblp.Formulation('1 + prices + hpwt + air + mpd + space'),
        pyblp.Formulation('1  + hpwt + air + mpd + space'),
        pyblp.Formulation(f'1 + log(hpwt) + air + log(mpg) + log(space) + trend'),
    ),
    costs_type='log',
    product_data=product_data,
    agent_data=agent_data
)

solve_options = dict(
    costs_bounds=(0.001, None),
    W_type='clustered',
    se_type='clustered',
    initial_update=True,
    iteration=pyblp.Iteration('squarem', {'atol': 1e-14}),
    optimization=pyblp.Optimization('bfgs', {'gtol': 1e-5}),
    scale_objective=False,
)

## Nested logit
nl_data=product_data.copy()
nl_data['nesting_ids']=1

problem_no_X2 = dict(
    product_formulations=(
        pyblp.Formulation('1 + prices + hpwt + air + mpd + space')
    ),
    product_data=nl_data
)

solve_nl = dict(
    costs_bounds=(0.001, None),
    W_type='clustered',
    se_type='clustered',
    initial_update=True,
    iteration=pyblp.Iteration('squarem', {'atol': 1e-14}),
    optimization=pyblp.Optimization('bfgs', {'gtol': 1e-5}),
    scale_objective=False,
)



# %%
filename_base = dict_dir / 'blp_results_base.npy'
filename_nocons = dict_dir / 'blp_results_noconst.npy'
filename_noalpha = dict_dir / 'blp_results_noalpha.npy'
filename_triple = dict_dir / 'blp_results_triple.npy'
filename_logit = dict_dir / 'blp_results_logit.npy'
filename_nl = dict_dir / 'blp_results_nl.npy'
    


# %%
# Base case (best practices but with fewer draws)

def get_blp_base():
    tick()
    sigma_base=np.diag([3.612, 0, 4.628, 1.818, 1.050, 2.056])
    pi_base=np.c_[[0, -43.501, 0, 0, 0, 0]].astype(float)
    beta_base=np.c_[[-6.67862016, -5, 2.7741249, 0.57237907, 0.34009843,3.91954976]]
    problem_base=pyblp.Problem(add_exogenous=False,product_data=product_data, **problem_options)
    results_base = problem_base.solve(sigma=sigma_base, pi=pi_base, **solve_options)
    save_pyblp_results(results_base, problem_base, filename_base)

    tock()
    return results_base


# %%
# Restrict constant to have no RC
def get_blp_nocons():
    tick()
    pi_base=np.c_[[0, -43.501, 0, 0, 0, 0]].astype(float)
    problem_base=pyblp.Problem(add_exogenous=False,product_data=product_data, **problem_options)
    sigma_noconst=np.diag([0, 0, 4.628, 1.818, 1.050, 2.056])
    results_nocons = problem_base.solve(sigma=sigma_noconst, pi=pi_base, **solve_options)
    save_pyblp_results(results_nocons, problem_base, filename_nocons)
    tock()
    
    return results_nocons
# %%
# Restrict to having common price coefficient (no demographics)
def get_blp_noalpha():
    tick()
    sigma_noalpha=np.diag([ 0.22539812, 10.21865474, -0.03314036, -0.07007719,  0.0184184 ])
    beta_noalpha=[ -7.05576337,  -0.31806557, -12.18847385,   2.22350266,  0.12090703,   2.71179815]
    problem_noalpha=pyblp.Problem(add_exogenous=False, **problem_no_alpha)
    results_noalpha=problem_noalpha.solve(sigma=sigma_noalpha, beta=beta_noalpha, **solve_options)
    save_pyblp_results(results_noalpha, problem_noalpha, filename_noalpha)
    tock()
    
    return results_noalpha
# %%
# Triple inside share
def get_blp_triple():
    tick()
    sigma_triple=np.diag([3.48532311,  0.        ,  0.13802172, 2.65830791, 0.49728683, 0.90109011])
    pi_triple=pi_base=np.c_[[0, -13.5483884, 0, 0, 0, 0]].astype(float)
    problem_triple=pyblp.Problem(add_exogenous=False,product_data=product_data_adj, **problem_options)
    results_triple=problem_triple.solve(sigma=sigma_triple, pi=pi_triple, **solve_options)
    save_pyblp_results(results_triple, problem_triple, filename_triple)
    tock()
    return results_triple

# %%
# Logit
def solve_logit_blp(inst,my_products):
    # only one formulation
    logit_form=pyblp.Formulation('1 + prices + hpwt + air + mpd + space')
    # only demand instruments
    my_products['demand_instruments'] = inst
    logit_problem = pyblp.Problem(logit_form, my_products,
                                  add_exogenous=False)
    logit_results = logit_problem.solve(W = np.identity(inst.shape[1]))
    return logit_problem, logit_results


def original_inst(Y,products):
    own = np.zeros_like(Y)
    total = np.zeros_like(Y)
    for n, (t, f) in enumerate(zip(products['market_ids'], products['firm_ids'])):
        own[n] = Y[n] * np.sum((products['market_ids'] == t) & (products['firm_ids'] == f))
        total[n] = Y[products['market_ids'] == t].sum(axis=0)
    return np.c_[own, total]


# define formulations
# formulation for the instrument

# load product data
# use optimal products again
def get_blp_logit():
    product_data_df = product_data.to_dict('series')
    logit_products = product_data_df.copy()
    
    inst_form = pyblp.Formulation('1 + hpwt + air + mpd + space')
    X = pyblp.build_matrix(inst_form, logit_products)
    
    # get the "original instruments"
    orig_inst = np.apply_along_axis(zscore,0,original_inst(X,logit_products))
    
    # solve one logit
    # first argument: instruments
    # second argument: logit_products (which are a copy)
    problem_logit, results_logit = solve_logit_blp(np.c_[X, orig_inst],logit_products)
    
    
    save_pyblp_results(results_logit, problem_logit,filename_logit)
    
    return results_logit

# %%
# Nested logit
def get_blp_nested():
    tick()
    init_b=[-5.37184814e+00,-1,  3.73890734e+00,  5.13047685e-01, -4.83872040e-03,3.61379854e+00]
    problem_nl=pyblp.Problem(add_exogenous=False, **problem_no_X2)
    results_nl = problem_nl.solve(rho=0.8,beta=init_b, **solve_nl)
    save_pyblp_results(results_nl, problem_nl,filename_nl)
    tock()
    return results_nl

