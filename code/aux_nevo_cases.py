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


from aux_table_functions import *

# %%
tick()
filename_base = dict_dir / 'nevo_results_base.npy'
filename_nocons = dict_dir / 'nevo_results_noconst.npy'
filename_noalpha = dict_dir / 'nevo_results_noalpha.npy'
filename_triple = dict_dir / 'nevo_results_triple.npy'
filename_logit = dict_dir / 'nevo_results_logit.npy'
filename_nl = dict_dir / 'nevo_results_nl.npy'


# %%
data_agent = pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)
nevo_products = pd.read_parquet(raw_dir / 'nevo_product_data_opt.parquet')
# Triple inside share
product_data_adj = nevo_products.copy()
product_data_adj['shares']=nevo_products['shares']/3.0

# initial values
init_sigma=np.diag([0.3302, 2.4526, 0.0163, 0.2441])
init_pi=np.array([    [5.4819, 0, 0.2037, 0],
        [15.8935, -.1, 0, 2.6342],
        [-0.2506, 0, 0.0511, 0],
        [1.2650, 0, -0.8091, 0]
    ])

pi_noalpha=init_pi.copy()
sigma_noalpha=init_sigma.copy()
sigma_noalpha[1,1]=0
pi_noalpha[1,:]=np.c_[0,0,0,0]

pi_nocons=init_pi.copy()
sigma_nocons=init_sigma.copy()
sigma_nocons[0,0]=0
pi_nocons[0,:]=np.c_[0,0,0,0]

# %%
problem_options = dict(
    product_formulations=(
        pyblp.Formulation(formula='prices', absorb='C(product_ids)'),
        pyblp.Formulation('1 + prices + sugar + mushy')
    ),
    agent_formulation=pyblp.Formulation('0 + income + income_squared + age + child'),
    agent_data = data_agent
)

# %%
solve_options = dict(
    initial_update=True,
    iteration=pyblp.Iteration('squarem', {'atol': 1e-14}),
    optimization=pyblp.Optimization('bfgs', {'gtol': 1e-5}),
    scale_objective=True,
    method='2s',
)

nl_data=nevo_products.copy()
nl_data['nesting_ids']=1
nl_options=dict(
    product_formulations=(
        pyblp.Formulation(formula='prices', absorb='C(product_ids)')
    ),
    product_data=nl_data
)

# Setup initial problem and get optimal IV
problem = pyblp.Problem(product_data=nevo_products,**problem_options)
# %%
# Base

def get_nevo_base():
    tick()
    results_base = problem.solve(sigma=init_sigma,pi=init_pi,**solve_options)
    save_pyblp_results(results_base, problem, filename_base)
    tock()
    return results_base


# %%
# No Constant
def get_nevo_nocons():
    tick()
    results_nocons = problem.solve(sigma=sigma_nocons,pi=pi_nocons,**solve_options)
    save_pyblp_results(results_nocons, problem, filename_nocons)
    tock()
    
    return results_nocons
# %%
# No Alpha (Price)
def get_nevo_noalpha():
    tick()
    results_noalpha = problem.solve(sigma=sigma_noalpha,pi=pi_noalpha,**solve_options)
    save_pyblp_results(results_noalpha, problem, filename_noalpha)
    tock()
    return results_noalpha
# %%
# Triple Shares
def get_nevo_triple():
    tick()
    problem_triple = pyblp.Problem(product_data=product_data_adj,**problem_options)
    results_triple=problem_triple.solve(sigma=init_sigma,pi=init_pi,**solve_options)
    save_pyblp_results(results_triple, problem_triple,filename_triple)
    tock()
    
    return results_triple

# %%
# Logit (not Nested)
def get_nevo_logit():
    tick()
    problem_logit = pyblp.Problem(
        product_formulations=(
           pyblp.Formulation('0 + prices', absorb='C(product_ids)')
        ),
        product_data = nevo_products,
        agent_formulation = None,
        agent_data = None
    )
        
    results_logit = problem_logit.solve()
    save_pyblp_results(results_logit, problem_logit,filename_logit)
    
    tock()
    return results_logit
# %%
# Nested logit: Note: we are "calibrating" rho for OG diversion
# if we estimate rho we get 0.99 and things look terrible
def get_nevo_nested():
    tick()
    problem_nl = pyblp.Problem(**nl_options)
    results_nl = problem_nl.solve(rho=0.375,optimization=pyblp.Optimization('return'))
    save_pyblp_results(results_nl, problem_nl,filename_nl)
    tock()
    return results_nl
