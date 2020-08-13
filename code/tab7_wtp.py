#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 12:57:27 2020

@author: chitra
"""

import pyblp
import numpy as np
import pandas as pd
import pathlib


import matplotlib.pyplot as plt
pyblp.options.digits = 2
pyblp.options.verbose = False


main_dir = pathlib.Path.cwd().parent
data_dir = main_dir / 'data'

dict_dir = data_dir / 'dict'
raw_dir = data_dir / 'raw'

tab_dir = main_dir / 'tables'
fig_dir = main_dir / 'figures'

from aux_plot_config import *
from aux_table_functions import draw_blp_agents,load_blp_base, do_single_market, do_single_market_indiv, reshape_wtp
from aux_nevo_cases import get_nevo_base


## Generic Options
solve_options = dict(
	#costs_bounds=(0.001, None),
    #W_type='clustered',
    #se_type='clustered',
    initial_update=True,
    iteration=pyblp.Iteration('squarem', {'atol': 1e-14}),
    optimization=pyblp.Optimization('bfgs', {'gtol': 1e-5}),
    scale_objective=True,
    method='2s',
)

# %%
################
### Nevo problem
################

# Import the Best Practices RESULTS OBJECT

nevo_products = pd.read_parquet(raw_dir / 'nevo_product_data_opt.parquet')

nevo_agents = pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)
nevo_agents['draw_ids']=np.tile(range(0,20),94) 

## Get Best Practices Nevo Results


results_nevo = get_nevo_base()

# get the alphas separately
demog = nevo_agents[['income','income_squared','age','child']].values
# trusting chris's code on this part
## Add the alpha_i
nevo_agents['alpha'] = np.abs(demog @ results_nevo.pi[1,] + results_nevo.sigma[(1,1)]+nevo_agents.nodes1*results_nevo.sigma[1,1]+results_nevo.beta.item())

nevo_alpha = nevo_agents['alpha'].copy()

# %%
################
### BLP problem
################
# get the BLP Results Back
# compare blp_base and results_blp

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

solve_options = dict(
    costs_bounds=(0.001, None),
    W_type='clustered',
    se_type='clustered',
    initial_update=True,
    iteration=pyblp.Iteration('squarem', {'atol': 1e-14}),
    optimization=pyblp.Optimization('bfgs', {'gtol': 1e-5}),
    scale_objective=False,
)

blp_problem=pyblp.Problem(add_exogenous=False,product_data=blp_products, **problem_options)

# why does it only work this way?
# using the original loading function works for base anyways...
filename_blp_base = dict_dir / 'blp_results_base.npy'
results_blp = load_blp_base(blp_problem,filename_blp_base)

# Get the alpha_i
blp_agents['alpha'] = np.abs(results_blp.pi[1].item()/blp_agents.income)

## copy only the alphas
blp_alpha = blp_agents['alpha'].copy()

# %%
############################
### Do the WTP calculations
############################


# CRM Update: Putting the functions into aux_table_functions


# not sure if this a summary of the below?
big_df_nevo=pd.concat([do_single_market(results_nevo,nevo_products,[x]) for x in nevo_products.market_ids.unique()],axis=0)
# wide meaning across individual IDs
wide_df_nevo=pd.concat([do_single_market_indiv(results_nevo,nevo_products,[x]) for x in nevo_products.market_ids.unique()],axis=0)
long_df_nevo=pd.merge(reshape_wtp(wide_df_nevo).reset_index(),nevo_agents[['market_ids','draw_ids','alpha']],on=['market_ids','draw_ids'])



# %%
# perhaps NOT as correlated as theory suggests?
#print(np.corrcoef(big_df_nevo[['wtp','shares','div0']].values,rowvar=False))
#print(np.corrcoef(long_df_nevo[['wtp','shares','div0','alpha']].values,rowvar=False))

# %%
big_df_blp=pd.concat([do_single_market(results_blp,blp_products,[x]) for x in blp_products.market_ids.unique()],axis=0)
wide_df_blp=pd.concat([do_single_market_indiv(results_blp,blp_products,[x]) for x in blp_products.market_ids.unique()],axis=0)

# add the alpha_i in for BLP
long_df_blp=pd.merge(reshape_wtp(wide_df_blp).reset_index(),blp_agents[['market_ids','draw_ids','alpha']],on=['market_ids','draw_ids'])

long_df_blp2=long_df_blp[long_df_blp.wtp > 0].copy().reset_index(drop=True)

#print(np.corrcoef(big_df_blp[['wtp','shares','div0']].values,rowvar=False))
#print(np.corrcoef(long_df_blp2[['wtp','shares','div0','alpha']].values,rowvar=False))

df_nevo_indiv = long_df_nevo
df_nevo_all = big_df_nevo
df_blp_indiv = long_df_blp2
df_blp_all = big_df_blp


# %%
# CRM Update: Moving the regressions to python from R

# take the log of wtp, shares, div0, and    
    
# clean up everything the same way
    
def clean_df(df):
    df['lwtp'] = np.log(df['wtp'])
    df['lshares'] = np.log(df['shares'])
    df['ldiv0'] = np.log(df['div0'])


    # add a constant
    
    df = df.assign(cons = 1)
    # might be missing
    # might be infinite
    # drop both cases
    # why am I missing four observations relative to the other?
    # my numbers are going to be slightly different
    # because i'm now using best practices data
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0)

    
    
    return df
# quickly take logs and clean the DF
    

df_nevo_indiv = clean_df(df_nevo_indiv)
df_nevo_all = clean_df(df_nevo_all)
df_blp_indiv = clean_df(df_blp_indiv)
df_blp_indiv['lalpha'] = np.log(df_blp_indiv['alpha'])
df_blp_all = clean_df(df_blp_all)

# also take log of alpha


# Run regressions with and without FE
# add a constant to the data
# %%

import statsmodels.formula.api as smf

mod_nevo_all = smf.ols(formula='lwtp ~ lshares + ldiv0', 
                    data = df_nevo_all).fit()
mod_nevo_all_fe = smf.ols(formula='lwtp ~ lshares + ldiv0+C(market_ids)', 
                    data = df_nevo_all).fit()
mod_nevo_indiv = smf.ols(formula='lwtp ~ lshares + ldiv0', 
                    data = df_nevo_indiv).fit()
mod_nevo_indiv_fe = smf.ols(formula='lwtp ~ lshares + ldiv0 +C(market_ids)', 
                    data = df_nevo_indiv).fit()
mod_nevo_base_indiv = smf.ols(formula='lwtp ~ lshares',
                       data = df_nevo_indiv).fit()

mod_nevo_base = smf.ols(formula='lwtp ~ lshares',
                       data = df_nevo_all).fit()



mod_blp_all = smf.ols(formula='lwtp ~ lshares + ldiv0', 
                    data = df_blp_all).fit()
mod_blp_all_fe = smf.ols(formula='lwtp ~ lshares + ldiv0+C(market_ids)', 
                    data = df_blp_all).fit()
mod_blp_indiv = smf.ols(formula='lwtp ~ lshares + ldiv0', 
                    data = df_blp_indiv).fit()
mod_blp_indiv_fe = smf.ols(formula='lwtp ~ lshares + ldiv0+C(market_ids)', 
                    data = df_blp_indiv).fit()
mod_blp_alpha= smf.ols(formula='lwtp ~ lshares + ldiv0 + lalpha', 
                    data = df_blp_indiv).fit()
mod_blp_alpha_nodiv = smf.ols(formula='lwtp ~ lshares + lalpha', 
                    data = df_blp_indiv).fit()
mod_blp_base_indiv = smf.ols(formula='lwtp ~ lshares',
                       data = df_blp_indiv).fit()
mod_blp_base = smf.ols(formula='lwtp ~ lshares',
                       data = df_blp_all).fit()




blp_base_indiv_r2 = mod_blp_base_indiv.rsquared
blp_base_r2 = mod_blp_base.rsquared.round(4)
    
nevo_base_indiv_r2 = mod_nevo_base_indiv.rsquared
nevo_base_r2 = mod_nevo_base.rsquared.round(4)


# %%
# Attempting to get all the models together into one
from statsmodels.iolib.summary2 import summary_col
import re


models = [mod_blp_base_indiv, mod_blp_indiv, mod_blp_alpha, mod_blp_alpha_nodiv, 
          mod_nevo_base_indiv, mod_nevo_indiv, 
          mod_blp_base, mod_blp_all, mod_nevo_base, mod_nevo_all]

names= ['BLP ', 'BLP  ', 'BLP   ', 'BLP    ', 'Nevo', 'Nevo ', 'BLP     ', 'BLP      ', 'Nevo   ', 'Nevo     ']


stats = {'R\sq': lambda x: f"{x.rsquared:.4f}",
             'Adjusted R\sq': lambda x: f"{x.rsquared_adj:.4f}",
             'Observations': lambda x: f"{int(x.nobs):d}"
         } 


coefs = ['lshares','ldiv0','lalpha']

#'\\multicolumn{2}{c|}{\\textbf{BLP}} & \\multicolumn{2}{c}{\\textbf{Nevo}} 




caption = '\\\\caption*{Table 7: Correlation with WTP Measure for Nevo (2000b) and Berry et al. (1999).}'


outreg = summary_col(results = models,
                      float_format='%0.4f',
                           stars=True,
                           info_dict = stats,
                           model_names= names,
                           regressor_order=coefs,
                           drop_omitted=True)
    
    
tab_wtp = outreg.as_latex()
tab_wtp = re.sub(r'\*\*\*', '*', tab_wtp)
tab_wtp = re.sub(r'hline', 'toprule', tab_wtp, count=1)
tab_wtp = re.sub(r'hline', 'bottomrule', tab_wtp, count=1)
tab_wtp = re.sub(r'lshares', '$\\\log(s_{jt})$', tab_wtp)
tab_wtp = re.sub(r'ldiv0', '$\\\log(D_{j,0})$', tab_wtp)
tab_wtp = re.sub(r'lalpha', '$\\\log(\\\\abs{\\\\alpha_i})$', tab_wtp)
tab_wtp = re.sub(r'\nR\\sq', '\n \\\hline $R^2$', tab_wtp)
tab_wtp = re.sub(r'R\\sq', '$R^2$', tab_wtp)
tab_wtp = re.sub(r'\begin{table}', '\begin{table}\footnotesize', tab_wtp)
tab_wtp = re.sub(r'\\caption{}', caption, tab_wtp)

tab_wtp = re.sub(r'\\caption{}', '\\\\caption*{' + caption + '}', tab_wtp) 

header = '\\\\toprule \\\multicolumn{7}{c|}{\\\\textbf{Individual}} & \\\multicolumn{4}{c}{\\\\textbf{Aggregate}} \\\\\ '


tab_wtp = re.sub('\\\\toprule', header, tab_wtp)

tab_wtp = re.sub('lcccccccccc', 'lcccccc|cccc', tab_wtp)
file_outreg = tab_dir / 'tab7_wtp.tex'
with open(file_outreg, 'w') as file:
    file.write(tab_wtp)




# %% 
# Graph Nevo and BLP Alphas together
# density plots for alpha_i for both models

B = 500

# drop blp values above 200

blp_alpha_trunc = blp_alpha.drop(blp_alpha[blp_alpha > 100].index).reset_index(drop=True)
nevo_alpha_trunc = nevo_alpha.drop(nevo_alpha[nevo_alpha > 100].index).reset_index(drop=True)


fig, ax = plt.subplots(figsize= (20,20))
#hist1 = ax.hist(blp_alpha, bins=500, density=True,
#                facecolor='blue',alpha=0.25, 
#                label = 'BLP')

hist_blp = plt.hist(blp_alpha_trunc, bins=B, density=True,
         facecolor='blue',alpha=0.25, label = 'BLP')
#hist = ax.hist(blp_alpha, bins=1000, 
#               density=True, logx=True,
#               facecolor='blue',alpha=0.25, 
#               label = 'BLP')

hist_nevo = plt.hist(nevo_alpha_trunc, bins=B, density=True,
         facecolor='red',alpha=0.25, label = 'Nevo')

#blp_alpha.plot.hist(bins=B)
savefile = fig_dir / 'hist_alpha.pdf'
plt.savefig(savefile, bbox_inches='tight')  


