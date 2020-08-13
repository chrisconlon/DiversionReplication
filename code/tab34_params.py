#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:07:45 2020

@author: chitra

Make Table 3: Parameter Estimates
Nevo Results
BLP Results

Make Table 4: PostEstimation Information for Table 2

First, run_blp_cases and run_nevo_cases 
Then this file

"""

import numpy as np
import pyblp
import pandas as pd
import pathlib

from tabulate import tabulate

main_dir = pathlib.Path.cwd().parent
data_dir = main_dir / 'data'
dict_dir = data_dir / 'dict'
raw_dir = data_dir / 'raw'

tab_dir = main_dir / 'tables'

pyblp.options.digits = 2
pyblp.options.verbose = False

from aux_table_functions import load_pyblp_dict, get_params_nevo, get_params_blp, outreg


BESTPRACTICES_TEXT = 'Best Practices'
NOCONS_TEXT = '$\\Sigma_{\\text{cons}} = \\pi_{\\text{cons}} = 0 $'
NOALPHA_TEXT = '$\\Sigma_p = \\pi_p = 0 $'
TRIPLE_TEXT = 'Rescaled Shares'
NL_TEXT = 'Nested Logit'

# %%

# get the BLP Results Back
# requires only the filename, not the problem!

# read in the data for use in the weights only
product_data = pd.read_parquet(raw_dir / 'blp_product_data_opt.parquet')

blp_w = product_data.shares.values[:,None]



filename_blp_base = dict_dir / 'blp_results_base.npy'
filename_blp_nocons = dict_dir / 'blp_results_noconst.npy'
filename_blp_noalpha = dict_dir / 'blp_results_noalpha.npy'
filename_blp_triple = dict_dir / 'blp_results_triple.npy'
filename_blp_nl = dict_dir / 'blp_results_nl.npy'


blp_base_dict = load_pyblp_dict(filename_blp_base)
blp_base = get_params_blp(blp_base_dict,blp_w)

blp_nocons_dict = load_pyblp_dict(filename_blp_nocons)
blp_nocons = get_params_blp(blp_nocons_dict,blp_w)

blp_noalpha_dict = load_pyblp_dict(filename_blp_noalpha)
blp_noalpha = get_params_blp(blp_noalpha_dict,blp_w)

blp_triple_dict = load_pyblp_dict(filename_blp_triple)
blp_triple = get_params_blp(blp_triple_dict,blp_w)

blp_nl_dict = load_pyblp_dict(filename_blp_nl)
blp_nl = get_params_blp(blp_nl_dict,blp_w)



blp_table=pd.concat([pd.Series(blp_base),
                    pd.Series(blp_nocons),
                    pd.Series(blp_noalpha),
                    pd.Series(blp_triple),
                    pd.Series(blp_nl)
                    ],axis=1)

blp_table.columns = [BESTPRACTICES_TEXT, NOCONS_TEXT, NOALPHA_TEXT, TRIPLE_TEXT, NL_TEXT]


blp_table = blp_table.fillna(0)
# %%
# get the Nevo Results Back
# not using the interesting saving for these
# to keep it simpler
filename_nevo_base = dict_dir / 'nevo_results_base.npy'
filename_nevo_nocons = dict_dir / 'nevo_results_noconst.npy'
filename_nevo_noalpha = dict_dir / 'nevo_results_noalpha.npy'
filename_nevo_triple = dict_dir / 'nevo_results_triple.npy'
filename_nevo_nl = dict_dir / 'nevo_results_nl.npy'


agent_data = pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)
product_data = pd.read_parquet(raw_dir / 'nevo_product_data_opt.parquet')
nevo_w = product_data.shares.values[:,None]

nevo_base_dict = load_pyblp_dict(filename_nevo_base)
nevo_base = get_params_nevo(nevo_base_dict,nevo_w)

nevo_nocons_dict = load_pyblp_dict(filename_nevo_nocons)
nevo_nocons = get_params_nevo(nevo_nocons_dict,nevo_w)

nevo_noalpha_dict = load_pyblp_dict(filename_nevo_noalpha)
nevo_noalpha = get_params_nevo(nevo_noalpha_dict,nevo_w)

nevo_triple_dict = load_pyblp_dict(filename_nevo_triple)
nevo_triple = get_params_nevo(nevo_triple_dict,nevo_w)

nevo_nl_dict = load_pyblp_dict(filename_nevo_nl)
nevo_nl = get_params_nevo(nevo_nl_dict,nevo_w)

nevo_table=pd.concat([pd.Series(nevo_base),
                      pd.Series(nevo_nocons),
                      pd.Series(nevo_noalpha),
                      pd.Series(nevo_triple),
                      pd.Series(nevo_nl)
                      ],axis=1)

nevo_table.columns = [BESTPRACTICES_TEXT, NOCONS_TEXT, NOALPHA_TEXT, TRIPLE_TEXT,
                      NL_TEXT]

nevo_table = nevo_table.round(decimals=5)



# %%
# Combine the Nevo and the BLP Tables
# First, make the postestimation tale because that is much easier
stats_order = ['median_own_elas',
               'median_agg_elas',
               'median_og_div',
               'mean_top5_div',
               'mean_markup',
               'median_cs'
               ]

nevo_stats = pd.DataFrame(nevo_table.loc[stats_order])
blp_stats = pd.DataFrame(blp_table.loc[stats_order])

nevo_stats.index = ['\\midrule \\textbf{Nevo} \\\ \\midrule Median Own-Elasticity', 'Median Aggregate Elasticity',
                  'Median Outside-Good Diversion', 'Mean Top 5 Diversion',
                  'Mean Markup', 'Median Consumer Surplus'
                  ]

blp_stats.index = ['\\textbf{BLP} \\\ \\midrule Median Own-Elasticity', 'Median Aggregate Elasticity',
                  'Median Outside-Good Diversion', 'Mean Top 5 Diversion',
                  'Mean Markup', 'Median Consumer Surplus'
                  ]


# make this into a nice two-panel NEVO and BLP Table
# in the style of Table 5

# stack the two
full_stats = blp_stats.append(nevo_stats)

tab_outreg_stats = tabulate(full_stats, tablefmt='latex_raw', 
                            floatfmt='0.3f',
                            headers=nevo_table.columns)
tab_outreg_stats = tab_outreg_stats.replace('\hline','\midrule')

file_outreg_stats = tab_dir / 'tab4_stats.tex'


print(tab_outreg_stats)
with open(file_outreg_stats, 'w') as file:
    file.write(tab_outreg_stats)


# %%
# Create the parameters table
# also a two-panel job
file_outreg_params = tab_dir / 'tab3_params.tex'


# using outreg to convert everything like we did earlier
blp_param_order = ['price_term', 'sigma_cons', 'sigma_hpwt', 'sigma_air', 'sigma_mpd',
                   'sigma_size' ]

blp_se_order = [ 'price_se', 'sigma_cons_se', 'sigma_hpwt_se', 'sigma_air_se', 
        'sigma_mpd_se', 'sigma_size_se']

blp_names = [
             '\\hline \\textbf{BLP} \\\ \hline $\\text{price}/\\text{inc}$',
             '$\\sigma_{\\text{cons}}$', 
             '$\\sigma_{\\text{HP/weight}}$', 
             '$\\sigma_{\\text{air}}$', 
             '$\\sigma_{\\text{MP\$}}$', 
             '$\\sigma_{\\text{size}}$'
             ]

paramcols = [BESTPRACTICES_TEXT, NOCONS_TEXT, NOALPHA_TEXT, TRIPLE_TEXT]

blp_table = blp_table[paramcols]
blp_params = blp_table.loc[blp_param_order]
blp_ses = blp_table.loc[blp_se_order]

blp_outreg = outreg(beta = blp_params, sigma = blp_ses, names = blp_names)
blp_outreg = blp_outreg[blp_table.columns]



## now Nevo Table
nevo_param_order = ['price_coeff', 'sigma_price', 'sigma_cons', 'sigma_sugar',
             'sigma_mushy', 'pi_price_inc', 'pi_price_inc2',
             'pi_price_child',
             'pi_cons_inc',  'pi_cons_age', 'pi_sugar_inc',
             'pi_sugar_age', 'pi_mushy_inc', 'pi_mushy_age']

nevo_se_order = ['price_se', 'sigma_price_se', 'sigma_cons_se', 'sigma_sugar_se',
             'sigma_mushy_se', 'pi_price_inc_se', 'pi_price_inc2_se',
             'pi_price_child_se',
             'pi_cons_inc_se', 'pi_cons_age_se', 'pi_sugar_inc_se',
             'pi_sugar_age_se', 'pi_mushy_inc_se', 'pi_mushy_age_se']

nevo_names = ['\\hline \\textbf{Nevo} \\\ \\hline $\\alpha_{\\text{price}}$', 
              '$\\sigma_{\\text{price}}$', 
             '$\\sigma_{\\text{cons}}$',
             '$\\sigma_{\\text{sugar}}$',
             '$\\sigma_{\\text{mushy}}$',
             '$\\pi_{\\text{price} \\times \\text{inc}}$', 
             '$\\pi_{\\text{price} \\times \\text{inc}^2}$',
             '$\\pi_{\\text{price} \\times \\text{kids}}$',
             '$\\pi_{\\text{cons} \\times \\text{inc}}$',
             '$\\pi_{\\text{cons}  \\times \\text{age}}$',
             '$\\pi_{\\text{sugar} \\times \\text{inc}}$',
             '$\\pi_{\\text{sugar} \\times \\text{age}}$',
             '$\\pi_{\\text{mushy} \\times \\text{inc}}$',
             '$\\pi_{\\text{mushy} \\times \\text{age}}$']

nevo_table = nevo_table[paramcols]
nevo_params = nevo_table.loc[nevo_param_order]
nevo_ses = nevo_table.loc[nevo_se_order]

nevo_outreg = outreg(beta = nevo_params, sigma = nevo_ses, names = nevo_names)
nevo_outreg = nevo_outreg[nevo_table.columns]



nevo_outreg.columns = paramcols
blp_outreg.columns = paramcols


# %%
# stack the tables

full_table = blp_outreg.append(nevo_outreg)
colnames = ['', BESTPRACTICES_TEXT, NOCONS_TEXT, NOALPHA_TEXT, TRIPLE_TEXT]

tab_outreg = tabulate(full_table, tablefmt='latex_raw',floatfmt='0.3f', headers=colnames)

# booktabby, not perfect though
tab_outreg = tab_outreg.replace('\\hline','\\midrule')

print(tab_outreg)

with open(file_outreg_params, 'w') as file:
    file.write(tab_outreg)
    
