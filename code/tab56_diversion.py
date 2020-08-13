
import numpy as np
import pyblp
import pandas as pd
import pathlib
import os
from scipy.stats import zscore
from tabulate import tabulate


main_dir = pathlib.Path.cwd().parent
data_dir = main_dir / 'data'

dict_dir = data_dir / 'dict'
raw_dir = data_dir / 'raw'
tab_dir = main_dir / 'tables'



pyblp.options.digits = 2
pyblp.options.verbose = False

from aux_table_functions import load_pyblp_dict


MTE_TEXT = '$ D_{jk}(p) $'
ATE_TEXT = 'Second Choice'
MTEP_TEXT = 'Small Price Change'
MTEQ_TEXT = 'Small Quality Change'


# %% 
# Define some initial functions
def do_tab4_col(z, og_mask):
    # best is the Djk we will want correlations for
    best=np.nanmax(z*(~og_mask),axis=1)
    matches = np.mean(np.argmax(z*(~og_mask),axis=1)==best_ids)
    return np.array([np.nanmedian(best), np.nanmean(best),matches, np.nanmedian(z[og_mask]),np.nanmean(z[og_mask])])*100.0, best



# %%
filename_nevo_base = dict_dir / 'nevo_results_base.npy'
filename_nevo_nocons = dict_dir / 'nevo_results_noconst.npy'
filename_nevo_noalpha = dict_dir / 'nevo_results_noalpha.npy'
filename_nevo_triple = dict_dir / 'nevo_results_triple.npy'
filename_nevo_logit = dict_dir / 'nevo_results_logit.npy'
filename_nevo_nl = dict_dir / 'nevo_results_nl.npy'


agent_data = pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)
product_data = pd.read_parquet(raw_dir / 'nevo_product_data_opt.parquet')
nevo_w = product_data.shares.values[:,None]

nevo_base_dict = load_pyblp_dict(filename_nevo_base)
nevo_nocons_dict = load_pyblp_dict(filename_nevo_nocons)
nevo_noalpha_dict = load_pyblp_dict(filename_nevo_noalpha)
nevo_triple_dict = load_pyblp_dict(filename_nevo_triple)
nevo_logit_dict = load_pyblp_dict(filename_nevo_logit)
nevo_nl_dict = load_pyblp_dict(filename_nevo_nl)


# calculate diversion ratios
# no argument = MTE q
nevo_mte_div = nevo_base_dict.item().get('diversion_ratios')
nevo_mteq_div = nevo_base_dict.item().get('quality_diversion_ratios')

nevo_ate_div = nevo_base_dict.item().get('long_run_diversion_ratios')
nevo_logit_div = nevo_logit_dict.item().get('long_run_diversion_ratios')

nevo_nl_mte_div = nevo_nl_dict.item().get('diversion_ratios')
nevo_nl_ate_div = nevo_nl_dict.item().get('long_run_diversion_ratios')


nevo_og_mask=np.vstack([np.eye(24).astype(bool) for i in range(0,94)])

best_ids=np.argmax(nevo_mte_div*(~nevo_og_mask),axis=1)


tab4_mte, nevo_mte_best     = do_tab4_col(nevo_mte_div,nevo_og_mask)
tab4_mteq, nevo_mteq_best   = do_tab4_col(nevo_mteq_div,nevo_og_mask)
tab4_ate, nevo_ate_best     = do_tab4_col(nevo_ate_div,nevo_og_mask)
tab4_logit, nevo_logit_best = do_tab4_col(nevo_logit_div,nevo_og_mask)
tab4_nl_mte, nevo_nl_mte_best = do_tab4_col(nevo_nl_mte_div,nevo_og_mask)
tab4_nl_ate, nevo_nl_ate_best = do_tab4_col(nevo_nl_ate_div,nevo_og_mask)

tab4_nevo = pd.DataFrame(np.vstack([tab4_mte,tab4_mteq,
                                    tab4_ate,tab4_logit,
                                    tab4_nl_mte]).T)
    
tab4_nevo.columns = [MTE_TEXT,MTEQ_TEXT, ATE_TEXT,'Logit', f'Nested Logit {MTE_TEXT}']
tab4_nevo.index = ['\\hline \\textbf{Nevo} \\\ \\hline Med($D_{jk}$)','Mean($D_{jk}$)','\% Correct',' \\hline Med($D_{j0}$)','Mean($D_{j0}$)']




# %%
# do the same for BLP

blp_products = pd.read_parquet(raw_dir / 'blp_product_data_opt.parquet')

blp_w = blp_products.shares.values[:,None]



filename_blp_base = dict_dir / 'blp_results_base.npy'
filename_blp_nocons = dict_dir / 'blp_results_noconst.npy'
filename_blp_noalpha = dict_dir / 'blp_results_noalpha.npy'
filename_blp_triple = dict_dir / 'blp_results_triple.npy'
filename_blp_logit = dict_dir / 'blp_results_logit.npy'
filename_blp_nl = dict_dir / 'blp_results_nl.npy'


blp_base_dict = load_pyblp_dict(filename_blp_base)
blp_nocons_dict = load_pyblp_dict(filename_blp_nocons)
blp_noalpha_dict = load_pyblp_dict(filename_blp_noalpha)
blp_triple_dict = load_pyblp_dict(filename_blp_triple)
blp_logit_dict = load_pyblp_dict(filename_blp_logit)
blp_nl_dict = load_pyblp_dict(filename_blp_nl)


blp_mte_div = blp_base_dict.item().get('diversion_ratios')
blp_mteq_div = blp_base_dict.item().get('quality_diversion_ratios')

blp_ate_div = blp_base_dict.item().get('long_run_diversion_ratios')
blp_logit_div = blp_logit_dict.item().get('long_run_diversion_ratios')

blp_nl_mte_div = blp_nl_dict.item().get('diversion_ratios')
blp_nl_ate_div = blp_nl_dict.item().get('long_run_diversion_ratios')

blp_og_mask=np.vstack([np.eye(len(blp_mte_div[blp_products['market_ids']==i,:]),150) for i in range(1971,1990+1)])==1
best_ids=np.argmax(blp_mte_div*(~blp_og_mask),axis=1)


tab4_mte, blp_mte_best      = do_tab4_col(blp_mte_div,blp_og_mask)
tab4_mteq, blp_mteq_best    = do_tab4_col(blp_mteq_div,blp_og_mask)
tab4_ate, blp_ate_best      = do_tab4_col(blp_ate_div,blp_og_mask)
tab4_logit, blp_logit_best  = do_tab4_col(blp_logit_div,blp_og_mask)
tab4_nl_mte,blp_nl_mte_best = do_tab4_col(blp_nl_mte_div,blp_og_mask)
tab4_nl_ate,blp_nl_ate_best = do_tab4_col(blp_nl_ate_div,blp_og_mask)

tab4_blp = pd.DataFrame(np.vstack([tab4_mte,
                                   tab4_mteq,
                                   tab4_ate,
                                   tab4_logit,
                                   tab4_nl_mte]).T)
    
tab4_blp.columns = [MTE_TEXT,MTEQ_TEXT, ATE_TEXT,'Logit', f'Nested Logit {MTE_TEXT}']
tab4_blp.index = [' \\textbf{BLP} \\\ \\hline Med($D_{jk}$)','Mean($D_{jk}$)','\% Correct',' \\hline Med($D_{j0}$)','Mean($D_{j0}$)']





# %%

tab4 = tab4_blp.append(tab4_nevo)
tab4_outreg = tabulate(tab4 ,headers=tab4.columns, tablefmt='latex_raw',floatfmt='0.2f')
tab4_outreg = tab4_outreg.replace('\\hline','\\midrule')
print(tab4_outreg)


file_outreg4 = tab_dir / 'tab5_div.tex'

print(tab4_outreg)
with open(file_outreg4, 'w') as file:
    file.write(tab4_outreg)


# %%
    
# Interesting pairwise correlation matrix
# Djk, which is "best" under do_tab4_col, from og_mask
# does it have the same number of obs as everywhere?

# put them all in a dataframe
    
    
def pairwise_corr(mte, mteq, ate, logit, nl_mte):
    df = pd.DataFrame([mte, mteq, ate, logit, nl_mte]).transpose()
    df.columns = [MTE_TEXT,MTEQ_TEXT, ATE_TEXT,'Logit', f'Nested Logit {MTE_TEXT}']
    
    
    return df.corr()

blp_corr = pairwise_corr(blp_mte_best, blp_mteq_best, blp_ate_best, blp_logit_best,blp_nl_mte_best)
tabB1_blp = tabulate(blp_corr ,headers = blp_corr.columns, tablefmt='latex_raw',floatfmt='0.3f') 

nevo_corr = pairwise_corr(nevo_mte_best, nevo_mteq_best, nevo_ate_best, nevo_logit_best,nevo_nl_mte_best)
tabB1_nevo = tabulate(nevo_corr ,headers = nevo_corr.columns, tablefmt='latex_raw',floatfmt='0.3f') 


file_tabB1_blp = tab_dir / 'tabB1_blp.tex'
file_tabB1_nevo = tab_dir / 'tabB1_nevo.tex'


with open(file_tabB1_blp, 'w') as file:
    file.write(tabB1_blp)
    
with open(file_tabB1_nevo, 'w') as file:
    file.write(tabB1_nevo)
# %%    
# Recreating Table 5
# this creates each ROW
    
def relative_error(z):
    # only concerned with two statistics for now
    med_abs = lambda x: np.nanmedian(np.abs(x))
    mean_abs = lambda x: np.nanmean(np.abs(x))
    st_abs = lambda x: np.nanstd(np.abs(x))
    
    # keep median abslute value and mean
    flist= [med_abs,np.nanmean]
    return 100.0*np.array([f(z) for f in flist])

MEDABS_NAME = 'med($|y-x|$)'
NANMEAN_NAME = 'mean($y-x$)'


nevo_mteq_diff=np.log(nevo_mteq_div/nevo_mte_div)
nevo_ate_diff=np.log(nevo_ate_div/nevo_mte_div)
nevo_log_diff=np.log(nevo_logit_div/nevo_mte_div)
nevo_nlog_diff=np.log(nevo_nl_mte_div/nevo_mte_div)

nevo_best_mte=np.nanmax(nevo_mte_div*(~nevo_og_mask),axis=1)
nevo_best_mteq=np.nanmax(nevo_mteq_div*(~nevo_og_mask),axis=1)
nevo_best_ate=np.nanmax(nevo_ate_div*(~nevo_og_mask),axis=1)
nevo_best_logit=np.nanmax(nevo_logit_div*(~nevo_og_mask),axis=1)
nevo_best_nlogit=np.nanmax(nevo_nl_mte_div*(~nevo_og_mask),axis=1)

nevo_mteq_diff_b=np.log(nevo_best_mteq/nevo_best_mte)
nevo_ate_diff_b=np.log(nevo_best_ate/nevo_best_mte)
nevo_log_diff_b=np.log(nevo_best_logit/nevo_best_mte)
nevo_nlog_diff_b=np.log(nevo_best_nlogit/nevo_best_mte)

a1=relative_error(nevo_mteq_diff_b)
a2=relative_error(nevo_ate_diff_b)
a3=relative_error(nevo_log_diff_b)
a4=relative_error(nevo_nlog_diff_b)

b1=relative_error(nevo_mteq_diff[~nevo_og_mask])
b2=relative_error(nevo_ate_diff[~nevo_og_mask])
b3=relative_error(nevo_log_diff[~nevo_og_mask])
b4=relative_error(nevo_nlog_diff[~nevo_og_mask])

c1=relative_error(nevo_mteq_diff[nevo_og_mask])
c2=relative_error(nevo_ate_diff[nevo_og_mask])
c3=relative_error(nevo_log_diff[nevo_og_mask])
c4=relative_error(nevo_nlog_diff[nevo_og_mask])

tab_nevo=pd.DataFrame(np.vstack([a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4]))

tab_nevo.columns = [ MEDABS_NAME, NANMEAN_NAME]

#tab_nevo.index = [f'hline \bf Top 5 Substitutes && \ hline {MTEQ_TEXT}', ATE_TEXT,'Logit',
#              f'hline \bf All Products&& \ hline {MTEQ_TEXT}', ATE_TEXT,'Logit',
#              f'hline \bf Outside Good&& \ hline {MTEQ_TEXT}', ATE_TEXT,'Logit'
#              ]

# %%
# Now that we have the Nevo table set up
# we will want to do the right half: BLP
blp_mteq_diff=np.log(blp_mteq_div/blp_mte_div)
blp_ate_diff=np.log(blp_ate_div/blp_mte_div)
blp_log_diff=np.log(blp_logit_div/blp_mte_div)
blp_log_diff=np.log(blp_logit_div/blp_mte_div)
blp_nlog_diff=np.log(blp_nl_mte_div/blp_mte_div)

blp_best_mte=np.nanmax(blp_mte_div*(~blp_og_mask),axis=1)
blp_best_mteq=np.nanmax(blp_mteq_div*(~blp_og_mask),axis=1)
blp_best_ate=np.nanmax(blp_ate_div*(~blp_og_mask),axis=1)
blp_best_logit=np.nanmax(blp_logit_div*(~blp_og_mask),axis=1)
blp_best_nlogit=np.nanmax(blp_nl_mte_div*(~blp_og_mask),axis=1)


blp_mteq_diff_b=np.log(blp_best_mteq/blp_best_mte)
blp_ate_diff_b=np.log(blp_best_ate/blp_best_mte)
blp_log_diff_b=np.log(blp_best_logit/blp_best_mte)
blp_nlog_diff_b=np.log(blp_best_nlogit/blp_best_mte)


a1=relative_error(blp_mteq_diff_b)
a2=relative_error(blp_ate_diff_b)
a3=relative_error(blp_log_diff_b)
a3=relative_error(blp_nlog_diff_b)

b1=relative_error(blp_mteq_diff[~blp_og_mask])
b2=relative_error(blp_ate_diff[~blp_og_mask])
b3=relative_error(blp_log_diff[~blp_og_mask])
b4=relative_error(blp_nlog_diff[~blp_og_mask])

c1=relative_error(blp_mteq_diff[blp_og_mask])
c2=relative_error(blp_ate_diff[blp_og_mask])
c3=relative_error(blp_log_diff[blp_og_mask])
c4=relative_error(blp_nlog_diff[blp_og_mask])

tab_blp=pd.DataFrame(np.vstack([a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4]))

tab_blp.columns = [ '\multicolumn{2}{c|}{\\textbf{BLP}} & \multicolumn{2}{c}{\\textbf{Nevo}} \\\\ \hline & ' + MEDABS_NAME, NANMEAN_NAME] # ,'mean($|y-x|$)','std($|y-x|$)']


# %%
# merge these two based on index

#tab_blp_mg = tab_blp.reset_index(drop=True)
#tab_nevo_mg = tab_nevo.reset_index(drop=True)

tab = tab_blp.join(tab_nevo, lsuffix = ' ')


tab.index = [f'\hline \\bf Top 5 Substitutes && \\\ \hline {MTEQ_TEXT}', ATE_TEXT,'Logit', f'Nested Logit {MTE_TEXT}',
              f'\hline \\bf All Products&& \\\\ \hline {MTEQ_TEXT}', ATE_TEXT,'Logit',f'Nested Logit {MTE_TEXT}',
              f'\hline \\bf Outside Good&& \\\\ \hline {MTEQ_TEXT}', ATE_TEXT,'Logit',f'Nested Logit {MTE_TEXT}'
              ]


#tab = tab_blp_mg.join(tab_nevo_mg, lsuffix=' ' )
#tab.index = tab.
#tab.drop('index ', axis=1)

# %%

#colnames = ['\multicolumn{2}{|c|}{Sets}', 'med($y-x$)','mean($y-x$)']

tab_outreg = tabulate(tab ,headers=tab.columns, tablefmt='latex_raw',
                       floatfmt='0.2f'
                       )
tab_outreg = tab_outreg.replace('hline','midrule')
print(tab_outreg)


tab_outreg = tab_outreg.replace('\begin{tabular}{lrr',
                    '\begin{tabular}{lrr|')



file_outreg = tab_dir / 'tab6_rel.tex'

print(tab_outreg)
with open(file_outreg, 'w') as file:
    file.write(tab_outreg)


