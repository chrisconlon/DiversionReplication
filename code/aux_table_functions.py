#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:10:36 2020

@author: chitra
"""
import time
_start_time = time.time()
def tick():
    global _start_time 
    _start_time = time.time()
def tock():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60) 
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))
    



import numpy as np
import pyblp
import pandas as pd

# the standard deviation of log income is constant across years, but it has year-varying means


# 0.375 is calibrated to match OG diversion of 2nd choice data
def solve_nl_nevo(df,rho=0.375):
    groups = df.groupby(['market_ids', 'nesting_ids'])
    df['demand_instruments20'] = groups['shares'].transform(np.size)
    nl_formulation = pyblp.Formulation('0 + prices')
    problem = pyblp.Problem(nl_formulation, df)
    res=problem.solve(rho=rho,optimization=pyblp.Optimization('return'))
    og=res.extract_diagonals(res.compute_diversion_ratios()).mean()
    print(og)
    return problem,res



def draw_blp_agents(ndraws=10000):
    log_income_sd = 1.72
    log_income_means = {
        1971: 2.01156,
        1972: 2.06526,
        1973: 2.07843,
        1974: 2.05775,
        1975: 2.02915,
        1976: 2.05346,
        1977: 2.06745,
        1978: 2.09805,
        1979: 2.10404,
        1980: 2.07208,
        1981: 2.06019,
        1982: 2.06561,
        1983: 2.07672,
        1984: 2.10437,
        1985: 2.12608,
        1986: 2.16426,
        1987: 2.18071,
        1988: 2.18856,
        1989: 2.21250,
        1990: 2.18377,
    }
    # construct agent data year-by-year
    market_ids = []
    weights = []
    nodes = []
    income = []
    for index, (year, log_income_mean) in enumerate(log_income_means.items()):
        integration = pyblp.Integration('halton', ndraws, {'discard': 1000 + index * ndraws,'seed': index})
        untransformed_agents = pyblp.build_integration(integration, 6)
        market_ids.append(np.repeat(year, untransformed_agents.weights.size))
        weights.append(untransformed_agents.weights)
        nodes.append(untransformed_agents.nodes[:, :-1])
        income.append(np.exp(log_income_mean + log_income_sd * untransformed_agents.nodes[:, -1]))

    # concatenate the constructed agent data
    agent_data = {
        'market_ids': np.concatenate(market_ids),
        'weights': np.concatenate(weights),
        'nodes': np.vstack(nodes),
        'income': np.concatenate(income),
    }

    # Make this a dataframe
    agents=agent_data.copy()
    del agents['nodes']
    del agents['weights']
    agent_df=pd.DataFrame.from_dict(agents)
    for index, vi in enumerate(np.vstack(nodes).T): 
        agent_df[f'nodes{index}'] = vi
    agent_df['weights']=np.concatenate(weights).flatten()
    return agent_df


def save_pyblp_results(results, problem,filename):
    
    ## add in all the other things we could potentially be interested in
    
    res_dict = results.to_dict()
    
    res_dict['diversion_ratios'] = results.compute_diversion_ratios()
    res_dict['quality_diversion_ratios'] = results.compute_diversion_ratios(name=None)
    res_dict['own_diversion'] = results.extract_diagonals(res_dict['diversion_ratios'])
    res_dict['long_run_diversion_ratios'] = results.compute_long_run_diversion_ratios()
    
    res_dict['objective'] = results.objective.item()
    res_dict['objective_scaled'] = results.objective.item()/problem.N

    res_dict['elasticities'] = results.compute_elasticities()
    res_dict['aggregate_elasticities'] = results.compute_aggregate_elasticities()
    res_dict['diag_elasticities'] = results.extract_diagonals(res_dict['elasticities'])

    res_dict['consumer_surplus'] = results.compute_consumer_surpluses()
    res_dict['markups'] =results.compute_markups()

    res_dict['probabilities'] = results.compute_probabilities()
    
    np.save(filename, res_dict, allow_pickle =True)


def load_pyblp_dict(filename):
    dict = np.load(filename, allow_pickle=True)
    return dict

# this ONLY works for the base!
def load_blp_base(problem, filename):
    base_res = np.load(filename, allow_pickle=True)
    
    dict_W = base_res.item().get('W')
    dict_delta = base_res.item().get('delta')
    dict_gamma = base_res.item().get('gamma')
    dict_beta = base_res.item().get('beta')
    dict_sigma = base_res.item().get('sigma')
    dict_pi = base_res.item().get('pi')
    
    ## Use these to quickly get the exact results as estimation
    fast_options = dict(
        method='1s',
        check_optimality='gradient',
        costs_bounds=(0.001, None),
        W_type='clustered',
        se_type='clustered',
        initial_update=False,
        iteration=pyblp.Iteration('squarem', {'atol': 1e-14}),
        optimization=pyblp.Optimization('return'),
        scale_objective=False,
        W=dict_W,
        delta=dict_delta,
        beta=dict_beta,
        gamma=dict_gamma,
        sigma = dict_sigma,
        pi = dict_pi
        )

    results_fast = problem.solve(**fast_options)
    
    return results_fast



def get_params_nevo(results_dict, w=None):

    elasticities = results_dict.item().get('diag_elasticities')
    agg_elas = results_dict.item().get('aggregate_elasticities')

    diversion0 = results_dict.item().get('own_diversion')
    div = results_dict.item().get('diversion_ratios')
    
    div[np.isnan(div)]=0
    div[div==diversion0]=0
    div.sort(axis=1)
    top5=div[:,-5:].sum(axis=1)
    
    price_param = results_dict.item().get('beta').item()
    price_param_se = results_dict.item().get('beta_se').item()
    
    cs = results_dict.item().get('consumer_surplus')*100
    markups=results_dict.item().get('markups')
    # CRM: Adding the interactions as pi
    

    if results_dict.item().get('sigma').shape[0] == 0:
        sigmas = np.zeros(5)
        sigma_ses = np.zeros((5,5))
    else:
        sigma_ses = results_dict.item().get('sigma_se')
        sigmas=np.abs(np.diag(results_dict.item().get('sigma')))

    if results_dict.item().get('pi').shape[0] == 0 :
        pis = np.zeros((5,5))
        pi_ses = np.zeros((5,5))
    else:
        pis = results_dict.item().get('pi')
        pi_ses = results_dict.item().get('pi_se')
    
    objective = results_dict.item().get('objective')
    objective_scaled = results_dict.item().get('objective_scaled')

    return {'sigma_cons': sigmas[0],
            'sigma_price': sigmas[1],
            'sigma_sugar': sigmas[2],
            'sigma_mushy': sigmas[3],
            
            'sigma_cons_se': sigma_ses[0,0],
            'sigma_price_se': sigma_ses[1,1],
            'sigma_sugar_se': sigma_ses[2,2],
            'sigma_mushy_se': sigma_ses[3,3],
            
            'pi_cons_inc': pis[0,0],
            'pi_cons_inc2': pis[0,1],
            'pi_cons_age': pis[0,2],
            'pi_price_inc': pis[1,0],
            'pi_price_inc2': pis[1,1],
            'pi_price_child': pis[1,3],
            'pi_sugar_inc': pis[2,0],
            'pi_sugar_age': pis[2,2],
            'pi_mushy_inc': pis[3,0],
            'pi_mushy_age': pis[3,2],
            
            'pi_cons_inc_se': pi_ses[0,0],
            'pi_cons_inc2_se': pi_ses[0,1],
            'pi_cons_age_se': pi_ses[0,2],
            'pi_price_inc_se': pi_ses[1,0],
            'pi_price_inc2_se': pi_ses[1,1],
            'pi_price_child_se': pi_ses[1,3],
            'pi_sugar_inc_se': pi_ses[2,0],
            'pi_sugar_age_se': pi_ses[2,2],
            'pi_mushy_inc_se': pi_ses[3,0],
            'pi_mushy_age_se': pi_ses[3,2],            
            
            'price_coeff':  price_param,
            'price_se': price_param_se,
            
            'median_own_elas':np.median(elasticities),
            'median_agg_elas': np.median(agg_elas),
            'mean_og_div': np.average(diversion0,weights=w),
            'median_og_div': np.median(diversion0),
            'mean_top5_div': np.average(top5[:,None],weights=w),
            'mean_markup': np.average(markups,weights=w),
            'median_cs': np.median(cs),
            'objective': objective,
            'objective_scaled': objective_scaled,
            }


def get_params_blp(results_dict, w=None):
    
    elasticities = results_dict.item().get('diag_elasticities')
    agg_elas = results_dict.item().get('aggregate_elasticities')


    diversion0 = results_dict.item().get('own_diversion')
    div = results_dict.item().get('diversion_ratios')

    # set missing and outside good diversion =0
    div[np.isnan(div)]=0
    div[div==diversion0]=0
    div.sort(axis=1)
    top5=div[:,-5:].sum(axis=1)
    # why the difference? weird


    if results_dict.item().get('pi').shape[1]>0:
        price_param = results_dict.item().get('pi')[1][0]
    else:
        price_param = results_dict.item().get('beta')[1][0]

    price_se = results_dict.item().get('beta_se')[1][0]    
    cs = results_dict.item().get('consumer_surplus')
    markups = results_dict.item().get('markups')
    
    
    objective = results_dict.item().get('objective')
    objective_scaled = results_dict.item().get('objective_scaled')


    betas = results_dict.item().get('beta')[:,0]
    beta_ses = results_dict.item().get('beta_se')[:,0]

    sigmas=np.abs(np.diag(results_dict.item().get('sigma')))
    sigma_ses = np.abs(np.diag(results_dict.item().get('sigma_se')))

    if sigmas.shape[0] == 0:
        sigmas = np.zeros(6)
        sigma_ses = sigmas
    other_sigmas=sigmas[-4:]
    other_sigma_ses=sigma_ses[-4:]
    
    # if pis are suppressed or not
    if results_dict.item().get('pi').shape[1] == 0:
        pis = np.zeros((results_dict.item().get('pi').shape[0],results_dict.item().get('pi').shape[0]))
    else:
        pis = results_dict.item().get('pi')[:,0]    

    if results_dict.item().get('pi_se').shape[1] == 0:
        pi_ses = np.zeros(5)
    else:
        pi_ses = results_dict.item().get('pi_se')[:,0]    
        
        
        

    if results_dict.item().get('gamma').shape[0] == 0:
        gammas = np.zeros(6)
        gamma_ses = gammas
    else:
        gammas = results_dict.item().get('gamma')[:,0]
        gamma_ses = results_dict.item().get('gamma_se')[:,0]


    # defining the sigmas is weird
    
    sigma_cons = sigmas[0]
    sigma_hpwt = other_sigmas[0]
    sigma_air = other_sigmas[1]
    sigma_mpd = other_sigmas[2]
    sigma_size = other_sigmas[3]

    sigma_cons_se = sigma_ses[0]
    sigma_hpwt_se = other_sigma_ses[0]
    sigma_air_se = other_sigma_ses[1]
    sigma_mpd_se = other_sigma_ses[2]
    sigma_size_se = other_sigma_ses[3]  
        
    return {
            'coeff_cons':betas[0],
            'coeff_hpwt':betas[1],
            'coeff_air':betas[2],
            'coeff_mpd':betas[3],
            'coeff_size':betas[4],
            
            'se_cons':beta_ses[0],
            'se_hpwt':beta_ses[1],
            'se_air':beta_ses[2],
            'se_mpd':beta_ses[3],
            'se_size':beta_ses[4],
            
            
            'sigma_cons':sigma_cons,
            'sigma_hpwt':sigma_hpwt,
            'sigma_air':sigma_air,
            'sigma_mpd':sigma_mpd,
            'sigma_size':sigma_size,

            
            'sigma_cons_se':sigma_cons_se,
            'sigma_hpwt_se':sigma_hpwt_se,
            'sigma_air_se':sigma_air_se,
            'sigma_mpd_se':sigma_mpd_se,
            'sigma_size_se':sigma_size_se,

            #not TOTALLY sure this should be absolute value
            'price_term':price_param,
            'price_se': price_se,
            
            
            'gamma_cons':gammas[0],
            'gamma_hpwt':gammas[1],
            'gamma_air':gammas[2],
            'gamma_mpg':gammas[3],
            'gamma_size':gammas[4],
            'gamma_trend':gammas[5],
            
            
            'gamma_cons_se':gamma_ses[0],
            'gamma_hpwt_se':gamma_ses[1],
            'gamma_air_se':gamma_ses[2],
            'gamma_mpg_se':gamma_ses[3],
            'gamma_size_se':gamma_ses[4],
            'gamma_trend_se':gamma_ses[5],
            
            
            'median_own_elas':np.median(elasticities),
            'median_agg_elas': np.median(agg_elas),
            'mean_own_elas:': np.average(elasticities,weights=w),
            'median_og_div': np.median(diversion0),
            'mean_og_div': np.average(diversion0,weights=w),
            'median_top5_div': np.median(top5[:,None]),
            'mean_top5_div': np.average(top5[:,None],weights=w),
            'median_markup': np.median(markups),
            'mean_markup': np.average(markups,weights=w),
            'median_cs': np.median(cs),
            'objective': objective,
            'objective_scaled': objective_scaled,

            }



def make_df(x,stub):
    df=pd.DataFrame(x)
    df.columns=[stub+str(x) for x in df.columns]
    return df

# for each market, do the WTP calculations 
def do_single_market(results,product_data,ids):
    prodlist = product_data[product_data.market_ids.isin(ids)]['product_ids'].unique()
    base=results.compute_consumer_surpluses(keep_all=False,market_id=ids)
    wtp=np.vstack([base-results.compute_consumer_surpluses(eliminate_product_ids=[x],keep_all=False,market_id=ids) for x in prodlist]).flatten()
    div0=np.diag(results.compute_diversion_ratios(market_id=ids))
    shares=product_data[product_data.market_ids.isin(ids)]['shares'].values
    df=pd.DataFrame(np.vstack([wtp,div0,shares]).transpose(),columns=['wtp','div0','shares'])
    df['market_ids']=ids[0]
    df['product_ids']=product_data[product_data.market_ids.isin(ids)]['product_ids'].values
    return df

def do_single_market_indiv(results,product_data,ids):
    # get the relevant market and product IDs
    mktslice = product_data[product_data.market_ids.isin(ids)].copy()
    prodlist = mktslice['product_ids'].unique()

    # compute consumer surplus in the market WITH every product
    base=results.compute_consumer_surpluses(keep_all=True,market_id=ids)

    # WTP is surplus WITH (base) MINUS surplus without (eliminate)
    
    wtp=np.vstack([base-results.compute_consumer_surpluses(eliminate_product_ids=[x],keep_all=True,market_id=ids) for x in prodlist])

    # get diversion ratios
    div0=np.diag(results.compute_diversion_ratios(market_id=ids))
    
    # get market share for i j  t
    sijt = results.compute_probabilities(market_id=ids)
    
    # Dij,0
    div_i0=((1-sijt.sum(axis=0)[None,:])/(1-sijt))

    shares=sijt.mean(axis=1)


    df=pd.concat([make_df(wtp,'wtp_'), make_df(sijt,'sijt_'), make_df(div_i0,'divi0_')],axis=1)
    df['market_ids']=ids[0]
    df['product_ids']=product_data[product_data.market_ids.isin(ids)]['product_ids'].values
    return df


def reshape_wtp(wide_df):
    wide_df2=wide_df.set_index(['market_ids','product_ids'])
    tmp=wide_df2.filter(regex='wtp_').stack()
    draw_ids=np.array([int(str1.split('_')[1]) for  str1 in tmp.index.get_level_values(2)])

    long_df=pd.concat([
        tmp.reset_index(level=2,drop=True),
        wide_df2.filter(regex='sijt_').stack().reset_index(level=2,drop=True),
        wide_df2.filter(regex='divi0_').stack().reset_index(level=2,drop=True)
        ],axis=1)
    long_df.columns=['wtp','shares','div0']
    long_df['draw_ids']=draw_ids
    return long_df




def outreg(beta, sigma,names=None):
    # assume everything is in the right order
    # i won't do any rearranging here
    
    # create a new table by drawing from each
    paramnames = beta.index
    paramnames_se = sigma.index
    modelnames = beta.columns    
    
    # first, cut off each at three decimal places
    tab_beta = beta.round(decimals=3)
    tab_sigma= sigma.round(decimals=3)
    
    # fill in NAs and Zeroes:
    tab_sigma = tab_sigma.fillna('--')
    #tab_sigma = tab_sigma.fillna('--')
    tab_beta = tab_beta.replace(0, '--')
    tab_beta = tab_beta.replace(0.0, '--')
    
    
    #tab_beta = tab_beta.astype(str)
    tab_sigma = tab_sigma.astype(str)
    # replace the ZEROES with '--'
    # which requires first converting to string
    
    tab_new = pd.DataFrame()
    # strip the rownames
    for i in range(0, len(beta)):
        name_p = paramnames[i]
        name_s = paramnames_se[i]
        
        new_beta = tab_beta.loc[name_p]
        
        new_sigma = '(' + tab_sigma.loc[name_s] + ')'
        #new_sigma = f'({tab_sigma.loc[name_s] + ')'
        
        
        tab_new = tab_new.append(new_beta)
        tab_new = tab_new.append(new_sigma)
        
    # reset the index according to the paramnames
    if names == None:
        names = paramnames
    
    
    tab_new=tab_new.replace('(0.0)', '--')
    
    tab_new=tab_new.replace('(--)', '--')
    
    indexcol = []
    for i in range(0, len(beta)):
        print(names[i])
        indexcol = np.append(indexcol,names[i]) # for the beta
        indexcol = np.append(indexcol,' ') # for the sigma
    
    tab_new.index = indexcol
       
    return tab_new