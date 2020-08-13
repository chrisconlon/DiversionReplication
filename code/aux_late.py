#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:49:27 2020

@author: chitra
"""
import pyblp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   

from aux_plot_config import *
# nu is the curvature parameter
# got this from past macro code <3
def curve_space(start, stop, M, nu, endpoint = True):
    a1 = start
    aM = stop
    a = np.zeros(M)
    a[0] = a1
    for k in range(0, M):
        a[k] = a1 + (aM - a1)*(((k)/(M-1))**nu)
        
    if endpoint == False:
        a = a[:-1]
    if endpoint == True:
        np.append(a, aM)
    return a  


# %%
def get_price_vector(results, shares0, prices0, mkt, j):
    i = 1
    TOL = 0.0001
    change = TOL * 50
    
    # compute the max price
    while (change > TOL):
        prices_new = prices0.copy()
        prices_new[j] = prices_new[j] * i # make prices larger and larger
    
        shares_new = results.compute_shares(prices = prices_new, market_id = [mkt])
        sharej_new = shares_new[j]
        sharej0 = shares0[j]
        change = sharej_new/sharej0
        i = i+1
    vec_prices = curve_space(prices0[j], prices0[j]*i, 1000, 3)
    vec_prices = vec_prices[1:]
    return vec_prices

def get_delta_vector(results, shares0, deltas0, mkt, j):
    i = 1
    TOL = 0.0001
    change = TOL * 50
    
    # compute the max price
    while (change > TOL):
        deltas_new = deltas0.copy()
        deltas_new[j] = deltas_new[j] - i ## make quality smaller and smaller
        # gotta be weird about it here
    
        shares_new = results.compute_shares(delta = deltas_new, market_id = [mkt])
        sharej_new = shares_new[j]
        sharej0 = shares0[j]
        change = sharej_new/sharej0
        #print(change)
        i = i+1
        
    vec_deltas = curve_space(deltas0[j] - i, deltas0[j] , 1000, 3, endpoint = False)
    return vec_deltas


# %%
def get_wald_points(results, vec_points, mkt, j, k, typ='price'):
    shares0 = results.compute_shares(market_id = [mkt])
    prices0 = results.compute_prices(market_id = [mkt])
    vec_wald = np.zeros(len(vec_points))
    vec_change = np.zeros(len(vec_points))
    
    vec_points_new = np.zeros(len(vec_points))
    for i in range(0, len(vec_points)):
        wald_i, change_i= get_single_wald(results, shares0, 
                                         prices0, vec_points[i], 
                                         mkt, j,k, typ=typ)
        vec_wald[i] = wald_i
        vec_change[i] = change_i
        
        vec_points_new[i] = vec_points[i]
        
   
    df_wald_points = pd.DataFrame({'wald': list(vec_wald), 
                              'change': list(vec_change),
                              'points': list(vec_points_new)
                              })

    df_wald_points = df_wald_points.sort_values('change').reset_index(drop=True)
    return df_wald_points
        
# %%
def get_wald(results, mkt, j, k,  typ = 'price'):
    
    # initial shares
    shares0 = results.compute_shares(market_id = [mkt])

    if typ == 'delta':
        deltas0 = results.compute_delta(market_id = [mkt])
        vec_deltas = get_delta_vector(results, shares0, deltas0, mkt, j)
        
        
        vec_wald = np.zeros(len(vec_deltas))
        vec_change = np.zeros(len(vec_deltas))
        
        for i in range(0, len(vec_deltas)):
            #print(vec_prices[i])
            wald_i, change_i = get_single_wald(results, shares0, 
                                                    deltas0, vec_deltas[i], 
                                                    mkt, j, k, typ='delta')
            #print(wald_i)
            vec_wald[i] = wald_i
        
            vec_change[i] = change_i
    
        
        df_wald = pd.DataFrame({'wald': list(vec_wald), 
                                  'change': list(vec_change),
                                  'deltas': list(vec_deltas)
                                  })

        df_wald = df_wald.sort_values('change').reset_index(drop=True)
        
   
    else:
        # first: get the vector of prices you want to consider for a particular market, product
        # initial prices
        prices0 = results.compute_prices(market_id = [mkt])
        vec_prices = get_price_vector(results, shares0, prices0, mkt, j)
        vec_wald = np.zeros(len(vec_prices))
        vec_change = np.zeros(len(vec_prices))
        # now, evaluate the wald at each point specified:
    
        for i in range(0, len(vec_prices)):
            #print(vec_prices[i])
            wald_i, change_i = get_single_wald(results, shares0, 
                                                   prices0, vec_prices[i], 
                                                   mkt, j, k, typ='price')
            #print(wald_i)
            vec_wald[i] = wald_i
        
            vec_change[i] = change_i
    

        df_wald = pd.DataFrame({'wald': list(vec_wald), 
                                  'change': list(vec_change),
                                  'prices': list(vec_prices)
                                  })
        df_wald = df_wald.sort_values('change').reset_index(drop=True)
    
        # how shall we change these?
        # need to find an upper bound, and then scatter points in between those
        # look at multiples of the price?
    
    
    # k is the multiple at which the change is basically zero
        
    # define a vector of prices to evaluate the wald at
    # and see what that graph looks like
    # i don't REALLY want linearly spaced points, but we'll come back to this
    # 3 because we want a LOT of curvature


    return df_wald

# %%


def get_single_wald(results, shares0, charac, characj_new, mkt, j, k, typ='price'):
    
    if typ == 'delta':
        deltas_new = charac.copy()
        deltas_new[j] = characj_new # put in the new price for product j
        shares_new = results.compute_shares(delta = deltas_new, market_id = [mkt])
        
        sharek_new = shares_new[k]
        sharek0 = shares0[k]
        sharej_new = shares_new[j]
        sharej0 = shares0[j]    
    
        change = sharej_new/sharej0
        
        wald = -(sharek_new - sharek0)/(sharej_new-sharej0)        
    else:
        prices_new = charac.copy()
        prices_new[j] = characj_new # put in the new price for product j
        shares_new = results.compute_shares(prices = prices_new, market_id = [mkt])
        
        sharek_new = shares_new[k]
        sharek0 = shares0[k]
        sharej_new = shares_new[j]
        sharej0 = shares0[j]    
    
        change = sharej_new/sharej0
        
        wald = -(sharek_new - sharek0)/(sharej_new-sharej0)
    return wald, change

# %%

# Define a regular plotting method
    
def plot_late_single(results, products, mkt, j, k, savepath, savename=None):
    
    unique_ids = products.clustering_ids[products.market_ids == mkt].reset_index(drop=True)
    name_j = nice_name(unique_ids[j])
    print(name_j)
    name_k = nice_name(unique_ids[k])
    print(name_k)


    ### Make the plot associated with t, j, k
    # Line 1: the ATE/Second Choice
    # Line 2: Wald Estimator for a Change in Price
    # Line 3: Wald Estimator for a Change in Quantity
    
    # First, calculate the ATEs from pyBLP methods
    # this returns the full matrix
    # we will need to extract the [j,k] element of this for our diversion ratios
    ates = results.compute_long_run_diversion_ratios(market_id = [mkt])
    ate_jk = ates[j,k]
    del(ates) # save some memory I suppose



    ## Get the Price/Delta Lines:
    
    
    df_wald_price = get_wald(results, mkt, j, k,  typ='price')
    df_wald_delta = get_wald(results, mkt, j, k,  typ='delta')
    
    
    # Get the points for 5 and 10%:
    # need the initial price for j
    
    prices0 = results.compute_prices(market_id = [mkt])
    prices5 = prices0[j]*1.05
    prices10 = prices0[j]*1.10
    vec_points = [prices5, prices10]
    
    df_points_price = get_wald_points(results, vec_points, mkt, j, k, typ='price')
        
    ## FINALLY, make the plot itself!
    #plt.clf()
    fig, ax = plt.subplots(figsize=(20,20))
    # blue line: price changes
    ax.plot(df_wald_price.change, df_wald_price.wald, 'b', label='Price')
    # blue dots: 5% and 10%
    ax.plot(df_points_price.change, df_points_price.wald, 'b.', mew=10, ms = 10)
    # red line: quality changes
    ax.plot(df_wald_delta.change, df_wald_delta.wald, 'r', label='Quality')
    # single line for the ATE
    ax.hlines(ate_jk, 0, 1, ls='--', label='Second Choice')
    
    ax.set_xlim(1.02, -0.02)
    ax.set_xlabel(f'Fraction of Initial Share for {name_j}')
    ax.set_ylabel(f'Local Average Treatment Effect: Diversion from {name_j} to {name_k}')
    ax.legend()
    
    if savename == None:
        savename = f'wald_{j}_{k}.png'
    savefile = savepath / savename
    plt.savefig(savefile)
    
