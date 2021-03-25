#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:24:03 2020

@author: chitra
"""

import pandas as pd
import numpy as np
import pyblp
from aux_plot_config import *
from aux_compute_mpc import compute_mpc
from scipy import  ndimage

# %%
# Compare the diversion ratios from one product to two different ones
def plot_double_mpc(results, product_df, mkt_id, j_id, k_id1, k_id2, PARAMS_INT, PARAMS_STR, fig_dir):
    
    SIGMA = PARAMS_INT[0] # larger sigma is smoother curve
    N = PARAMS_INT[1] # number of simulated individuals # not used anywhere
    B = PARAMS_INT[2] # number of bins for the histogram

    MO = PARAMS_STR[0]
    MPC_TEXT = PARAMS_STR[1]
    QC_TEXT = PARAMS_STR[2]
    SC_TEXT = PARAMS_STR[3]


    mkt_df=product_df[product_df.market_ids==mkt_id]    
    
    ## Compute marginal, 2nd choice, and individual specific diversion
    diversion=results.compute_diversion_ratios(market_id=[mkt_id])
    diversion_2nd=results.compute_long_run_diversion_ratios(market_id=[mkt_id])    
    probs=results.compute_probabilities(market_id=[mkt_id])
    
    denom=1-probs
    div_mat=np.einsum('ji,ki->jki',1.0/denom,probs)

    j_name=mkt_df.iloc[j_id]['clustering_ids']
    j_lab = nice_name(j_name)
    
    k_name1=mkt_df.iloc[k_id1]['clustering_ids']
    k_lab1 = nice_name(k_name1)

    k_name2=mkt_df.iloc[k_id2]['clustering_ids']
    k_lab2 = nice_name(k_name2)
    ## Compute Marginal Price Change (comes w/ alpha_i)    
    mpc, alpha_i, alpha_bar, quality_change, second_choice = compute_mpc(
            results, product_df, probs, mkt_id,j_id)

    
    ## X-AXIS pretty much always the same:
    x=probs[j_id,:].copy() ## original share of HH i buying i
    idx = np.argsort(x)
    x_sort=x[idx]
    ## drop those points that are smaller than 0.01 -- they make the graph weird

    x_g = ndimage.gaussian_filter1d(x_sort, SIGMA,  mode='nearest')
    ## Plot TWO scatter plots on the same axis
    # it seems like they're all in sort of the same range anyways
    # for both k_id1 and k_id2
    ## Blue: Diversion from j to k1, MTE
    y1=div_mat[j_id,k_id1,:].copy()
    y1_sort = y1[idx]
    y1_g = ndimage.gaussian_filter1d(y1_sort, SIGMA, mode='nearest')
    ## Purple: Diversion from j to k2, MTE
    y2=div_mat[j_id,k_id2,:].copy()
    y2_sort = y2[idx]
    y2_g = ndimage.gaussian_filter1d(y2_sort, SIGMA, mode='nearest')
 
    # Blue Histogram: MPC, normalized
    mpc_norm = mpc[j_id,:].copy()/(mpc[j_id,:].sum())
    mpc_norm_sort = mpc_norm[idx] # sort to match

    pd_bin = pd.DataFrame((x_sort, mpc_norm_sort)).transpose()
    pd_bin.columns = ('x', 'mpc')
    # 51 points
    x_cut, x_bins = pd.cut(pd_bin.x, B, retbins = True, duplicates='drop')

    # sum up the corresponding y's -- got this off stackexchange
    hist_data = pd_bin.groupby([x_cut])[['mpc']].sum().reset_index()
    # 50 values associated with those point
    mpc_cut = hist_data['mpc']
    
    ## Red Histogram: QC / Second Choice
    sc_norm = second_choice[j_id,:].copy()/(second_choice[j_id,:].sum())
    sc_norm_sort = sc_norm[idx] # sort to match

    pd_bin = pd.DataFrame((x_sort, sc_norm_sort)).transpose()
    pd_bin.columns = ('x', 'sc')
    # 51 points
    x_cut, x_bins = pd.cut(pd_bin.x, B, retbins = True, duplicates='drop')

    # sum up the corresponding y's -- got this off stackexchange
    hist_data = pd_bin.groupby([x_cut])[['sc']].sum().reset_index()
    # 50 values associated with those point
    sc_cut = hist_data['sc']
    
    
    
    
    ## Plot the histogram 
    fig1, ax1 = plt.subplots(figsize=(15, 15))
        ## Plot the Histograms
    
    hist1 = ax1.hist(x_bins[0:50], bins=x_bins[0:50], weights= mpc_cut, 
             facecolor='navy', hatch="\\", alpha=0.7, label = MPC_TEXT,
             zorder = 0)
    hist2 = ax1.hist(x_bins[0:50], bins=x_bins[0:50], weights= sc_cut, 
             facecolor='maroon',hatch='/', alpha=0.5, label = SC_TEXT,
             zorder = 1)
    

    ax1.set_xlim((-0.0001,None))
    ax1.set_ylim((0,None))    
    ax1.set_ylabel("Weighting")
    ax1.set_xlabel('Individual $s_{ij}$ for '+j_lab )
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()
    handles_ax1, labels_ax1 =  ax1.get_legend_handles_labels()

    plt.legend(handles_ax1, labels_ax1,
               loc= 'upper center',
               bbox_to_anchor =(0.5, -0.05), frameon=True,
               ncol = 2,
               fontsize=20, fancybox=True)
    savename = 'hist_mte_'+ str(j_id) +'_' + str(k_id1) + '_' + str(k_id2) + '.pdf'
    savefile = fig_dir / savename
    plt.savefig(savefile, bbox_inches='tight')  

        
    
        
   # LINES/SCATTERPLOT second
    
    fig2, ax2 = plt.subplots(figsize=(15, 15))
    line1 = ax2.hlines(y=diversion[j_id,k_id1],xmin=x_sort.min(),xmax=x_sort.max(),
               linestyle='dotted',color='gray', linewidth=5, 
               label = k_lab1 + ' (Small Change) ',
               zorder = 2)
    line2 = ax2.hlines(y=diversion_2nd[j_id,k_id1],xmin=x_sort.min(),xmax=x_sort.max(),
               linestyle='dashed',color='gray', linewidth=5,
               label = k_lab1+ ' (Second Choice) ',
               zorder = 2)


    line3 = ax2.hlines(y=diversion[j_id,k_id2],xmin=x_sort.min(),xmax=x_sort.max(),
               linestyle='dotted',color='black', linewidth=5,
               label = k_lab2 + ' (Small Change)',
               zorder = 2)
    line4 = ax2.hlines(y=diversion_2nd[j_id,k_id2],xmin=x_sort.min(),xmax=x_sort.max(),
               linestyle='dashed',color='black', linewidth=5,
               label = k_lab2 + ' (Second Choice) ',
               zorder = 2)

    # Scatter plot last
    scatl1, = ax2.plot(x_g, y1_g, color='gray', zorder = 1,
                     linewidth = 5,
                       label = '$D_{jk,i}$ to ' + k_lab1)
    
    scatl2, = ax2.plot(x_g, y2_g, color='black', zorder = 1,
                     linewidth = 5, 
                       label = '$D_{jk,i}$ to ' + k_lab2)

    scat1 = ax2.scatter(x_sort, y1_sort, color='gray', marker='s', zorder = 4)
                      
                      
    scat2 = ax2.scatter(x_sort, y2_sort, color='black', marker='v', zorder = 4)
                   
    
    
    
    ax2.set_xlim((-0.0001,None))    
    ax2.set_ylim((0,None))        
    ax2.set_ylabel("Diversion Ratios")
    ax2.set_xlabel('Individual $s_{ij}$ for '+j_lab )    
    ax2.yaxis.set_label_position("right") #changed from left
    ax2.yaxis.tick_right() # changed from left
    handles_ax2, labels_ax2 =  ax2.get_legend_handles_labels()

    plt.legend(handles_ax2, labels_ax2,
               loc= 'upper center',
               bbox_to_anchor =(0.5, -0.05), frameon=True,
               ncol = 2,
               fontsize=20, fancybox=True)
    savename = 'lines_mte_'+ str(j_id) +'_' + str(k_id1) + '_' + str(k_id2) + '.pdf'
    savefile = fig_dir / savename
    plt.savefig(savefile, bbox_inches='tight')  
        
    

