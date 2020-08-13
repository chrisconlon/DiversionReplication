#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:32:47 2020

@author: chitra
"""
import pandas as pd
import pyblp
import numpy as np

def compute_mpc(results, product_df, probs, mkt_id, j_id):
    shares_ijt =results.compute_probabilities(market_id=[mkt_id])    
    quality_change = probs * ( 1 - probs)
    # 1b: alpha_i
    # use the compute_beta_i function here
    share0 = results.compute_shares(market_id=[mkt_id])
    alpha_bar = results.beta[0][0] ## the mean
    
    # collect the corresponding nodes from the agent data
    # using the long way aruond per diversion_plot.py
    market = pyblp.markets.results_market.ResultsMarket(
             results.problem,
             mkt_id,
             results._parameters,
             results.sigma,
             results.pi,
             results.rho,
             results.beta,
             results.delta)
    
    X2 = market.products.X2.copy()
    prices = market.products.prices.flatten()
    delta = results.delta
    adjustment = market.products.prices * alpha_bar
    
    nodes = market.agents.nodes
    demographics = market.agents.demographics
    
    Sigma  = results.sigma
    pi = results.pi
    # Matrix Multiplication: Sigma nu' + Pi d' (see pg C1)
    # which is in fact the whole coefficient vector beta
    # we are interested in the "first" entry of beta (starts at zero)
    beta_i = (Sigma @ nodes.T) + (pi @ demographics.T)
    beta_i[1,:] = beta_i[1,:] + alpha_bar
    alpha_i = beta_i[1,:]

    ## Thus, Marginal Price Change is:
    mpc = quality_change * (-alpha_i) # -alpha_bar)
    
    second_choice = probs
    
    return mpc, alpha_i, alpha_bar, quality_change, second_choice


