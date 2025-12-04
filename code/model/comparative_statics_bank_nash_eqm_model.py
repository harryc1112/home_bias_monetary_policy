#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 12:23:17 2025

@author: harry_cooperman
"""

import numpy as np
import solve_bank_joint_nash_equilibrium as bank_nash_eqm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

# create output directories
if not(os.path.isdir('../../output')):
    os.mkdir('../../output')
    
if not(os.path.isdir('../../output/nash_model_comp_statics')):
    os.mkdir('../../output/nash_model_comp_statics')

"""
run comparative statics on r
"""

def comp_static_r(N_r = 30, r_low = 0.005, r_high = 0.075): 
    
    # set the phis to run comparative statics on
    r_s = np.linspace(r_low, r_high, N_r)
    
    # set the cross-sectional moments to store 
    avg_I_D = np.zeros(N_r)
    avg_I_at = np.zeros(N_r)
    avg_L_D = np.zeros(N_r)
    avg_rL = np.zeros(N_r)
    avg_rD = np.zeros(N_r)
    
    # set the aggregate moments to store
    agg_I_D = np.zeros(N_r)
    agg_I_at = np.zeros(N_r)
    agg_L_D = np.zeros(N_r)
    
    # get correlations
    corr_rL_rD = np.zeros(N_r)
    var_I_D = np.zeros(N_r)
    var_rL = np.zeros(N_r)
    var_rD = np.zeros(N_r)
    reg_I_D_on_r_exp = np.zeros(N_r)
    cov_rL_I_D = np.zeros(N_r)
    corr_rL_I_D = np.zeros(N_r)
    reg_rL_on_rD = np.zeros(N_r)
    
    for idx in range(N_r): 
        
        # get current model simulated equilibrium
        curr_out = bank_nash_eqm.simulate_and_solve(
            J=70, M=100, B=100, seed=123, equity_share = 0.17,
            r = r_s[idx], max_iter = 1000, max_bisect = 50
        )
        
        # get the current levels of balance sheet variables
        curr_D_j = curr_out['D_j']
        curr_L_j = curr_out['L_j']
        curr_I_j = curr_out['I_j']
        curr_E_j = curr_out['E_j']
        curr_rL_j = curr_out['rL']
        curr_rD_j = curr_out['rD']
        curr_r_fass = curr_out['sim_data']['r_fass_at']
        curr_r_comp = curr_out['sim_data']['r_comp_at']
        
        # create transformations of variables
        curr_I_D = curr_I_j / curr_D_j
        curr_I_at = curr_I_j / (curr_D_j + curr_E_j)
        curr_L_D = curr_L_j / curr_D_j
        curr_corr_rL_rD = np.corrcoef(curr_rL_j, curr_rD_j)[0, 1]
        
        # Run OLS of I/D on marginal costs
        X = np.column_stack([curr_r_fass + curr_r_comp])
        X = sm.add_constant(X)  # adds an intercept column
        y = curr_I_D
        model = sm.OLS(y, X).fit()
        
        # Run OLS of rL on rD 
        X = np.column_stack([curr_rD_j])
        X = sm.add_constant(X)
        y = curr_rL_j
        model2 = sm.OLS(y, X).fit()
        
        # get avarages
        avg_I_D[idx] = np.mean(curr_I_D)
        avg_I_at[idx] = np.mean(curr_I_at)
        avg_L_D[idx] = np.mean(curr_L_D)
        avg_rL[idx] = np.mean(curr_rL_j)
        avg_rD[idx] = np.mean(curr_rD_j)
        agg_I_D[idx] = np.sum(curr_I_j) / np.sum(curr_D_j)
        agg_I_at[idx] = np.sum(curr_I_j) / (np.sum(curr_D_j) + np.sum(curr_E_j))
        agg_L_D[idx] = np.sum(curr_L_j) / np.sum(curr_D_j)
        corr_rL_rD[idx] = curr_corr_rL_rD
        var_I_D[idx] = np.var(curr_I_D)
        reg_I_D_on_r_exp[idx] = model.params[1]
        reg_rL_on_rD[idx] = model2.params[1]
        var_rL[idx] = np.var(curr_rL_j)
        var_rD[idx] = np.var(curr_rD_j)
        cov_rL_I_D[idx] = np.cov(curr_rL_j, curr_I_D)[0, 1]
        corr_rL_I_D[idx] = np.corrcoef(curr_rL_j, curr_I_D)[0, 1]
        
        # update status
        print('Finished with iteration ' + str(idx))
        
    # return the moments
    return (r_s, avg_I_D, avg_I_at, avg_L_D, avg_rL, avg_rD, agg_I_D, agg_I_at, 
            agg_L_D, corr_rL_rD, var_I_D, reg_I_D_on_r_exp, var_rL, var_rD,
            cov_rL_I_D, corr_rL_I_D, reg_rL_on_rD)

# get results 
(r_s, avg_I_D_r, avg_I_at_r, avg_L_D_r, avg_rL_r, avg_rD_r, agg_I_D_r, agg_I_at_r, 
        agg_L_D_r, corr_rL_rD_r, var_I_D_r, reg_I_D_on_r_exp_r, var_rL_r, var_rD_r,
        cov_rL_I_D_r, corr_rL_I_D_r, reg_rL_on_rD_r) = comp_static_r()

# plot the comparative static of level of rates on loan dispersion
plt.plot(r_s, np.sqrt(var_rL_r))
plt.xlabel(r'$r$')
plt.ylabel(r'$\sigma(r_L)$')
plt.savefig('../../output/nash_model_comp_statics/comp_static_r_vs_var_rL.png')
plt.show()

# plot the comparative static of level of rates on deposit dispersion
plt.plot(r_s, np.sqrt(var_rD_r))
plt.xlabel(r'$r$')
plt.ylabel(r'$\sigma(r_D)$')
plt.savefig('../../output/nash_model_comp_statics/comp_static_r_vs_var_rD.png')
plt.show()

# plot the comparative static of level of rates on loan/deposit covariance
plt.plot(r_s, reg_rL_on_rD_r)
plt.xlabel(r'$r$')
plt.ylabel(r'$\beta$')
plt.savefig('../../output/nash_model_comp_statics/comp_static_r_vs_cov_rL_rD.png')
plt.show()

"""
run comparative statics on phi
"""

def comp_static_phi(N_phi = 30, phi_low = 0.01, phi_high = 2.0):
    
    # set the phis to run comparative statics on
    phis = np.linspace(phi_low, phi_high, N_phi)
    
    # set the cross-sectional moments to store 
    avg_I_D = np.zeros(N_phi)
    avg_I_at = np.zeros(N_phi)
    avg_L_D = np.zeros(N_phi)
    avg_rL = np.zeros(N_phi)
    avg_rD = np.zeros(N_phi)
    
    # set the aggregate moments to store
    agg_I_D = np.zeros(N_phi)
    agg_I_at = np.zeros(N_phi)
    agg_L_D = np.zeros(N_phi)
    
    # get correlations
    corr_rL_rD = np.zeros(N_phi)
    var_I_D = np.zeros(N_phi)
    var_rL = np.zeros(N_phi)
    var_rD = np.zeros(N_phi)
    reg_I_D_on_r_exp = np.zeros(N_phi)
    cov_rL_I_D = np.zeros(N_phi)
    corr_rL_I_D = np.zeros(N_phi)
    
    for idx in range(N_phi): 
        
        # get current model simulated equilibrium
        curr_out = bank_nash_eqm.simulate_and_solve(
            J=50, M=25, B=100, seed=123, equity_share = 0.17,
            phi = phis[idx], max_iter = 1000, max_bisect = 50
        )
        
        # get the current levels of balance sheet variables
        curr_D_j = curr_out['D_j']
        curr_L_j = curr_out['L_j']
        curr_I_j = curr_out['I_j']
        curr_E_j = curr_out['E_j']
        curr_rL_j = curr_out['rL']
        curr_rD_j = curr_out['rD']
        curr_r_fass = curr_out['sim_data']['r_fass_at']
        curr_r_comp = curr_out['sim_data']['r_comp_at']
        
        # create transformations of variables
        curr_I_D = curr_I_j / curr_D_j
        curr_I_at = curr_I_j / (curr_D_j + curr_E_j)
        curr_L_D = curr_L_j / curr_D_j
        curr_corr_rL_rD = np.corrcoef(curr_rL_j, curr_rD_j)[0, 1]
        
        # Run OLS of I/D on marginal costs
        X = np.column_stack([curr_r_fass + curr_r_comp])
        X = sm.add_constant(X)  # adds an intercept column
        y = curr_I_D
        model = sm.OLS(y, X).fit()
        
        # get avarages
        avg_I_D[idx] = np.mean(curr_I_D)
        avg_I_at[idx] = np.mean(curr_I_at)
        avg_L_D[idx] = np.mean(curr_L_D)
        avg_rL[idx] = np.mean(curr_rL_j)
        avg_rD[idx] = np.mean(curr_rD_j)
        agg_I_D[idx] = np.sum(curr_I_j) / np.sum(curr_D_j)
        agg_I_at[idx] = np.sum(curr_I_j) / (np.sum(curr_D_j) + np.sum(curr_E_j))
        agg_L_D[idx] = np.sum(curr_L_j) / np.sum(curr_D_j)
        corr_rL_rD[idx] = curr_corr_rL_rD
        var_I_D[idx] = np.var(curr_I_D)
        reg_I_D_on_r_exp[idx] = model.params[1]
        var_rL[idx] = np.var(curr_rL_j)
        var_rD[idx] = np.var(curr_rD_j)
        cov_rL_I_D[idx] = np.cov(curr_rL_j, curr_I_D)[0, 1]
        corr_rL_I_D[idx] = np.corrcoef(curr_rL_j, curr_I_D)[0, 1]
        
        # update status
        print('Finished with iteration ' + str(idx))
        
    # return the moments
    return (phis, avg_I_D, avg_I_at, avg_L_D, avg_rL, avg_rD, agg_I_D, agg_I_at, 
            agg_L_D, corr_rL_rD, var_I_D, reg_I_D_on_r_exp, var_rL, var_rD,
            cov_rL_I_D, corr_rL_I_D)

# get results 
(phis, avg_I_D, avg_I_at, avg_L_D, avg_rL, avg_rD, agg_I_D, agg_I_at, 
        agg_L_D, corr_rL_rD, var_I_D, reg_I_D_on_r_exp, var_rL, var_rD,
        cov_rL_I_D, corr_rL_I_D) = comp_static_phi()

# plot the comparative static that makes sense 
plt.plot(phis, avg_rL)
plt.xlabel(r'$\phi$')
plt.ylabel(r'$r_L$')
plt.savefig('../../output/nash_model_comp_statics/comp_static_phi_vs_avg_loan_rate.png')
plt.show()

# plot the comparative static of dispersion
plt.plot(phis, reg_I_D_on_r_exp)
plt.xlabel(r'$\phi$')
plt.ylabel(r'$\beta_{\frac{I}{D} \sim r_{EXP}}$')
plt.show()

"""
run comparative statics on lambda
"""

def comp_static_lambda(N_lambda = 30, lambda_low = 0.01, lambda_high = 0.7):
    
    # set the phis to run comparative statics on
    lambdas = np.linspace(lambda_low, lambda_high, N_lambda)
    
    # set the cross-sectional moments to store 
    avg_I_D = np.zeros(N_lambda)
    avg_I_at = np.zeros(N_lambda)
    avg_L_D = np.zeros(N_lambda)
    avg_rL = np.zeros(N_lambda)
    avg_rD = np.zeros(N_lambda)
    
    # set the aggregate moments to store
    agg_I_D = np.zeros(N_lambda)
    agg_I_at = np.zeros(N_lambda)
    agg_L_D = np.zeros(N_lambda)
    
    # get correlations
    corr_rL_rD = np.zeros(N_lambda)
    var_I_D = np.zeros(N_lambda)
    
    for idx in range(N_lambda): 
        
        # get current model simulated equilibrium
        curr_out = bank_nash_eqm.simulate_and_solve(
            J=35, M=18, B=100, seed=123, equity_share = 0.17,
            phi = 0.5, max_iter = 1000, max_bisect = 50, 
            lambda_liq = lambdas[idx]
        )
        
        # get the current levels of balance sheet variables
        curr_D_j = curr_out['D_j']
        curr_L_j = curr_out['L_j']
        curr_I_j = curr_out['I_j']
        curr_E_j = curr_out['E_j']
        curr_rL_j = curr_out['rL']
        curr_rD_j = curr_out['rD']
        
        # create transformations of variables
        curr_I_D = curr_I_j / curr_D_j
        curr_I_at = curr_I_j / (curr_D_j + curr_E_j)
        curr_L_D = curr_L_j / curr_D_j
        curr_corr_rL_rD = np.corrcoef(curr_rL_j, curr_rD_j)[0, 1]
        
        # get avarages
        avg_I_D[idx] = np.mean(curr_I_D)
        avg_I_at[idx] = np.mean(curr_I_at)
        avg_L_D[idx] = np.mean(curr_L_D)
        avg_rL[idx] = np.mean(curr_rL_j)
        avg_rD[idx] = np.mean(curr_rD_j)
        agg_I_D[idx] = np.sum(curr_I_j) / np.sum(curr_D_j)
        agg_I_at[idx] = np.sum(curr_I_j) / (np.sum(curr_D_j) + np.sum(curr_E_j))
        agg_L_D[idx] = np.sum(curr_L_j) / np.sum(curr_D_j)
        corr_rL_rD[idx] = curr_corr_rL_rD
        var_I_D[idx] = np.var(curr_I_D)
        
        # update status
        print('Finished with iteration ' + str(idx))
        
    # return the moments
    return (lambdas, avg_I_D, avg_I_at, avg_L_D, avg_rL, avg_rD, agg_I_D, agg_I_at, 
            agg_L_D, corr_rL_rD, var_I_D)

# get results 
(lambdas, avg_I_D_l, avg_I_at_l, avg_L_D_l, avg_rL_l, avg_rD_l, agg_I_D_l, agg_I_at_l, 
        agg_L_D_l, corr_rL_rD_l, var_I_D_l) = comp_static_lambda()

# plot the comparative static that makes sense 
plt.plot(lambdas, avg_I_D_l)
plt.plot(lambdas, lambdas, color = "red", linestyle = "dashed")
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\frac{I}{D}$')
plt.savefig('../../output/nash_model_comp_statics/comp_static_lambda_vs_cash_dep.png')
plt.show()

# plot the comparative static of dispersion
plt.plot(lambdas, var_I_D_l)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$Var(\frac{I}{D})$')
plt.show()

"""
run comparative statics on beta_L_fass
"""

def comp_static_beta_L_fass(N_beta = 30, beta_L_low = 0.01, beta_L_high = 1.0):
    
    # set the phis to run comparative statics on
    betas = np.linspace(beta_L_low, beta_L_high, N_beta)
    
    # set the cross-sectional moments to store 
    avg_I_D = np.zeros(N_beta)
    avg_I_at = np.zeros(N_beta)
    avg_L_D = np.zeros(N_beta)
    avg_rL = np.zeros(N_beta)
    avg_rD = np.zeros(N_beta)
    
    # set the aggregate moments to store
    agg_I_D = np.zeros(N_beta)
    agg_I_at = np.zeros(N_beta)
    agg_L_D = np.zeros(N_beta)
    
    # get correlations
    corr_rL_rD = np.zeros(N_beta)
    corr_rL_r_fass = np.zeros(N_beta)
    corr_rL_r_comp = np.zeros(N_beta)
    
    for idx in range(N_beta): 
        
        # get current model simulated equilibrium
        curr_out = bank_nash_eqm.simulate_and_solve(
            J=35, M=18, B=100, seed=123, equity_share = 0.17,
            phi = 0.5, max_iter = 1000, max_bisect = 50, 
            beta_L_fass = betas[idx], beta_L_comp = betas[idx]
        )
        
        # get the current levels of balance sheet variables
        curr_D_j = curr_out['D_j']
        curr_L_j = curr_out['L_j']
        curr_I_j = curr_out['I_j']
        curr_E_j = curr_out['E_j']
        curr_rL_j = curr_out['rL']
        curr_rD_j = curr_out['rD']
        curr_r_fass = curr_out['sim_data']['r_fass_at']
        curr_r_comp = curr_out['sim_data']['r_comp_at']
        
        # Build the design matrix with constant term
        X = np.column_stack([curr_r_fass + curr_r_comp])
        X = sm.add_constant(X)  # adds an intercept column
        y = curr_rL_j
        
        # Run OLS
        model = sm.OLS(y, X).fit()
        
        # create transformations of variables
        curr_I_D = curr_I_j / curr_D_j
        curr_I_at = curr_I_j / (curr_D_j + curr_E_j)
        curr_L_D = curr_L_j / curr_D_j
        curr_corr_rL_rD = np.corrcoef(curr_rL_j, curr_rD_j)[0, 1]
        curr_corr_rL_r_fass = model.params[1] # np.corrcoef(curr_rL_j, curr_r_fass)[0, 1]
        # curr_corr_rL_r_comp = model.params[2] # np.corrcoef(curr_rL_j, curr_r_comp)[0, 1]
        
        # get avarages
        avg_I_D[idx] = np.mean(curr_I_D)
        avg_I_at[idx] = np.mean(curr_I_at)
        avg_L_D[idx] = np.mean(curr_L_D)
        avg_rL[idx] = np.mean(curr_rL_j)
        avg_rD[idx] = np.mean(curr_rD_j)
        agg_I_D[idx] = np.sum(curr_I_j) / np.sum(curr_D_j)
        agg_I_at[idx] = np.sum(curr_I_j) / (np.sum(curr_D_j) + np.sum(curr_E_j))
        agg_L_D[idx] = np.sum(curr_L_j) / np.sum(curr_D_j)
        corr_rL_rD[idx] = curr_corr_rL_rD
        corr_rL_r_fass[idx] = curr_corr_rL_r_fass
        # corr_rL_r_comp[idx] = curr_corr_rL_r_comp
        
        # update status
        print('Finished with iteration ' + str(idx))
        
    # return the moments
    return (betas, avg_I_D, avg_I_at, avg_L_D, avg_rL, avg_rD, agg_I_D, agg_I_at, 
            agg_L_D, corr_rL_rD, corr_rL_r_fass, corr_rL_r_comp)

# get results 
(betas, avg_I_D_b, avg_I_at_b, avg_L_D_b, avg_rL_b, avg_rD_b, agg_I_D_b, agg_I_at_b, 
        agg_L_D_b, corr_rL_rD_b, corr_rL_r_fass_b, corr_rL_r_comp_b) = comp_static_beta_L_fass()

# plot the comparative static that makes sense 
plt.plot(betas, corr_rL_r_fass_b)
plt.plot(betas, betas, color = "red", linestyle = "dashed")
plt.xlabel(r'$\beta_L^{FASS}$')
plt.ylabel(r'$(r_{cost}^T r_{cost})^{-1} r_{cost}^T r_L$')
plt.savefig('../../output/nash_model_comp_statics/comp_static_beta_vs_reg_beta_price.png')
plt.show()

"""
run comparative statics on beta_D_fass
"""

def comp_static_beta_D_fass(N_beta = 30, beta_D_low = 0.01, beta_D_high = 1.0):
    
    # set the phis to run comparative statics on
    betas = np.linspace(beta_D_low, beta_D_high, N_beta)
    
    # set the cross-sectional moments to store 
    avg_I_D = np.zeros(N_beta)
    avg_I_at = np.zeros(N_beta)
    avg_L_D = np.zeros(N_beta)
    avg_rL = np.zeros(N_beta)
    avg_rD = np.zeros(N_beta)
    
    # set the aggregate moments to store
    agg_I_D = np.zeros(N_beta)
    agg_I_at = np.zeros(N_beta)
    agg_L_D = np.zeros(N_beta)
    
    # get correlations
    corr_rL_rD = np.zeros(N_beta)
    corr_rD_r_fass = np.zeros(N_beta)
    corr_rD_r_comp = np.zeros(N_beta)
    
    for idx in range(N_beta): 
        
        # get current model simulated equilibrium
        curr_out = bank_nash_eqm.simulate_and_solve(
            J=35, M=18, B=100, seed=123, equity_share = 0.17,
            phi = 0.5, max_iter = 1000, max_bisect = 50, 
            beta_D_fass = betas[idx], beta_D_comp = betas[idx]
        )
        
        # get the current levels of balance sheet variables
        curr_D_j = curr_out['D_j']
        curr_L_j = curr_out['L_j']
        curr_I_j = curr_out['I_j']
        curr_E_j = curr_out['E_j']
        curr_rL_j = curr_out['rL']
        curr_rD_j = curr_out['rD']
        curr_r_fass = curr_out['sim_data']['r_fass_at']
        curr_r_comp = curr_out['sim_data']['r_comp_at']
        
        # Build the design matrix with constant term
        X = np.column_stack([curr_r_fass + curr_r_comp])
        X = sm.add_constant(X)  # adds an intercept column
        y = curr_rD_j
        
        # Run OLS
        model = sm.OLS(y, X).fit()
        
        # create transformations of variables
        curr_I_D = curr_I_j / curr_D_j
        curr_I_at = curr_I_j / (curr_D_j + curr_E_j)
        curr_L_D = curr_L_j / curr_D_j
        curr_corr_rL_rD = np.corrcoef(curr_rL_j, curr_rD_j)[0, 1]
        curr_corr_rD_r_fass = model.params[1] # np.corrcoef(curr_rL_j, curr_r_fass)[0, 1]
        # curr_corr_rD_r_comp = model.params[2] # np.corrcoef(curr_rL_j, curr_r_comp)[0, 1]
        
        # get avarages
        avg_I_D[idx] = np.mean(curr_I_D)
        avg_I_at[idx] = np.mean(curr_I_at)
        avg_L_D[idx] = np.mean(curr_L_D)
        avg_rL[idx] = np.mean(curr_rL_j)
        avg_rD[idx] = np.mean(curr_rD_j)
        agg_I_D[idx] = np.sum(curr_I_j) / np.sum(curr_D_j)
        agg_I_at[idx] = np.sum(curr_I_j) / (np.sum(curr_D_j) + np.sum(curr_E_j))
        agg_L_D[idx] = np.sum(curr_L_j) / np.sum(curr_D_j)
        corr_rL_rD[idx] = curr_corr_rL_rD
        corr_rD_r_fass[idx] = curr_corr_rD_r_fass
        # corr_rD_r_comp[idx] = curr_corr_rD_r_comp
        
        # update status
        print('Finished with iteration ' + str(idx))
        
    # return the moments
    return (betas, avg_I_D, avg_I_at, avg_L_D, avg_rL, avg_rD, agg_I_D, agg_I_at, 
            agg_L_D, corr_rL_rD, corr_rD_r_fass, corr_rD_r_comp)

# get results 
(betas_d, avg_I_D_bd, avg_I_at_bd, avg_L_D_bd, avg_rL_bd, avg_rD_bd, agg_I_D_bd, agg_I_at_bd, 
        agg_L_D_bd, corr_rL_rD_bd, corr_rD_r_fass_bd, corr_rD_r_comp_bd) = comp_static_beta_D_fass()

# plot the comparative static that makes sense 
plt.plot(betas_d, corr_rD_r_fass_bd)
plt.plot(betas_d, betas_d, color = "red", linestyle = "dashed")
plt.xlabel(r'$\beta_D^{FASS}$')
plt.ylabel(r'$(r_{cost}^T r_{cost})^{-1} r_{cost}^T r_D$')
plt.savefig('../../output/nash_model_comp_statics/comp_static_beta_vs_reg_beta_price_deposits.png')
plt.show()

"""
run comparative statics on kappaD0 = beta0
"""

def comp_static_kappaD0(N_beta = 30, beta_L_low = 0.0001, beta_L_high = 0.01):
    
    # set the phis to run comparative statics on
    betas = np.linspace(beta_L_low, beta_L_high, N_beta)
    
    # set the cross-sectional moments to store 
    avg_I_D = np.zeros(N_beta)
    avg_I_at = np.zeros(N_beta)
    avg_L_D = np.zeros(N_beta)
    avg_rL = np.zeros(N_beta)
    avg_rD = np.zeros(N_beta)
    
    # set the aggregate moments to store
    agg_I_D = np.zeros(N_beta)
    agg_I_at = np.zeros(N_beta)
    agg_L_D = np.zeros(N_beta)
    
    # get correlations
    corr_rL_rD = np.zeros(N_beta)
    corr_rL_r_fass = np.zeros(N_beta)
    corr_rL_r_comp = np.zeros(N_beta)
    
    for idx in range(N_beta): 
        
        # get current model simulated equilibrium
        curr_out = bank_nash_eqm.simulate_and_solve(
            J=35, M=18, B=100, seed=123, equity_share = 0.17,
            phi = 0.5, max_iter = 1000, max_bisect = 50, 
            kappaD0 = betas[idx], kappaL0 = betas[idx]
        )
        
        # get the current levels of balance sheet variables
        curr_D_j = curr_out['D_j']
        curr_L_j = curr_out['L_j']
        curr_I_j = curr_out['I_j']
        curr_E_j = curr_out['E_j']
        curr_rL_j = curr_out['rL']
        curr_rD_j = curr_out['rD']
        curr_r_fass = curr_out['sim_data']['r_fass_at']
        curr_r_comp = curr_out['sim_data']['r_comp_at']
        
        # Build the design matrix with constant term
        X = np.column_stack([curr_r_fass, curr_r_comp])
        X = sm.add_constant(X)  # adds an intercept column
        y = curr_rL_j
        
        # Run OLS
        model = sm.OLS(y, X).fit()
        
        # create transformations of variables
        curr_I_D = curr_I_j / curr_D_j
        curr_I_at = curr_I_j / (curr_D_j + curr_E_j)
        curr_L_D = curr_L_j / curr_D_j
        curr_corr_rL_rD = np.corrcoef(curr_rL_j, curr_rD_j)[0, 1]
        curr_corr_rL_r_fass = model.params[1] # np.corrcoef(curr_rL_j, curr_r_fass)[0, 1]
        curr_corr_rL_r_comp = model.params[2] # np.corrcoef(curr_rL_j, curr_r_comp)[0, 1]
        
        # get avarages
        avg_I_D[idx] = np.mean(curr_I_D)
        avg_I_at[idx] = np.mean(curr_I_at)
        avg_L_D[idx] = np.mean(curr_L_D)
        avg_rL[idx] = np.mean(curr_rL_j)
        avg_rD[idx] = np.mean(curr_rD_j)
        agg_I_D[idx] = np.sum(curr_I_j) / np.sum(curr_D_j)
        agg_I_at[idx] = np.sum(curr_I_j) / (np.sum(curr_D_j) + np.sum(curr_E_j))
        agg_L_D[idx] = np.sum(curr_L_j) / np.sum(curr_D_j)
        corr_rL_rD[idx] = curr_corr_rL_rD
        corr_rL_r_fass[idx] = curr_corr_rL_r_fass
        corr_rL_r_comp[idx] = curr_corr_rL_r_comp
        
        # update status
        print('Finished with iteration ' + str(idx))
        
    # return the moments
    return (betas, avg_I_D, avg_I_at, avg_L_D, avg_rL, avg_rD, agg_I_D, agg_I_at, 
            agg_L_D, corr_rL_rD, corr_rL_r_fass, corr_rL_r_comp)

# get results 
(kappaD0s, avg_I_D_k, avg_I_at_k, avg_L_D_k, avg_rL_k, avg_rD_k, agg_I_D_k, agg_I_at_k, 
        agg_L_D_k, corr_rL_rD_k, corr_rL_r_fass_k, corr_rL_r_comp_k) = comp_static_kappaD0()

# plot the comparative static that makes sense 
plt.plot(kappaD0s, avg_rD_k)
plt.xlabel(r'$\kappa^D_0$')
plt.ylabel(r'$r_D$')
plt.savefig('../../output/nash_model_comp_statics/comp_static_kappaD0_vs_avg_rD.png')
plt.show()
