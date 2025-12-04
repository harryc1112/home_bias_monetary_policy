#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Joint loan–deposit Nash equilibrium with:
- Loan utility: u^L_{bjm} = δ^L_j + δ^L_m + α_L θ_{bm} (c_L - (1 - e) r_{jL})
- Loan profit integrand: (r_{jL} θ - κ^L_j) s^L_{bjm} * (1 - e)
- Deposit utility: u^D_{jm} = δ^D_j + δ^D_m + α_D r_{jD} + β log(branches_{jm})
- Deposit outside option: invest at risk-free r ⇒ u^D_outside = α_D r ⇒ e0D = exp(α_D r)
- Soft liquidity constraint: penalty = -phi / 2 (I/D - λ)^(2), I = E + D - L

Created with significant help from ChatGPT, but revised by the authors in 
advance 
"""

import numpy as np
from numba import njit, prange
import statsmodels.api as sm

"""
helpers for computation purposes
"""

@njit(cache=True)
def _safe_log(x):
    # log with floor to avoid -inf and NaNs
    if x <= 0.0 or not np.isfinite(x):
        return -1.0e300  # ~ -inf but finite for numba
    return np.log(x)

@njit(cache=True)
def _safe_exp_clip(u, max_val = 1000.0):
    # avoid overflow in exp; exp(±50) is plenty large for probabilities
    if u > max_val:
        u = max_val
    elif u < -max_val:
        u = -max_val
    return np.exp(u)

"""
loan market pre-compute helpers
"""

# ---------------------- helpers: outside option for loans --------------------

@njit(cache=True, fastmath=True)
def _compute_outside_e0_BM(theta_BM, alpha_L, c_L,
                           vary_outside_L, equity_share):
    """
    Return e0_BM of shape (B,M) for the loan outside option.
    - If outside option varies, then e0 = exp( α_F * θ * c_L * equity_share )
    - else, e0 = 1
    """
    B, M = theta_BM.shape
    e0 = np.ones((B, M), dtype=np.float64)
    if vary_outside_L:
        for b in range(B):
            for m in range(M):
                e0[b, m] = np.exp(alpha_L * equity_share * c_L * theta_BM[b, m])
    return e0

# ---------------------------- loan side precompute ---------------------------

@njit(cache=True, fastmath=True)
def _loan_precompute_base(deltaL_m, theta_BM, alpha_L, c_L, equity_share):
    """
    Precompute base_BM and kappa_BM used on the loan side.
      base_BM[b,m]  = exp( δ^L_m + θ_{bm} * α_L * c_L )
      kappa_BM[b,m] = θ_{bm} * α_L * (1 - e)
    Shapes: base_BM, kappa_BM ∈ R^{B×M}
    """
    B, M = theta_BM.shape
    base_BM  = np.empty((B, M), dtype=np.float64)
    kappa_BM = np.empty((B, M), dtype=np.float64)
    for b in range(B):
        for m in range(M):
            th = theta_BM[b, m]
            base_BM[b, m]  = np.exp(deltaL_m[m] + th * (alpha_L * c_L))
            kappa_BM[b, m] = th * (alpha_L * (1.0 - equity_share))
    return base_BM, kappa_BM

@njit(parallel=True, cache=True, fastmath=True)
def _loan_build_S_all_BM(rL_vec, deltaL_j, base_BM, kappa_BM, partL_JM):
    """
    Parallel over markets: for each (b,m) accumulate Σ_j exp(δ_j) * exp(-kappa[b,m]*rL_j)
    only if bank j participates in market m. No races; each (b,m) is written once.
    """
    J = deltaL_j.size
    B, M = base_BM.shape
    wj_vec = np.empty(J, np.float64)
    for j in range(J):
        wj_vec[j] = np.exp(deltaL_j[j])

    S_all = np.zeros((B, M), dtype=np.float64)

    for m in prange(M):                  # ← parallel dimension
        for b in range(B):
            k = kappa_BM[b, m]
            s = 0.0
            for j in range(J):
                if partL_JM[j, m]:
                    s += wj_vec[j] * np.exp(-k * rL_vec[j])
            S_all[b, m] = base_BM[b, m] * s
    return S_all

# --------------------------- deposit side helpers ----------------------------

@njit(cache=True, fastmath=True)
def _precompute_expu_all_deposits(rD_old, deltaD_j, deltaD_m, alpha_D,
                                  beta_branch, log_branches_JM, partD_JM):
    """
    Precompute exp(u_j) for deposits at current rD_old and sum over j.
    u^D_{jm} = δ^D_j + δ^D_m + α_D r_{jD} + β log(branches_{jm}).
    Returns: sum_expu_all (M,), expu_old_j (J,M)
    """
    J = deltaD_j.size
    M = deltaD_m.size
    expu_old_j = np.zeros((J, M), dtype=np.float64)
    sum_expu_all = np.zeros(M, dtype=np.float64)

    for j in range(J):
        dj = deltaD_j[j]
        aj = alpha_D * (rD_old[j] - 1)
        for m in range(M):
            if partD_JM[j, m]:
                u = dj + deltaD_m[m] + aj + beta_branch * log_branches_JM[j, m]
                e = _safe_exp_clip(u)
            else:
                e = 0.0
            expu_old_j[j, m] = e
            sum_expu_all[m] += e
    return sum_expu_all, expu_old_j

"""
pre-compute the market participation identifiers
"""

# ---------- helpers to iterate only over markets a bank serves (CSR-style) --
def build_bank_market_indices(partJ_M: np.ndarray):
    """
    Return (idx, offsets) where:
      - idx is the concatenation of market indices for each bank (int64)
      - offsets[j]:offsets[j+1] gives the slice in idx for bank j
    """
    partJ_M = np.asarray(partJ_M)
    J, M = partJ_M.shape
    idx_list = []
    offsets = np.zeros(J + 1, dtype=np.int64)
    acc = 0
    for j in range(J):
        ms = np.flatnonzero(partJ_M[j]).astype(np.int64, copy=False)
        idx_list.append(ms)
        acc_next = acc + ms.size
        offsets[j+1] = acc_next
        acc = acc_next
    idx = np.concatenate(idx_list).astype(np.int64, copy=False) if idx_list else np.zeros(0, dtype=np.int64)
    return idx, offsets

"""
pre-compute the market participation indicators for the holding company version
of the model
"""

def build_holding_market_indices(partJ_M: np.ndarray,
                                 hc_offsets: np.ndarray,
                                 hc_bank_idx: np.ndarray):
    """
    For each holding (CSR described by hc_offsets/hc_bank_idx),
    return a CSR of the union of markets where ANY bank in the holding participates.

    Returns:
      h_m_idx: concatenation of market indices by holding (int64)
      h_off: CSR row ptrs (H+1,)
    """
    J, M = partJ_M.shape
    H = hc_offsets.shape[0] - 1
    h_m_lists = []
    for h in range(H):
        start, end = hc_offsets[h], hc_offsets[h+1]
        banks = hc_bank_idx[start:end]
        # union of markets across banks in holding
        mkts = np.flatnonzero(partJ_M[banks].any(axis=0)).astype(np.int64)
        h_m_lists.append(mkts)
    h_off = np.zeros(H + 1, dtype=np.int64)
    for h, mkts in enumerate(h_m_lists, start=1):
        h_off[h] = h_off[h-1] + mkts.size
    h_m_idx = (np.concatenate(h_m_lists).astype(np.int64) 
               if h_m_lists and h_off[-1] > 0 else np.zeros(0, dtype=np.int64))
    return h_m_idx, h_off

@njit(cache=True)
def _precompute_holding_deposit_terms(
    banks_in_holding,                # (Kh,)
    deltaD_j, deltaD_m, alpha_D, beta_branch,
    log_branches_JM, partD_JM,      # JxM
    sum_expu_all_M, expu_old_all,   # (M,), (J,M) at rD_old
    e0D, sizeD_M,                   # scalar e0D, (M,)
    h_m_idx                         # subset of markets for this holding (Mh,)
):
    """
    For each market in the holding, build:
      dep_G_h[m]      = sum_j exp(deltaD_j + beta * log(branches_jm)) over j in holding & participating
      dep_old_h[m]    = sum_j expu_old_all[j,m] over j in holding & participating (depends on rD_old)
    Returned arrays are length Mh; indices align to h_m_idx.
    """
    Mh = h_m_idx.size
    dep_G_h   = np.zeros(Mh, dtype=np.float64)
    dep_old_h = np.zeros(Mh, dtype=np.float64)
    for t in range(Mh):
        m = h_m_idx[t]
        G = 0.0
        O = 0.0
        for k in range(banks_in_holding.shape[0]):
            j = banks_in_holding[k]
            if partD_JM[j, m]:
                G += np.exp(deltaD_j[j] + beta_branch * log_branches_JM[j, m])
                O += expu_old_all[j, m]
        dep_G_h[t]   = G
        dep_old_h[t] = O
    return dep_G_h, dep_old_h

@njit(cache=True)
def _precompute_holding_loan_terms(
    banks_in_holding,                # (Kh,)
    deltaL_j,                        # (J,)
    base_BM, kappa_BM,               # (B,M), (B,M)
    partL_JM,                        # (J,M)
    rL_old,                          # (J,)
    S_all_BM,                        # (B,M) at rL_old
    h_m_idx                          # markets for this holding (Mh,)
):
    """
    For each (b, m in holding) build:
      loan_wsum_h[m]  = sum_j wj * 1[partL] (wj = exp(deltaL_j[j]))
      old_sum_h[b,m]  = sum_j wj * exp(-kappa_BM[b,m]*rL_old[j]) * 1[partL]
    We only return slices for markets in the holding.
    """
    B = base_BM.shape[0]
    Mh = h_m_idx.size
    loan_wsum_h = np.zeros(Mh, dtype=np.float64)
    old_sum_h   = np.zeros((B, Mh), dtype=np.float64)
    for t in range(Mh):
        m = h_m_idx[t]
        wsum = 0.0
        for k in range(banks_in_holding.shape[0]):
            j = banks_in_holding[k]
            if partL_JM[j, m]:
                wj = np.exp(deltaL_j[j])
                wsum += wj
        loan_wsum_h[t] = wsum
        # now old contribution by borrower draw b
        for b in range(B):
            s_old = 0.0
            kb = kappa_BM[b, m]
            bb = base_BM[b, m]
            for k in range(banks_in_holding.shape[0]):
                j = banks_in_holding[k]
                if partL_JM[j, m]:
                    wj = np.exp(deltaL_j[j])
                    s_old += bb * wj * np.exp(-kb * rL_old[j])
            old_sum_h[b, t] = s_old
    return loan_wsum_h, old_sum_h

# ----------------------- shares, volumes, diagnostics ------------------------

"""
compute loan and deposit shares given prices
"""

@njit(cache=True, fastmath=True, parallel=True)
def _loan_shares_and_moments(rL_vec, deltaL_j, deltaL_m, alpha_L, c_L,
                             theta_BM, partL_JM, e0_BM, equity_share):
    """
    Return per-market loan shares and moments without any (J×B×M) temporary.
    """
    J = deltaL_j.size
    B, M = theta_BM.shape

    S_L_MJ         = np.zeros((M, J), dtype=np.float64)
    Eths_L_MJ      = np.zeros((M, J), dtype=np.float64)
    Eths_oms_L_MJ  = np.zeros((M, J), dtype=np.float64)
    Eth2s_oms_L_MJ = np.zeros((M, J), dtype=np.float64)

    # Precompute and build S_all at the given rL_vec
    base_BM, kappa_BM = _loan_precompute_base(deltaL_m, theta_BM, alpha_L, c_L, equity_share)
    S_all_BM = _loan_build_S_all_BM(rL_vec, deltaL_j, base_BM, kappa_BM, partL_JM)

    for j in prange(J):
        wj  = np.exp(deltaL_j[j])
        rLj = rL_vec[j]

        # accumulate over B, markets where (j,m) participates
        for b in range(B):
            for m in range(M):
                if partL_JM[j, m]:
                    # term for bank j at (b, m) at the current rLj
                    term = base_BM[b, m] * wj * np.exp(-kappa_BM[b, m] * rLj)
                    # denominator at the evaluation point is just outside + total sum over all banks
                    denom = e0_BM[b, m] + S_all_BM[b, m]
                    s = term / denom
                else:
                    s = 0.0

                oms = 1.0 - s
                th  = theta_BM[b, m]
                S_L_MJ[m, j]        += s
                Eths_L_MJ[m, j]     += th * s
                Eths_oms_L_MJ[m, j] += th * s * oms
                Eth2s_oms_L_MJ[m, j]+= th * th * s * oms

        # average over draws
        invB = 1.0 / float(B)
        for m in range(M):
            S_L_MJ[m, j]        *= invB
            Eths_L_MJ[m, j]     *= invB
            Eths_oms_L_MJ[m, j] *= invB
            Eth2s_oms_L_MJ[m, j]*= invB

    return S_L_MJ, Eths_L_MJ, Eths_oms_L_MJ, Eth2s_oms_L_MJ

@njit(cache=True, fastmath=True, parallel=True)
def _deposit_shares_and_moments(rD_vec, deltaD_j, deltaD_m, alpha_D, beta_branch,
                                log_branches_JM, partD_JM, e0D):
    """
    Return per-market deposit shares and s(1-s) for own-price derivatives.
      S_D_MJ: (M,J) shares
      S_Doms_MJ: (M,J) s_j (1 - s_j)
    """
    J = deltaD_j.size
    M = deltaD_m.size
    S_D_MJ    = np.zeros((M, J), dtype=np.float64)
    S_Doms_MJ = np.zeros((M, J), dtype=np.float64)

    sum_expu_all_M, expu_old_all = _precompute_expu_all_deposits(
        rD_vec, deltaD_j, deltaD_m, alpha_D, beta_branch, log_branches_JM, partD_JM
    )

    for j in prange(J):
        for m in range(M):
            if partD_JM[j, m]:
                u = deltaD_j[j] + deltaD_m[m] + alpha_D * (rD_vec[j] - 1) + beta_branch * log_branches_JM[j, m]
                e_new = _safe_exp_clip(u)
                denom = e0D + (sum_expu_all_M[m] - expu_old_all[j, m] + e_new)
                s = e_new / denom
            else:
                s = 0.0
            S_D_MJ[m, j] = s
            S_Doms_MJ[m, j] = s * (1.0 - s)
    return S_D_MJ, S_Doms_MJ

"""
solve the joint FOCs for the deposit and loan rates at a bank
"""

@njit(cache=True, fastmath=True)
def _dd_marginal_cost_dep(phi, I_D, lam):
    # Your marginal per-unit DD object (same used inside FOCs and inversion)
    G = I_D - lam
    return phi * G * (1 - 0.5 * (lam + I_D)) # + 0.5 * phi * (G * G - 2.0 * G * I_D)

@njit(cache=True, fastmath=True)
def _joint_focs_for_bank(
    rLj, rDj, j,
    # market index helpers
    bank_mL_idx, offL, bank_mD_idx, offD,
    # loan side caches
    deltaL_j, deltaL_m, alpha_L, c_L, theta_BM, sizeL_M, partL_jM, e0_BM,
    base_BM, kappa_BM, S_all_BM, rL_old_j, equity_share,
    # deposit side caches
    deltaD_j, deltaD_m, alpha_D, beta_branch, log_branches_JM,
    sizeD_M, partD_jM, e0D, sum_expu_all_M, expu_old_j_M,
    # costs/params
    kappaL_j, kappaD_j, R, E_j, lambda_liq, phi
):
    """
    Return (F_L, F_D) for bank j at a candidate pair (rLj, rDj),
    with D_j and L_j computed at those SAME prices.
    """
    
    B, M = theta_BM.shape
    
    # Compute the bank's deposit size, given their choice of prices
    D_cand = 0.0
    sD = np.zeros(M, dtype=np.float64)
    startD, endD = offD[j], offD[j+1]
    for t in range(startD, endD):
        m = bank_mD_idx[t]
        u = deltaD_j[j] + deltaD_m[m] + alpha_D * (rDj - 1.0) + beta_branch * log_branches_JM[j, m]
        e_new = _safe_exp_clip(u)
        denom = e0D + (sum_expu_all_M[m] - expu_old_j_M[m] + e_new)
        s = e_new / denom
        sD[m] = s
        D_cand += sizeD_M[m] * s
    D_safe = max(D_cand, 1e-4)

    # store important parameters for the loan size
    invB = 1.0 / float(B)
    one_minus_e = 1.0 - equity_share
    wj = np.exp(deltaL_j[j])
   
    # construct the loan size, given the interest rate choice 
    L_cand = 0.0
    sum_s_by_m = np.zeros(M)
    sum_th_s_by_m      = np.zeros(M)
    sum_th_s_oms_by_m  = np.zeros(M)
    sum_th2_s_oms_by_m = np.zeros(M)
    
    startL, endL = offL[j], offL[j+1]
    for b in range(B):
        for t in range(startL, endL):
            m = bank_mL_idx[t]
            old_term = base_BM[b, m] * wj * np.exp(-kappa_BM[b, m] * rL_old_j)
            new_term = base_BM[b, m] * wj * np.exp(-kappa_BM[b, m] * rLj)
            denom = e0_BM[b, m] + (S_all_BM[b, m] - old_term) + new_term
            s = new_term / denom
            sum_s_by_m[m] += s
            oms = 1.0 - s
            th  = theta_BM[b, m]
            sum_th_s_by_m[m]      += th * s
            sum_th_s_oms_by_m[m]  += th * s * oms
            sum_th2_s_oms_by_m[m] += th * th * s * oms

    for m in range(M):
        sum_s_by_m[m] *= invB
        L_cand += one_minus_e * sizeL_M[m] * sum_s_by_m[m]

    # create the Diamond-Dybvig loan costs
    I_cand = E_j + D_safe - L_cand
    I_D = I_cand / D_safe
    DD_mc_loan = phi * (I_D - lambda_liq)
    DD_mc_dep = _dd_marginal_cost_dep(phi, I_D, lambda_liq)

    # ----- Loan FOC (use the pre-accumulated moments structure)
    # Recompute the exact moments for F_L:        
    F_L = 0.0
    for m in range(M):
        a1 = (sum_th_s_by_m[m] * invB)
        a2 = (sum_th_s_oms_by_m[m] * invB)
        a3 = (sum_th2_s_oms_by_m[m] * invB)
        w  = sizeL_M[m]
        F_L += w * one_minus_e * (
            a1 
            + alpha_L * (one_minus_e) * a2 * (kappaL_j + R - DD_mc_loan)
            - alpha_L * (one_minus_e) * a3 * rLj
        )

    # ----- Deposit FOC (at same x and DD_mc)
    F_D = 0.0
    for m in range(M):
        s = sD[m]
        oms = 1.0 - s
        F_D += sizeD_M[m] * s * (alpha_D * oms * (R - rDj - kappaD_j - DD_mc_dep) - 1.0)

    return F_L, F_D

"""
find the pair of (rL, rD) that solves the bank's joint FOC
"""

@njit(cache=True, fastmath=True)
def _solve_bank_joint_prices_newton(
    rLj0, rDj0, j,
    # market indicator helpers
    bank_mL_idx, offL, bank_mD_idx, offD,
    # pass-through of all caches/params required by _joint_focs_for_bank
    deltaL_j, deltaL_m, alpha_L, c_L, theta_BM, sizeL_M, partL_jM, e0_BM,
    base_BM, kappa_BM, S_all_BM, rL_old_j, equity_share,
    deltaD_j, deltaD_m, alpha_D, beta_branch, log_branches_JM,
    sizeD_M, partD_jM, e0D, sum_expu_all_M, expu_old_j_M,
    kappaL_j, kappaD_j, R, E_j, lambda_liq, phi,
    max_iter = 500, tol = 1e-10, rL_min = 1.0, rL_max = 2.0,
    rD_min = 1.0, rD_max = 2.0
):
    rLj = rLj0
    rDj = rDj0
    err = 10.0
    for it in range(max_iter):
        F_L, F_D = _joint_focs_for_bank(
            rLj, rDj, j,
            bank_mL_idx, offL, bank_mD_idx, offD,
            deltaL_j, deltaL_m, alpha_L, c_L, theta_BM, sizeL_M, partL_jM, e0_BM,
            base_BM, kappa_BM, S_all_BM, rL_old_j, equity_share,
            deltaD_j, deltaD_m, alpha_D, beta_branch, log_branches_JM,
            sizeD_M, partD_jM, e0D, sum_expu_all_M, expu_old_j_M,
            kappaL_j, kappaD_j, R, E_j, lambda_liq, phi
        )
        err = np.abs(F_L) + np.abs(F_D)
        if err < tol:
            break

        # numerical Jacobian (2x2)
        hL = 1e-6 * (np.abs(rLj) + 1.0)
        hD = 1e-6 * (np.abs(rDj) + 1.0)

        FLp, FDp = _joint_focs_for_bank(rLj + hL, rDj, j,
            bank_mL_idx, offL, bank_mD_idx, offD,
            deltaL_j, deltaL_m, alpha_L, c_L, theta_BM, sizeL_M, partL_jM, e0_BM,
            base_BM, kappa_BM, S_all_BM, rL_old_j, equity_share,
            deltaD_j, deltaD_m, alpha_D, beta_branch, log_branches_JM,
            sizeD_M, partD_jM, e0D, sum_expu_all_M, expu_old_j_M,
            kappaL_j, kappaD_j, R, E_j, lambda_liq, phi
        )
        FLm, FDm = _joint_focs_for_bank(rLj, rDj + hD, j,
            bank_mL_idx, offL, bank_mD_idx, offD,
            deltaL_j, deltaL_m, alpha_L, c_L, theta_BM, sizeL_M, partL_jM, e0_BM,
            base_BM, kappa_BM, S_all_BM, rL_old_j, equity_share,
            deltaD_j, deltaD_m, alpha_D, beta_branch, log_branches_JM,
            sizeD_M, partD_jM, e0D, sum_expu_all_M, expu_old_j_M,
            kappaL_j, kappaD_j, R, E_j, lambda_liq, phi
        )
        J11 = (FLp - F_L) / hL
        J21 = (FDp - F_D) / hL
        J12 = (FLm - F_L) / hD
        J22 = (FDm - F_D) / hD

        # solve J * step = -F
        det = J11 * J22 - J12 * J21
        if np.abs(det) < 1e-12:
            # fallback: small gradient step
            stepL = -0.1 * F_L
            stepD = -0.1 * F_D
        else:
            stepL = (-F_L * J22 + F_D * J12) / det
            stepD = (-J11 * F_D + J21 * F_L) / det

        # damping/line search
        damp = 0.75
        rLj = min(max(rL_min, rLj + damp * stepL), rL_max)
        rDj = min(max(rD_min, rDj + damp * stepD), rD_max)
    return rLj, rDj, err, it

"""
solve the FOCs for a bank under a holding company

Note:   banks_in_holding will be a list/set of idx's that map to the set of 
        banks that are under control of a single holding company
"""

@njit(cache=True, fastmath=True)
def _joint_focs_for_bank_under_hc(
    rLj, rDj, banks_in_holding,
    # loan side caches
    deltaL_j, deltaL_m, alpha_L, c_L, theta_BM, sizeL_M, partL_JM, e0_BM,
    base_BM, kappa_BM, S_all_BM, rL_old, equity_share,
    # deposit side caches
    deltaD_j, deltaD_m, alpha_D, beta_branch, log_branches_JM,
    sizeD_M, partD_JM, e0D, sum_expu_all_M, expu_old_M, 
    # costs/params
    kappaL_J, kappaD_J, R, E_J, lambda_liq, phi,
    # NEW: market sets / precomputes for this holding
    h_mL_idx, h_mD_idx,    # (MhL,), (MhD,)
    dep_G_h, dep_old_h,    # (MhD,), (MhD,)
    loan_wsum_h, old_sum_h # (MhL,), (B,MhL)
):
    """
    Same return as before, but with per-holding market restriction & precomputes.
    """
    B = theta_BM.shape[0]
    one_minus_e = 1.0 - equity_share
    invB = 1.0 / float(B)

    # E- and cost-averages used in the FOCs (unchanged)
    E_tot = 0.0
    for k in range(banks_in_holding.shape[0]):
        E_tot += E_J[banks_in_holding[k]]
    kL = 0.0; kD = 0.0
    for k in range(banks_in_holding.shape[0]):
        j = banks_in_holding[k]
        wE = E_J[j] / E_tot
        kL += wE * kappaL_J[j]
        kD += wE * kappaD_J[j]

    # ---------- Deposits: use dep_G_h and dep_old_h ----------
    D_cand = 0.0
    MhD = h_mD_idx.size
    sD = np.zeros(MhD, dtype=np.float64)
    common_dep = alpha_D * (rDj - 1.0)
    for t in range(MhD):
        m = h_mD_idx[t]
        # new numerator/denom for the holding’s combined product in market m
        e_new_total = np.exp(deltaD_m[m] + common_dep) * dep_G_h[t]
        denom = e0D + (sum_expu_all_M[m] - dep_old_h[t] + e_new_total)
        s = e_new_total / max(denom, 1e-300)
        sD[t] = s
        D_cand += sizeD_M[m] * s
    D_safe = max(D_cand, 1e-4)

    # ---------- Loans: use loan_wsum_h and old_sum_h ----------
    L_cand = 0.0
    MhL = h_mL_idx.size
    # For loan FOC moments:
    sum_s_by_m         = np.zeros(MhL, dtype=np.float64)
    sum_th_s_by_m      = np.zeros(MhL, dtype=np.float64)
    sum_th_s_oms_by_m  = np.zeros(MhL, dtype=np.float64)
    sum_th2_s_oms_by_m = np.zeros(MhL, dtype=np.float64)

    for t in range(MhL):
        m = h_mL_idx[t]
        # For each borrower draw
        for b in range(B):
            # new numerator for holding’s combined product at (b,m)
            new_sum_h_bm = base_BM[b, m] * np.exp(-kappa_BM[b, m] * rLj) * loan_wsum_h[t]
            # denom: replace old contrib of the holding with new one
            denom = e0_BM[b, m] + (S_all_BM[b, m] - old_sum_h[b, t] + new_sum_h_bm)
            s = new_sum_h_bm / max(denom, 1e-300)
            oms = 1.0 - s
            th  = theta_BM[b, m]
            sum_s_by_m[t]         += s
            sum_th_s_by_m[t]      += th * s
            sum_th_s_oms_by_m[t]  += th * s * oms
            sum_th2_s_oms_by_m[t] += th * th * s * oms

    for t in range(MhL):
        m = h_mL_idx[t]
        sum_s_by_m[t] *= invB
        L_cand += one_minus_e * sizeL_M[m] * sum_s_by_m[t]

    # ---------- Liquidity costs ----------
    I_cand = E_tot + D_safe - L_cand
    I_D = I_cand / D_safe
    DD_mc_loan = phi * (I_D - lambda_liq)
    DD_mc_dep  = _dd_marginal_cost_dep(phi, I_D, lambda_liq)

    # ---------- FOCs (aggregate over holding markets only) ----------
    F_L = 0.0
    for t in range(MhL):
        m = h_mL_idx[t]
        a1 = (sum_th_s_by_m[t] * invB)
        a2 = (sum_th_s_oms_by_m[t] * invB)
        a3 = (sum_th2_s_oms_by_m[t] * invB)
        w  = sizeL_M[m]
        F_L += w * one_minus_e * (
            a1 
            + alpha_L * one_minus_e * a2 * (kL + R - DD_mc_loan)
            - alpha_L * one_minus_e * a3 * rLj
        )

    F_D = 0.0
    for t in range(MhD):
        m = h_mD_idx[t]
        s = sD[t]
        oms = 1.0 - s
        F_D += sizeD_M[m] * s * (alpha_D * oms * (R - rDj - kD - DD_mc_dep) - 1.0)

    return F_L, F_D


"""
solve the joint pricing problem for a holding company 
"""

@njit(cache=True, fastmath=True)
def _solve_holding_company_joint_prices_newton(
    rLj0, rDj0, banks_in_holding,
    # pass-through of all caches/params required by _joint_focs_for_bank
    deltaL_j, deltaL_m, alpha_L, c_L, theta_BM, sizeL_M, partL_JM, e0_BM,
    base_BM, kappa_BM, S_all_BM, rL_old, equity_share,
    # deposit side caches
    deltaD_j, deltaD_m, alpha_D, beta_branch, log_branches_JM,
    sizeD_M, partD_JM, e0D, sum_expu_all_M, expu_old_M, 
    # costs/params
    kappaL_J, kappaD_J, R, E_J, lambda_liq, phi,
    # helpers
    h_mL_idx, h_mD_idx, dep_G_h, dep_old_h, loan_wsum_h, old_sum_h,
    # params
    max_iter = 500, tol = 1e-10, rL_min = 1.0, rL_max = 2.0,
    rD_min = 1.0, rD_max = 2.0
):
    # set the inital prices
    rLj = rLj0
    rDj = rDj0
    err = 10.0
    
    for it in range(max_iter):
        
        # calculate HC first order condition at given prices
        F_L, F_D = _joint_focs_for_bank_under_hc(
            rLj, rDj, banks_in_holding,
            deltaL_j, deltaL_m, alpha_L, c_L, theta_BM, sizeL_M, partL_JM, e0_BM,
            base_BM, kappa_BM, S_all_BM, rL_old, equity_share,
            deltaD_j, deltaD_m, alpha_D, beta_branch, log_branches_JM,
            sizeD_M, partD_JM, e0D, sum_expu_all_M, expu_old_M, 
            kappaL_J, kappaD_J, R, E_J, lambda_liq, phi,
            h_mL_idx, h_mD_idx, dep_G_h, dep_old_h, loan_wsum_h, old_sum_h
        )
        err = np.abs(F_L) + np.abs(F_D)
        if err < tol:
            break

        # numerical Jacobian (2x2)
        hL = 1e-6 * (np.abs(rLj) + 1.0)
        hD = 1e-6 * (np.abs(rDj) + 1.0)

        FLp, FDp = _joint_focs_for_bank_under_hc(
            rLj + hL, rDj, banks_in_holding,
            deltaL_j, deltaL_m, alpha_L, c_L, theta_BM, sizeL_M, partL_JM, e0_BM,
            base_BM, kappa_BM, S_all_BM, rL_old, equity_share,
            deltaD_j, deltaD_m, alpha_D, beta_branch, log_branches_JM,
            sizeD_M, partD_JM, e0D, sum_expu_all_M, expu_old_M, 
            kappaL_J, kappaD_J, R, E_J, lambda_liq, phi,
            h_mL_idx, h_mD_idx, dep_G_h, dep_old_h, loan_wsum_h, old_sum_h
        )
        FLm, FDm = _joint_focs_for_bank_under_hc(
            rLj, rDj + hD, banks_in_holding,
            deltaL_j, deltaL_m, alpha_L, c_L, theta_BM, sizeL_M, partL_JM, e0_BM,
            base_BM, kappa_BM, S_all_BM, rL_old, equity_share,
            deltaD_j, deltaD_m, alpha_D, beta_branch, log_branches_JM,
            sizeD_M, partD_JM, e0D, sum_expu_all_M, expu_old_M, 
            kappaL_J, kappaD_J, R, E_J, lambda_liq, phi,
            h_mL_idx, h_mD_idx, dep_G_h, dep_old_h, loan_wsum_h, old_sum_h
        )
        J11 = (FLp - F_L) / hL
        J21 = (FDp - F_D) / hL
        J12 = (FLm - F_L) / hD
        J22 = (FDm - F_D) / hD

        # solve J * step = -F
        det = J11 * J22 - J12 * J21
        if np.abs(det) < 1e-12:
            # fallback: small gradient step
            stepL = -0.1 * F_L
            stepD = -0.1 * F_D
        else:
            stepL = (-F_L * J22 + F_D * J12) / det
            stepD = (-J11 * F_D + J21 * F_L) / det

        # damping/line search
        damp = 0.75
        rLj = min(max(rL_min, rLj + damp * stepL), rL_max)
        rDj = min(max(rD_min, rDj + damp * stepD), rD_max)
    return rLj, rDj, err, it

"""
solve all of the banks problems in parallel
"""

# solve these in parallel
@njit(parallel=True, cache=True, fastmath=True)
def _update_rL_rD_bisect_parallel(rL_old, rD_old, 
                                  
                                  # helpers for market participation 
                                  bank_mL_idx, offL, bank_mD_idx, offD,
                                  
                                  # loans
                                  deltaL_j, deltaL_m, alpha_L, c_L, theta_BM, 
                                  sizeL_M, partL_JM, e0_BM, 
                                  base_BM, kappa_BM, S_all_BM, equity_share, 
                                  
                                  # deposits
                                  deltaD_j, deltaD_m, alpha_D, beta_branch, 
                                  log_branches_JM, sizeD_M, partD_JM, e0D, 
                                  sum_expu_all_M, expu_old_all,
                                  
                                  # inputs 
                                  kappaL_J, kappaD_J, R, E_J,
                               
                                  # DD constraint pieces
                                  phi, lambda_liq, 
                               
                                  # computation parameters 
                                  rL_min = 1.0, rL_max = 2.0,
                                  rD_min = 1.0, rD_max = 2.0):
    
    
    # get the sizes 
    J = rD_old.size
    
    # create a vector to store the loan and deposit prices
    rL_candidate = np.ones_like(rL_old)
    rD_candidate = np.ones_like(rD_old)
    errors = np.ones_like(rD_old)
    iterations = np.ones_like(rD_old)
    
    """
    update all banks using Newton methods to solve both FOC simultaneously
    """
    for j in prange(J):
        rLj_new, rDj_new, err, it_ct = _solve_bank_joint_prices_newton(
            rL_old[j], rD_old[j], j,
            # helpers for market participation 
            bank_mL_idx, offL, bank_mD_idx, offD,
            # loan side
            deltaL_j, deltaL_m, alpha_L, c_L, theta_BM, sizeL_M, partL_JM[j, :], e0_BM,
            base_BM, kappa_BM, S_all_BM, rL_old[j], equity_share,
            # deposit side
            deltaD_j, deltaD_m, alpha_D, beta_branch, log_branches_JM,
            sizeD_M, partD_JM[j, :], e0D, sum_expu_all_M, expu_old_all[j, :],
            # params
            kappaL_J[j], kappaD_J[j], R, E_J[j], lambda_liq, phi,
            max_iter = 100, tol=1e-6, rL_min = rL_min, rL_max = rL_max,  
            rD_min = rD_min, rD_max = rD_max
        )
        rL_candidate[j] = rLj_new
        rD_candidate[j] = rDj_new
        errors[j] = err
        iterations[j] = it_ct
    
    return rL_candidate, rD_candidate, errors, iterations

"""
solve the holding company problem in parallel
"""

def build_holding_csr(bank_to_holding: np.ndarray,
                      optimize_mask: np.ndarray):
    """
    Build CSR-style mapping from holding companies to their bank indices.

    Returns
    -------
    hc_ids : (H,) int64
        Sorted unique holding identifiers (values from bank_to_holding).
    hc_offsets : (H+1,) int64
        CSR row pointers; banks for hc h_idx live in
        hc_bank_idx[hc_offsets[h_idx]:hc_offsets[h_idx+1]].
    hc_bank_idx : (J,) int64
        Concatenation of bank indices grouped by holding.
    hc_optimize_mask : (H,) bool
        True if ANY bank in that holding is set to optimize.
    """
    bank_to_holding = np.asarray(bank_to_holding)
    optimize_mask   = np.asarray(optimize_mask, dtype=bool)

    # map each bank to [0..H-1]
    hc_ids, inv = np.unique(bank_to_holding, return_inverse=True)
    H = hc_ids.shape[0]

    # counts per holding
    counts = np.bincount(inv, minlength=H).astype(np.int64)

    # CSR offsets
    hc_offsets = np.empty(H + 1, dtype=np.int64)
    hc_offsets[0] = 0
    np.cumsum(counts, out=hc_offsets[1:])

    # fill bank indices, grouped by holding
    hc_bank_idx = np.empty(bank_to_holding.shape[0], dtype=np.int64)
    cursor = hc_offsets[:-1].copy()
    for bank_idx, h_idx in enumerate(inv):
        pos = cursor[h_idx]
        hc_bank_idx[pos] = bank_idx
        cursor[h_idx] = pos + 1

    # holding-level optimize flag (any bank optimizing)
    hc_optimize_mask = np.zeros(H, dtype=bool)
    for h_idx in range(H):
        start = hc_offsets[h_idx]; end = hc_offsets[h_idx+1]
        banks = hc_bank_idx[start:end]
        if banks.size > 0:
            hc_optimize_mask[h_idx] = np.any(optimize_mask[banks])

    return hc_ids.astype(np.int64), hc_offsets, hc_bank_idx, hc_optimize_mask

# solve these things in parallel
@njit(parallel=True, cache=True, fastmath=True)
def _update_rL_rD_bisect_parallel_with_holdings(rL_old, rD_old, 
                                  
                                  # loans
                                  deltaL_j, deltaL_m, alpha_L, c_L, theta_BM, 
                                  sizeL_M, partL_JM, e0_BM, 
                                  base_BM, kappa_BM, S_all_BM, equity_share, 
                                  
                                  # deposits
                                  deltaD_j, deltaD_m, alpha_D, beta_branch, 
                                  log_branches_JM, sizeD_M, partD_JM, e0D, 
                                  sum_expu_all_M, expu_old_all,
                                  
                                  # inputs 
                                  kappaL_J, kappaD_J, R, E_J,
                               
                                  # DD constraint pieces
                                  phi, lambda_liq, 
                                  
                                  # mergers and optimization control
                                  hc_offsets,          # (H+1,)
                                  hc_bank_idx,         # (J,)
                                  hc_optimize_mask,    # (H,)
                                  H,                   # scalar int
                                  
                                  h_mL_idx, h_mL_off,    # loan markets per holding
                                  h_mD_idx, h_mD_off,    # dep  markets per holding
                               
                                  # computation parameters 
                                  rL_min = 1.0, rL_max = 2.0,
                                  rD_min = 1.0, rD_max = 2.0):
    
    # Initialize outputs
    rL_new = rL_old.copy()
    rD_new = rD_old.copy()
    errors = np.zeros(H)
    iterations = np.zeros(H)
    
    # Process each holding company
    for h_idx in prange(H):
        
        # skip the holding companies that don't optimize
        if not hc_optimize_mask[h_idx]:
            continue  
        
        # for the others, get the list of subsidiary banks
        start = hc_offsets[h_idx]
        end   = hc_offsets[h_idx + 1]
        banks_in_holding = hc_bank_idx[start:end]
        
        # markets for this holding
        Ls, Le = h_mL_off[h_idx], h_mL_off[h_idx+1]
        Ds, De = h_mD_off[h_idx], h_mD_off[h_idx+1]
        hmL = h_mL_idx[Ls:Le]
        hmD = h_mD_idx[Ds:De]
        
        # precompute per-holding constant pieces (at r_old)
        dep_G_h, dep_old_h = _precompute_holding_deposit_terms(
            banks_in_holding, deltaD_j, deltaD_m, alpha_D, beta_branch,
            log_branches_JM, partD_JM, sum_expu_all_M, expu_old_all, e0D, sizeD_M, hmD
        )
        loan_wsum_h, old_sum_h = _precompute_holding_loan_terms(
            banks_in_holding, deltaL_j, base_BM, kappa_BM, partL_JM,
            rL_old, S_all_BM, hmL
        )
        
        # For holding companies, all banks in the holding use the same rates
        # Use the first bank's rates as the starting point for the holding
        first_bank = banks_in_holding[0]
        rL_holding = rL_old[first_bank]
        rD_holding = rD_old[first_bank]
        
        # Solve for optimal rates for this holding company
        rL_opt, rD_opt, error, iters = _solve_holding_company_joint_prices_newton(
            rL_holding, rD_holding, banks_in_holding,
            # pass-through of all caches/params required by _joint_focs_for_bank
            deltaL_j, deltaL_m, alpha_L, c_L, theta_BM, sizeL_M, partL_JM, e0_BM,
            base_BM, kappa_BM, S_all_BM, rL_old, equity_share,
            # deposit side caches
            deltaD_j, deltaD_m, alpha_D, beta_branch, log_branches_JM,
            sizeD_M, partD_JM, e0D, sum_expu_all_M, expu_old_all, 
            # costs/params
            kappaL_J, kappaD_J, R, E_J, lambda_liq, phi,
            # pre-compute helpers
            h_mL_idx = hmL, h_mD_idx = hmD,
            dep_G_h = dep_G_h, dep_old_h = dep_old_h,
            loan_wsum_h = loan_wsum_h, old_sum_h = old_sum_h,
            # params
            max_iter = 500, tol = 1e-6, rL_min = 1.0, rL_max = 2.0,
            rD_min = 1.0, rD_max = 2.0
        )
        
        # Apply the optimized rates to all banks in this holding company
        for k in range(banks_in_holding.shape[0]):
            j = banks_in_holding[k]
            rL_new[j] = rL_opt
            rD_new[j] = rD_opt
        errors[h_idx] = error
        iterations[h_idx] = iters
    
    return rL_new, rD_new, errors, iterations

"""
calculate depositor and borrower surplus
"""

@njit(cache=True, parallel=True, fastmath=True)
def depositor_surplus(rD_j, deltaD_j, deltaD_m, alpha_D, beta_branch,
                      log_branches_JM, sizeD_M, partD_JM, e0D,
                      optimize_mask=None, affected_markets=None):
    J, M = partD_JM.shape

    # precompute terms that don't depend on m
    a_j = np.empty(J, dtype=np.float64)
    for j in range(J):
        a_j[j] = np.exp(deltaD_j[j] + alpha_D * (rD_j[j] - 1.0))

    exp_deltaD_m = np.empty(M, dtype=np.float64)
    for m in range(M):
        exp_deltaD_m[m] = np.exp(deltaD_m[m])

    util_dep_market = np.empty(M, dtype=np.float64)

    # parallelize over markets
    for m in prange(M):
        G = 0.0
        for j in range(J):
            if partD_JM[j, m]:
                # exp(beta * log(branches)) = branches**beta, but we only have log, so exponentiate
                G += a_j[j] * np.exp(beta_branch * log_branches_JM[j, m])
        denom = e0D + exp_deltaD_m[m] * G
        if denom < 1e-300:
            denom = 1e-300
        util_dep_market[m] = (1.0 / alpha_D) * np.log(denom)

    wsum = 0.0
    total = 0.0
    for m in range(M):
        total += util_dep_market[m] * sizeD_M[m]
        wsum += sizeD_M[m]
    avg_all = total / max(wsum, 1e-300)

    # optimized subset (pass a precomputed market mask if you can)
    if optimize_mask is not None:
        # build once: markets with any optimized bank
        optimized_markets = np.zeros(M, dtype=np.uint8)
        for j in range(J):
            if optimize_mask[j]:
                for m in range(M):
                    if partD_JM[j, m]:
                        optimized_markets[m] = 1

        wsum2 = 0.0
        total2 = 0.0
        for m in range(M):
            if optimized_markets[m]:
                total2 += util_dep_market[m] * sizeD_M[m]
                wsum2 += sizeD_M[m]
        avg_opt = total2 / max(wsum2, 1e-300)
    else:
        avg_opt = avg_all
        
    # return welfare for the affected markets
    if affected_markets is not None: 
        wsum3 = 0.0
        total3 = 0.0
        for m in range(M): 
            if affected_markets[m]: 
                total3 += util_dep_market[m] * sizeD_M[m]
                wsum3 += sizeD_M[m]
        avg_aff = total3 / max(wsum3, 1e-300)
    else:
        avg_aff = avg_all

    return avg_all, avg_opt, avg_aff

@njit(cache=True, fastmath=True)
def _compute_w_b_j(alpha_L, theta_b, one_minus_e, rL_j, deltaL_j):
    J = rL_j.size
    w = np.empty(J, dtype=np.float64)
    k = alpha_L * theta_b * one_minus_e
    for j in range(J):
        w[j] = np.exp(deltaL_j[j] - k * rL_j[j])
    return w

@njit(cache=True, parallel=True, fastmath=True)
def borrower_surplus(rL_j, deltaL_j, deltaL_m, alpha_L, c_L, theta_BM,
                     sizeL_M, partL_JM, e0_BM, equity_share,
                     optimize_mask=None, affected_markets=None):
    J, M = partL_JM.shape
    B = theta_BM.shape[0]
    one_minus_e = 1.0 - equity_share

    exp_deltaL_m = np.empty(M, dtype=np.float64)
    for m in range(M):
        exp_deltaL_m[m] = np.exp(deltaL_m[m])

    # Precompute for each borrower draw b:
    #   - w_b[j] (J-vector)
    #   - scale_b = exp(alpha_L * theta_b * c_L)
    # We store in lists-of-arrays to keep memory predictable.
    w_list = [np.empty(1, dtype=np.float64) for _ in range(B)]
    scale = np.empty(B, dtype=np.float64)
    for b in range(B):
        theta_b = theta_BM[b, 0]  # theta_BM is (B,M), but theta_b is draw-specific only
        w_list[b] = _compute_w_b_j(alpha_L, theta_b, one_minus_e, rL_j, deltaL_j)
        scale[b] = np.exp(alpha_L * theta_b * c_L)

    util_loan_market = np.empty(M, dtype=np.float64)

    # Parallelize over markets
    for m in prange(M):
        accum = 0.0
        for b in range(B):
            w_b = w_list[b]
            Sb = 0.0
            for j in range(J):
                if partL_JM[j, m]:
                    Sb += w_b[j]
            denom = e0_BM[b, m] + exp_deltaL_m[m] * scale[b] * Sb
            if denom < 1e-300:
                denom = 1e-300
            accum += (1.0 / alpha_L) * np.log(denom)
        util_loan_market[m] = accum / B

    # Weighted averages
    wsum = 0.0
    total = 0.0
    for m in range(M):
        total += util_loan_market[m] * sizeL_M[m]
        wsum += sizeL_M[m]
    avg_all = total / max(wsum, 1e-300)

    if optimize_mask is not None:
        # build once: markets with any optimized bank
        optimized_markets = np.zeros(M, dtype=np.uint8)
        for j in range(J):
            if optimize_mask[j]:
                for m in range(M):
                    if partL_JM[j, m]:
                        optimized_markets[m] = 1

        wsum2 = 0.0
        total2 = 0.0
        for m in range(M):
            if optimized_markets[m]:
                total2 += util_loan_market[m] * sizeL_M[m]
                wsum2 += sizeL_M[m]
        avg_opt = total2 / max(wsum2, 1e-300)
    else:
        avg_opt = avg_all
        
    # return welfare for the affected markets
    if affected_markets is not None: 
        wsum3 = 0.0
        total3 = 0.0
        for m in range(M): 
            if affected_markets[m]: 
                total3 += util_loan_market[m] * sizeL_M[m]
                wsum3 += sizeL_M[m]
        avg_aff = total3 / max(wsum3, 1e-300)
    else:
        avg_aff = avg_all

    return avg_all, avg_opt, avg_aff

"""
calculate the size of the optimized markets
"""

def calc_optimized_market_size(params, partL_JM, partD_JM, sizeL_M, sizeD_M, 
                               optimize_mask):
    
    # get the model parameters
    J = params["J"]; M = params["M"]
    
    # get the size of the markets affected by the optimization
    if optimize_mask is not None:
        # build once: markets with any optimized bank
        optimized_loan_markets = np.zeros(M, dtype=np.uint8)
        optimized_dep_markets = np.zeros(M, dtype=np.uint8)
        for j in range(J):
            if optimize_mask[j]:
                for m in range(M):
                    if partL_JM[j, m]:
                        optimized_loan_markets[m] = 1
                    if partD_JM[j, m]: 
                        optimized_dep_markets[m] = 1
        # get the size of the optimized market
        optimize_loan_market_size = np.sum(sizeL_M[optimized_loan_markets])
        optimize_dep_market_size = np.sum(sizeD_M[optimized_loan_markets])
    else: 
        optimize_loan_market_size = np.sum(sizeL_M)
        optimize_dep_market_size = np.sum(sizeD_M)

    return optimize_loan_market_size, optimize_dep_market_size


"""
solve the equilibrium
"""

# initial guesses
def get_robust_initial_rates(deltaL_j, deltaL_m, deltaD_j, deltaD_m, 
                             kappaL_j, kappaD_j, model_params):
    """Get better initial guesses based on simple markup rules"""
    
    R = 1 + model_params['r']
    
    # For loans: start with cost-plus pricing
    # Account for expected loss and liquidity costs
    expected_loss_premium = 0.02  # 2% base premium
    liquidity_premium = model_params['phi'] * (model_params['lambda_liq'] - 0.3)
    
    rL_init = R + kappaL_j + expected_loss_premium + liquidity_premium
    
    # For deposits: start below risk-free rate
    deposit_margin = 0.001  # 1% margin
    rD_init = R - kappaD_j - deposit_margin
    
    # Bound the initial guesses
    rL_init = np.clip(rL_init, model_params['rL_min'], model_params['rL_max'])
    rD_init = np.clip(rD_init, model_params['rD_min'], model_params['rD_max'])
    
    return rL_init, rD_init

# ------------------------------ master solver --------------------------------

def solve_joint_eqm(
    params,
    deltaL_j, deltaL_m,
    deltaD_j, deltaD_m,
    branches_JM,
    partL_JM, partD_JM,
    theta_BM,
    sizeL_M, sizeD_M,
    kappaL_J, kappaD_J, E_J,
    
    # parameters
    rL_min = 1.0, rL_max = 2.0, 
    rD_min = 1.0, rD_max = 1.35,
    
    # options
    max_iter=1000, tol=1e-6, max_bisect=30,
    init_rL=None, init_rD=None,
    time_vary_rD_outside = True
):
    """
    Solve for (r_L, r_D) jointly with a soft liquidity constraint and the
    corrected loan-side structure described in the docstring.
    Returns a dict with equilibrium prices, shares, volumes, and diagnostics.
    """
    
    # get model parameters
    J = params["J"]
    alpha_L = float(params["alpha_L"]) ; alpha_D = float(params["alpha_D"])
    c_L = float(params["c_L"]); r = float(params["r"])
    lambda_liq = float(params["lambda_liq"])
    beta_branch = float(params["beta_branch"])
    phi = float(params["phi"])
    equity_share = float(params["equity_share"])
    vary_outside_L = params["vary_outside_L"]
    
    # set gross return rate
    R = 1 + r    
    
    # ensure that we have deposit and loan market fixed effects for each bank
    assert deltaL_j.size == J and deltaD_j.size == J

    # precompute log(branches)
    log_branches_JM = np.log(np.maximum(branches_JM, 1.0))

    # initialize (convert initial guesses to net if needed)
    rL_temp, rD_temp = get_robust_initial_rates(deltaL_j, deltaL_m, deltaD_j, deltaD_m, 
                                              kappaL_J, kappaD_J, params)
    
    # update guesses if values are seeded
    if init_rL is None:
        rL_old = rL_temp 
    else:
        rL_old = init_rL 
    if init_rD is None:
        rD_old = rD_temp
    else:
        rD_old = init_rD

    # Deposit outside good (constant)
    if time_vary_rD_outside:
        e0D =  _safe_exp_clip(alpha_D * (R - 1.0))
    else: 
        e0D = 1

    # get the loan outside good matrix
    e0_BM = _compute_outside_e0_BM(theta_BM, alpha_L, c_L,
                                   vary_outside_L, equity_share)
    
    # get the helpers for market participation 
    bank_mL_idx, offL = build_bank_market_indices(partL_JM)
    bank_mD_idx, offD = build_bank_market_indices(partD_JM)

    # Update first order conditions until convergence of the model
    for it in range(1, max_iter + 1):
        
        # build the values to helpfully pre-compute things        
        base_BM, kappa_BM = _loan_precompute_base(deltaL_m, theta_BM, alpha_L, c_L, equity_share)
        S_all_BM = _loan_build_S_all_BM(rL_old, deltaL_j, base_BM, kappa_BM, partL_JM)
        sum_expu_all_M, expu_old_all = _precompute_expu_all_deposits(
            rD_old, deltaD_j, deltaD_m, alpha_D, beta_branch, log_branches_JM, partD_JM
        )

        # guess loan prices
        rL_candidate, rD_candidate, errors, iterations = _update_rL_rD_bisect_parallel(
            
            # old prices
            rL_old, rD_old, 
            
            # indices for market participation
            bank_mL_idx, offL, bank_mD_idx, offD,
                                          
            # loans
            deltaL_j, deltaL_m, alpha_L, c_L, theta_BM, 
            sizeL_M, partL_JM, e0_BM, 
            base_BM, kappa_BM, S_all_BM, equity_share, 
            
            # deposits
            deltaD_j, deltaD_m, alpha_D, beta_branch, 
            log_branches_JM, sizeD_M, partD_JM, e0D, 
            sum_expu_all_M, expu_old_all,
            
            # inputs 
            kappaL_J, kappaD_J, R, E_J,
         
            # DD constraint pieces
            phi, lambda_liq, 
         
            # computation parameters 
            rL_min, rL_max,
            rD_min, rD_max)

        # Damped update of loan and deposit prices, for stability
        rL_new = 0.5 * rL_old + 0.5 * rL_candidate
        rD_new = 0.5 * rD_old + 0.5 * rD_candidate

        # Check convergence
        loan_gap = np.abs(rL_new - rL_old)
        dep_gap = np.abs(rD_new - rD_old)
        maxdiff = max(np.max(loan_gap), np.max(dep_gap))
        avgdiff = np.mean([np.mean(loan_gap), np.mean(dep_gap)])
        
        # update iteration
        print('processed iteration' + str(it) + ' | max error: ' + str(maxdiff) + ' | avg error: ' + str(avgdiff))
        
        rL_old, rD_old = rL_new, rD_new
        if maxdiff < tol:
            info = {"iterations": it, "converged": True, 
                    "max_dep_gap": np.max(dep_gap),
                    "max_loan_gap": np.max(loan_gap)}
            break
        else:
            info = {"iterations": max_iter, "converged": False,
                    "max_dep_gap": np.max(dep_gap),
                    "max_loan_gap": np.max(loan_gap)}


    ## ----- ##
    # Recompute shares at final prices
    S_L_MJ, Eths_L_MJ, Eths_oms_L_MJ, _Eth2s_oms_L_MJ = _loan_shares_and_moments(
        rL_new, deltaL_j, deltaL_m, alpha_L, c_L, theta_BM, partL_JM, e0_BM, equity_share
    )
    S_D_MJ, S_Doms_MJ = _deposit_shares_and_moments(
        rD_new, deltaD_j, deltaD_m, alpha_D, beta_branch, log_branches_JM, partD_JM, e0D
    )
    
    # BLAS-backed matrix–vector products
    L_j      = (sizeL_M @ S_L_MJ)            * (1.0 - equity_share)
    Eths_L_j = (sizeL_M @ Eths_L_MJ)         * (1.0 - equity_share)
    D_j      =  (sizeD_M @ S_D_MJ)
    I_j = E_J + D_j - L_j
    Loss_Rate_j = 1 - (Eths_L_j / L_j)
    ## ----- ##

    # Profits 
    loan_term = np.zeros(J, dtype=np.float64)
    for j in range(J): 
        loan_term[j] = rL_old[j] * Eths_L_j[j] - kappaL_J[j] * L_j[j]
    dep_term = np.zeros(J, dtype=np.float64)
    for j in range(J):
        dep_term[j] = - D_j[j] * (rD_old[j] + kappaD_J[j])
            
    # get profits
    profit_by_bank = loan_term + dep_term + R * I_j
    liq_cost = (phi / 2) * D_j * ((I_j / D_j) - lambda_liq) ** 2
    profit_by_bank_less_shadow_cost = profit_by_bank - liq_cost

    # calculate the deposit marginal costs
    B1 = sizeD_M @ S_D_MJ 
    B2 = sizeD_M @ S_Doms_MJ
    B1_over_B2 = B1 / B2
    I_D_j = I_j / D_j 
    G = I_D_j - lambda_liq
    Lambda_I = phi * G 
    Lambda_D = 0.5 * phi * (G ** 2 - 2.0 * G * I_D_j)
    DD_mc = Lambda_I + Lambda_D
    kappaD_J2 = ((R - rD_old)
            - (1 / alpha_D) * B1_over_B2
            - DD_mc)
     
    # --- Deposit FOC residuals at final prices, for BOTH kappas ---
    res_with_true = alpha_D * B2 * (R - rD_old - kappaD_J - DD_mc) - B1
    res_with_back = alpha_D * B2 * (R - rD_old - kappaD_J2 - DD_mc) - B1
    
    # (optional) max norms for quick printing/debug
    max_res_true = float(np.max(np.abs(res_with_true)))
    max_res_back = float(np.max(np.abs(res_with_back)))

    # get the depositor and borrower surplus
    avg_util_dep_market, _, _ = depositor_surplus(
        rD_old, deltaD_j, deltaD_m, alpha_D, beta_branch, 
        log_branches_JM, sizeD_M, partD_JM, e0D,
        optimize_mask = None)
    avg_util_loan_market, _, _ = borrower_surplus(
        rL_old, deltaL_j, deltaL_m, alpha_L, c_L, theta_BM, 
        sizeL_M, partL_JM, e0_BM, equity_share, 
        optimize_mask = None)

    return {
        "rL": rL_old,
        "rD": rD_old,
        "S_L_MJ": S_L_MJ,
        "E_theta_s_L_MJ": Eths_L_MJ,
        "S_D_MJ": S_D_MJ,
        "L_j": L_j,
        "Eths_L_j": Eths_L_j, 
        "Loss_Rate_j": Loss_Rate_j,
        "D_j": D_j,
        "E_j": E_J,
        "I_j": I_j,
        "kappaL_j": kappaL_J, 
        "kappaD_j": kappaD_J,
        "kappaD_j2": kappaD_J2,
        "profit_by_bank": profit_by_bank,
        "profit_by_bank_less_shadow_cost": profit_by_bank_less_shadow_cost,
        "info": info,
        "res_foc_dep_with_true": res_with_true,
        "res_foc_dep_with_back": res_with_back,
        "max_res_foc_dep_with_true": max_res_true,
        "max_res_foc_dep_with_back": max_res_back,
        "FOC_errors": errors,
        "final_iterations": iterations,
        "depositor_surplus": avg_util_dep_market, 
        "borrower_surplus": avg_util_loan_market 
    }

"""
solve the equilibrium, when we allow for holding companies as well
"""

def solve_joint_eqm_hc(
    params,
    deltaL_j, deltaL_m,
    deltaD_j, deltaD_m,
    branches_JM,
    partL_JM, partD_JM,
    theta_BM,
    sizeL_M, sizeD_M,
    kappaL_J, kappaD_J, E_J,
    
    # economic options: (1) Holding company toggle; (2) Optimization toggle
    bank_to_holding = None, optimize_mask = None, affected_markets=None,
    
    # parameters
    rL_min = 1.0, rL_max = 2.0, 
    rD_min = 1.0, rD_max = 1.35,
    
    # options
    max_iter=1000, tol=1e-6, max_bisect=30,
    init_rL=None, init_rD=None,
):
    """
    Solve for (r_L, r_D) jointly with a soft liquidity constraint and the
    corrected loan-side structure described in the docstring.
    Returns a dict with equilibrium prices, shares, volumes, and diagnostics.
    """
    
    # get model parameters
    J = params["J"]
    alpha_L = float(params["alpha_L"]) ; alpha_D = float(params["alpha_D"])
    c_L = float(params["c_L"]); r = float(params["r"])
    lambda_liq = float(params["lambda_liq"])
    beta_branch = float(params["beta_branch"])
    phi = float(params["phi"])
    equity_share = float(params["equity_share"])
    vary_outside_L = params["vary_outside_L"]
    
    # set gross return rate
    R = 1 + r    
    
    # ensure that we have deposit and loan market fixed effects for each bank
    assert deltaL_j.size == J and deltaD_j.size == J

    # precompute log(branches)
    log_branches_JM = np.log(np.maximum(branches_JM, 1.0))

    # Handle holding company mapping
    if bank_to_holding is None:
        bank_to_holding = np.arange(J)  # Each bank is its own holding company
    
    # Handle optimization mask
    if optimize_mask is None:
        optimize_mask = np.ones(J, dtype=bool)  # Optimize all banks by default
    
    # create the key parallelization variables that we can use to run the model 
    hc_ids, hc_offsets, hc_bank_idx, hc_optimize_mask = build_holding_csr(
        bank_to_holding=np.asarray(bank_to_holding, dtype=np.int64),
        optimize_mask=np.asarray(optimize_mask, dtype=bool)
        )
    H = hc_ids.shape[0]
    
    # build the parallelization variables that we can use to run the model with
    # a holding company
    h_mL_idx, h_mL_off = build_holding_market_indices(partL_JM, hc_offsets, hc_bank_idx)
    h_mD_idx, h_mD_off = build_holding_market_indices(partD_JM, hc_offsets, hc_bank_idx)

    # initialize (convert initial guesses to net if needed)
    rL_temp, rD_temp = get_robust_initial_rates(deltaL_j, deltaL_m, deltaD_j, deltaD_m, 
                                              kappaL_J, kappaD_J, params)
    
    # update guesses if values are seeded
    if init_rL is None:
        rL_old = rL_temp 
    else:
        rL_old = init_rL 
    if init_rD is None:
        rD_old = rD_temp
    else:
        rD_old = init_rD
    
    # store the initial price values for later regressions
    yL = rL_old[optimize_mask]
    yD = rD_old[optimize_mask]

    # Deposit outside good (constant)
    e0D =  _safe_exp_clip(alpha_D * (R - 1.0))

    # get the loan outside good matrix
    e0_BM = _compute_outside_e0_BM(theta_BM, alpha_L, c_L,
                                   vary_outside_L, equity_share)

    # Update first order conditions until convergence of the model
    for it in range(1, max_iter + 1):
        
        # build the values to helpfully pre-compute things        
        base_BM, kappa_BM = _loan_precompute_base(deltaL_m, theta_BM, alpha_L, c_L, equity_share)
        S_all_BM = _loan_build_S_all_BM(rL_old, deltaL_j, base_BM, kappa_BM, partL_JM)
        sum_expu_all_M, expu_old_all = _precompute_expu_all_deposits(
            rD_old, deltaD_j, deltaD_m, alpha_D, beta_branch, log_branches_JM, partD_JM
        )

        # guess loan prices
        rL_candidate, rD_candidate, errors, iterations = _update_rL_rD_bisect_parallel_with_holdings(
            
            # last step prices
            rL_old, rD_old, 
                                          
            # loans
            deltaL_j, deltaL_m, alpha_L, c_L, theta_BM, 
            sizeL_M, partL_JM, e0_BM, 
            base_BM, kappa_BM, S_all_BM, equity_share, 
            
            # deposits
            deltaD_j, deltaD_m, alpha_D, beta_branch, 
            log_branches_JM, sizeD_M, partD_JM, e0D, 
            sum_expu_all_M, expu_old_all,
            
            # inputs 
            kappaL_J, kappaD_J, R, E_J,
         
            # DD constraint pieces
            phi, lambda_liq, 
            
            # mergers and optimization control
            hc_offsets, hc_bank_idx, hc_optimize_mask, H, 
            
            # parallelization helpers
            h_mL_idx, h_mL_off, h_mD_idx, h_mD_off,
         
            # computation parameters 
            rL_min, rL_max,
            rD_min, rD_max)

        # Damped update of loan and deposit prices, for stability
        rL_new = 0.5 * rL_old + 0.5 * rL_candidate
        rD_new = 0.5 * rD_old + 0.5 * rD_candidate

        # Check convergence
        loan_gap = np.abs(rL_new - rL_old)
        dep_gap = np.abs(rD_new - rD_old)
        maxdiff = max(np.max(loan_gap), np.max(dep_gap))
        avgdiff = np.mean([np.mean(loan_gap), np.mean(dep_gap)])
        
        # update iteration
        print('processed iteration' + str(it) + ' | max error: ' + str(maxdiff) + ' | avg error: ' + str(avgdiff))
        
        rL_old, rD_old = rL_new, rD_new
        if maxdiff < tol:
            info = {"iterations": it, "converged": True, 
                    "max_dep_gap": np.max(dep_gap),
                    "max_loan_gap": np.max(loan_gap)}
            break
        else:
            info = {"iterations": max_iter, "converged": False,
                    "max_dep_gap": np.max(dep_gap),
                    "max_loan_gap": np.max(loan_gap)}


    ## ----- ##
    # Recompute shares at final prices
    S_L_MJ, Eths_L_MJ, Eths_oms_L_MJ, _Eth2s_oms_L_MJ = _loan_shares_and_moments(
        rL_new, deltaL_j, deltaL_m, alpha_L, c_L, theta_BM, partL_JM, e0_BM, equity_share
    )
    S_D_MJ, S_Doms_MJ = _deposit_shares_and_moments(
        rD_new, deltaD_j, deltaD_m, alpha_D, beta_branch, log_branches_JM, partD_JM, e0D
    )
    
    # BLAS-backed matrix–vector products
    L_j      = (sizeL_M @ S_L_MJ)            * (1.0 - equity_share)
    Eths_L_j = (sizeL_M @ Eths_L_MJ)         * (1.0 - equity_share)
    D_j      =  (sizeD_M @ S_D_MJ)
    I_j = E_J + D_j - L_j
    Loss_Rate_j = 1 - (Eths_L_j / L_j)
    ## ----- ##

    # Profits 
    loan_term = np.zeros(J, dtype=np.float64)
    for j in range(J): 
        loan_term[j] = rL_old[j] * Eths_L_j[j] - kappaL_J[j] * L_j[j]
    dep_term = np.zeros(J, dtype=np.float64)
    for j in range(J):
        dep_term[j] = - D_j[j] * (rD_old[j] + kappaD_J[j])
            
    # get profits
    profit_by_bank = loan_term + dep_term + R * I_j
    liq_cost = (phi / 2) * D_j * ((I_j / D_j) - lambda_liq) ** 2
    profit_by_bank_less_shadow_cost = profit_by_bank - liq_cost

    # calculate the deposit marginal costs
    B1 = sizeD_M @ S_D_MJ 
    B2 = sizeD_M @ S_Doms_MJ
    B1_over_B2 = B1 / B2
    I_D_j = I_j / D_j 
    G = I_D_j - lambda_liq
    Lambda_I = phi * G 
    Lambda_D = 0.5 * phi * (G ** 2 - 2.0 * G * I_D_j)
    DD_mc = Lambda_I + Lambda_D
    kappaD_J2 = ((R - rD_old)
            - (1 / alpha_D) * B1_over_B2
            - DD_mc)
     
    # --- Deposit FOC residuals at final prices, for BOTH kappas ---
    res_with_true = alpha_D * B2 * (R - rD_old - kappaD_J - DD_mc) - B1
    res_with_back = alpha_D * B2 * (R - rD_old - kappaD_J2 - DD_mc) - B1
    
    # (optional) max norms for quick printing/debug
    max_res_true = float(np.max(np.abs(res_with_true)))
    max_res_back = float(np.max(np.abs(res_with_back)))
    
    # get the depositor and borrower surplus
    avg_util_dep_market, avg_util_dep_market_optimized, avg_util_dep_market_aff = depositor_surplus(
        rD_old, deltaD_j, deltaD_m, alpha_D, beta_branch, 
        log_branches_JM, sizeD_M, partD_JM, e0D,
        optimize_mask, affected_markets)
    avg_util_loan_market, avg_util_loan_market_optimized, avg_util_loan_market_aff = borrower_surplus(
        rL_old, deltaL_j, deltaL_m, alpha_L, c_L, theta_BM, 
        sizeL_M, partL_JM, e0_BM, equity_share, 
        optimize_mask, affected_markets)
    
    # get the size of the markets affected by the optimization
    optimize_loan_market_size, optimize_dep_market_size = calc_optimized_market_size(
        params, partL_JM, partD_JM, sizeL_M, sizeD_M, optimize_mask 
    )

    # get the R-squared of predicted prices on actual prices
    XL = rL_old[optimize_mask]
    XL = sm.add_constant(XL)
    rsquaredL = sm.OLS(yL, XL).fit().rsquared
    XD = rD_old[optimize_mask]
    XD = sm.add_constant(XD)
    rsquaredD = sm.OLS(yD, XD).fit().rsquared

    return {
        "rL": rL_old,
        "rD": rD_old,
        "S_L_MJ": S_L_MJ,
        "E_theta_s_L_MJ": Eths_L_MJ,
        "S_D_MJ": S_D_MJ,
        "L_j": L_j,
        "Eths_L_j": Eths_L_j, 
        "Loss_Rate_j": Loss_Rate_j,
        "D_j": D_j,
        "E_j": E_J,
        "I_j": I_j,
        "kappaL_j": kappaL_J, 
        "kappaD_j": kappaD_J,
        "kappaD_j2": kappaD_J2,
        "profit_by_bank": profit_by_bank,
        "profit_by_bank_less_shadow_cost": profit_by_bank_less_shadow_cost,
        "info": info,
        "res_foc_dep_with_true": res_with_true,
        "res_foc_dep_with_back": res_with_back,
        "max_res_foc_dep_with_true": max_res_true,
        "max_res_foc_dep_with_back": max_res_back,
        "FOC_errors": errors,
        "final_iterations": iterations,
        "depositor_surplus": avg_util_dep_market, 
        "reoptimized_depositor_surplus": avg_util_dep_market_optimized,
        "affected_depositor_surplus": avg_util_dep_market_aff,
        "borrower_surplus": avg_util_loan_market,
        "reoptimized_borrower_surplus": avg_util_loan_market_optimized,
        "affected_borrower_surplus": avg_util_loan_market_aff,
        "reoptimized_loan_market_size": optimize_loan_market_size, 
        "reoptimized_dep_market_size": optimize_dep_market_size,
        "fit_weight": (rsquaredL + rsquaredD) / 2
    }


"""
create code that will simulate and solve a Nash Equilibrium model for a given
set of input parameters
"""

# this code will create a dictionary of the key model parameters
def create_nash_eqm_parameters(
        alpha_L = 29.751, alpha_D = 1806.0936, c_L = 1.17, r = 0.016433, beta_branch = 2.869, 
        lambda_liq = 0.2, phi = 0.01, equity_share = 0.17, m = 0.9361317, tau = 64.60046,
        beta_L_fass = 0.5, beta_L_comp = 0.5, beta_D_fass = 0.5, beta_D_comp = 0.5,
        kappaL0 = 0.005, kappaD0 = 0.005,
        rL_min = 1.0, rL_max = 2.0, rD_min = 1.0, rD_max = 1.5
    ):
    
    # get the \alpha and \beta parameters of the \beta distribution, given the 
    # mean and dispersion parameterization 
    a = m * tau 
    b = (1.0 - m) * tau
    
    # set the parameter values
    curr_params = {
        "alpha_L": alpha_L, 
        "alpha_D": alpha_D, 
        "c_L": c_L, 
        "r": r, 
        "beta_branch": beta_branch,
        "lambda_liq": lambda_liq, 
        "phi": phi, 
        "equity_share": equity_share,
        "m": m, 
        "tau": tau,
        "a": a, 
        "b": b,
        "beta_L_fass": beta_L_fass, 
        "beta_L_comp": beta_L_comp,
        "beta_D_fass": beta_D_fass, 
        "beta_D_comp": beta_D_comp,
        "kappaL0": kappaL0,
        "kappaD0": kappaD0,
        "rL_min": rL_min + r, 
        "rL_max": rL_max, 
        "rD_min": rD_min, 
        "rD_max": rD_max
    }
    
    return curr_params

def simulate_nash_eqm_params(
        nash_eqm_params, J = 6, M = 8, B = 200, seed = 42, 
        kappaL_mean = 0.0015, kappaD_mean = 0.002, 
        sizeL_scale = 50.0, sizeD_scale = 100.0,
        deltaL_j_mean = 5.5, deltaL_j_sd = 0.9, 
        deltaL_m_mean = 0.5, deltaL_m_sd = 0.5, 
        deltaD_j_mean = -1.5, deltaD_j_sd = 1.1, 
        deltaD_m_mean = -1.5, deltaD_m_sd = 2.2
    ):
    
    """Generate a toy instance and solve the joint equilibrium."""
    rng = np.random.default_rng(seed)

    # tastes, FEs
    deltaL_j = rng.normal(deltaL_j_mean, deltaL_j_sd, size=J)
    deltaL_m = rng.normal(deltaL_m_mean, deltaL_m_sd, size=M)
    deltaD_j = rng.normal(deltaD_j_mean, deltaD_j_sd, size=J)
    deltaD_m = rng.normal(deltaD_m_mean, deltaD_m_sd, size=M)

    # participation
    partL_JM = rng.random((J, M)) < 0.8
    partD_JM = rng.random((J, M)) < 0.8
    for m in range(M):
        if not partL_JM[:, m].any():
            partL_JM[rng.integers(J), m] = True
        if not partD_JM[:, m].any():
            partD_JM[rng.integers(J), m] = True
    for j in range(J):
        if not partL_JM[j, :].any():
            partL_JM[j, rng.integers(M)] = True
        if not partD_JM[j, :].any():
            partD_JM[j, rng.integers(M)] = True

    # borrower θ draws
    a = nash_eqm_params["a"]
    b = nash_eqm_params["b"]
    theta_BM = rng.beta(a, b, size=(B, M))

    # sizes
    sizeL_M = sizeL_scale * (1.1 + 0.4 * rng.random(M))
    sizeD_M = sizeD_scale * (1.5 + 0.6 * rng.random(M))

    # branches
    branches_JM = (1.0 + (rng.integers(1, 20, size=(J, M))).astype(np.float64))

    # costs, equity
    kappaL_J = kappaL_mean + 0.025 * rng.random(J)
    kappaD_J = kappaD_mean + 0.025 * rng.random(J)
    E_J = 1 / 4 * sizeL_scale * np.clip(0.05 + 0.10 * rng.random(J), 0.05, 0.1)
    
    return (deltaL_j, deltaL_m, deltaD_j, deltaD_m, partL_JM, partD_JM, theta_BM, 
            sizeL_M, sizeD_M, branches_JM, kappaL_J, kappaD_J, E_J)

# ------------------------------ toy simulator --------------------------------

def simulate_and_solve(
    J = 6, M = 8, B = 200, seed = 42, 
    kappaL_mean = 0.015, kappaD_mean = 0.01, 
    sizeL_scale = 75.0, sizeD_scale = 100.0,
    alpha_L = 23.028383767505165, alpha_D = 91.77452, c_L = 1.18888, r = 0.0240333333, beta_branch = 1.29207, 
    lambda_liq = 0.3912, phi = 0.00484, equity_share = 0.17, m = 0.94, tau = 64.0,
    beta_L_fass = 0.11, beta_L_comp = 0.12, beta_D_fass = 0.24, beta_D_comp = 0.0,
    kappaL0 = 0.005, kappaD0 = 0.005,
    rL_min = 1.0, rL_max = 2.0, rD_min = 1.0, rD_max = 1.5,
    max_iter = 500, tol = 1e-6, max_bisect = 30):
    
    # create the parameters
    nash_eqm_params = create_nash_eqm_parameters(
            alpha_L = alpha_L, alpha_D = alpha_D, c_L = c_L, r = r, 
            beta_branch = beta_branch, lambda_liq = lambda_liq, phi = phi, 
            equity_share = equity_share, m = m, tau = tau,
            beta_L_fass = beta_L_fass, beta_L_comp = beta_L_comp,
            beta_D_fass = beta_D_fass, beta_D_comp = beta_D_comp,
            kappaL0 = kappaL0, kappaD0 = kappaD0, 
            rL_min = rL_min, rL_max = rL_max, rD_min = rD_min, rD_max = rD_max
        )

    # create the model shocks
    (deltaL_j, deltaL_m, deltaD_j, deltaD_m, partL_JM, partD_JM, theta_BM, 
     sizeL_M, sizeD_M, branches_JM, r_fass_at, r_comp_at, E_J) = simulate_nash_eqm_params(
             nash_eqm_params, J = J, M = M, B = B, seed = seed, 
             kappaL_mean = kappaL_mean, kappaD_mean = kappaD_mean, 
             sizeL_scale = sizeL_scale, sizeD_scale = sizeD_scale
         )

    # create true marginal costs, which are the same for both loans and deposits
    # at the moment
    # rng = np.random.default_rng(seed + 10)
    J = r_fass_at.shape[0]
    kappaL_J = kappaL0 + beta_L_fass * r_fass_at + beta_L_comp * r_comp_at #+ rng.normal(0.0, 0.0025, size = J)
    kappaD_J = kappaD0 + beta_D_fass * r_fass_at + beta_D_comp * r_comp_at #+ rng.normal(0.0, 0.0025, size = J)
         
    # add other necessary parameters 
    nash_eqm_params["J"] = J
    nash_eqm_params["vary_outside_L"] = True
      
    # solve the equilibrium
    out = solve_joint_eqm(
        nash_eqm_params,
        deltaL_j, deltaL_m,
        deltaD_j, deltaD_m,
        branches_JM,
        partL_JM, partD_JM,
        theta_BM,
        sizeL_M, sizeD_M,
        kappaL_J, kappaD_J, E_J,
        rL_min = nash_eqm_params['rL_min'], rL_max = nash_eqm_params['rL_max'],
        rD_min = nash_eqm_params['rD_min'], rD_max = nash_eqm_params['rD_max'],
        max_iter = max_iter, tol = tol, max_bisect = max_bisect,
    )
    
    # store the parameters in the exporting results
    out["sim_params"] = nash_eqm_params 
    
    # store hte simulation data in what gets returned, as well
    out["sim_data"] = {
        "deltaL_j": deltaL_j, "deltaL_m": deltaL_m,
        "deltaD_j": deltaD_j, "deltaD_m": deltaD_m,
        "branches_JM": branches_JM,
        "partL_JM": partL_JM, "partD_JM": partD_JM,
        "theta_BM": theta_BM,
        "sizeL_M": sizeL_M, "sizeD_M": sizeD_M,
        "r_fass_at": r_fass_at, "r_comp_at": r_comp_at,
        "kappaL_J": kappaL_J, "kappaD_J": kappaD_J, "E_J": E_J,
    }
    return out

"""
For GMM estimation, create code that will take in various subcomponents of the 
loan first order conditions and return a vector that is loan first order
conditions. When evaluated at the true values of the parameter, it should return
0 (since it is the first order condition)
"""

# this function will build the multipliers on the different loan FOC terms, 
# given existing model data
def build_foc_iv_targets(params_base, rL_obs_J, 
                         deltaL_j, deltaL_m, theta_BM, partL_JM,
                         sizeL_M):
    """
    Precompute constants that enter the loan-pricing FOC moment evaluated at the
    observed/simulated equilibrium prices rL_obs_J.

    Returns a dict with arrays keyed by bank j:
      A1_J, A2_J, A3_J, DD_J, rL_obs_J, R
    """

    # unpack model pieces we need
    alpha_L = params_base["alpha_L"]
    c_L     = params_base["c_L"]
    e_share = params_base["equity_share"]   # "e" in the code
    one_minus_e = 1.0 - e_share
    vary_outside_L = params_base['vary_outside_L']
    
    # outside-option shares for the *loan* logit at the observed prices
    # (same object solver uses in its fixed-point / FOC code)
    e0_BM = _compute_outside_e0_BM(theta_BM, alpha_L, c_L, vary_outside_L, e_share)

    # moments of demand at the observed rL
    # returns (S_L_MJ, Eths_L_MJ, Eths_oms_L_MJ, Eth2s_oms_L_MJ)
    S_L_MJ, Eths_L_MJ, Eths_oms_L_MJ, Eth2s_oms_L_MJ = _loan_shares_and_moments(
        rL_obs_J, deltaL_j, deltaL_m, alpha_L, c_L, theta_BM, partL_JM, e0_BM, e_share
    )

    # Aggregate across markets with loan weights (exactly how the code aggregates)
    A1_J = one_minus_e * (sizeL_M @ Eths_L_MJ)
    A2_J = alpha_L * (one_minus_e**2) * (sizeL_M @ Eths_oms_L_MJ)  
    A3_J = alpha_L * (one_minus_e**2) * (sizeL_M @ Eth2s_oms_L_MJ)

    return {
        "A1_J": A1_J, "A2_J": A2_J, "A3_J": A3_J
    }

# this function will build the multipliers on the different deposit FOC terms,
# given existing model data
def build_foc_iv_targets_dep(params_base, 
                             rD_obs_J, deltaD_j, deltaD_m, branches_JM, partD_JM, 
                             sizeD_M):

    # extract key model parameters
    r = float(params_base['r'])
    R = 1 + r
    alpha_D = float(params_base['alpha_D'])
    beta_branch = float(params_base["beta_branch"])
    
    # create model objects
    e0D = np.exp(np.clip(alpha_D * (R - 1.0), -700.0, 700.0))
    log_branches_JM = np.log(np.maximum(branches_JM, 1.0))

    # get market shares and 1 - market share in each market
    S_D_MJ, S_Doms_MJ = _deposit_shares_and_moments(
        rD_obs_J, deltaD_j, deltaD_m, alpha_D, beta_branch, log_branches_JM, partD_JM, e0D
    )
    
    # calculate key numerator and denominator
    B1_J = sizeD_M @ S_D_MJ
    B2_J = sizeD_M @ S_Doms_MJ
    
    # return these key values 
    return {
        "B1_J": B1_J, "B2_J": B2_J    
    }
    
    
    