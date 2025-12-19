#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Joint loan–deposit Nash equilibrium with home-bias.

A) Per-submarket pricing
   - Banks choose rL[j,m] and rD[j,m] (gross rates) for each market m.

B) Heterogeneity on BOTH sides
   - Loans integrate over borrower types zL[b,m] with weights wL[b,m].
   - Deposits integrate over depositor types zD[d,m] with weights wD[d,m].
   - You can interpret zL/zD as “wealth / size / risk aversion” draws.

C) Bank-level liquidity constraint (balance-sheet coupling across markets)
   Default:
      E_j + (1-lambda_liq)*D_j  -  L_j  >= 0
   implemented via KKT multiplier mu_j >= 0.

D) Same loan pricing kernel structure as your original
   base_L[b,m]  = exp(deltaL_m[m] + zL[b,m]*alpha_L*c_L)
   kappa_L[b,m] = zL[b,m]*alpha_L*(1-e)
   term = base_L * exp(deltaL_j + home_bias) * exp(-kappa_L * rL[j,m])

E) Home bias
   - Implemented as an additive utility intercept that can depend on type:
       + home_jm * (gammaL0 + gammaL1*zL)
       + home_jm * (gammaD0 + gammaD1*zD)
   - Set gammas to 0 to turn off.

Notes on speed:
- The bank problem is solved by:
    (i) for a given mu_j, solve each market’s 1D FOC (loans and deposits)
    (ii) adjust mu_j by bisection until liquidity constraint holds (if binding)
  This avoids a  (2M+1)-dim Newton, but is equivalent for the hard constraint.

"""

import numpy as np

# Numba is strongly recommended for performance.
# This fallback lets the module run (slowly) in environments where numba
# cannot be imported.
try:
    from numba import njit, prange  # type: ignore
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        def _wrap(func):
            return func
        return _wrap

    prange = range

# =============================================================================
# 0) Numerics
# =============================================================================

@njit(cache=True)
def _safe_log(x):
    # log with floor to avoid -inf and NaNs
    if x <= 0.0 or (not np.isfinite(x)):
        return -1.0e300  # ~ -inf but finite for numba
    return np.log(x)

@njit(cache=True)
def _safe_exp_clip(u, max_val=700.0):
    # avoid overflow in exp; in float64 overflow happens around ~709
    if u > max_val:
        u = max_val
    elif u < -max_val:
        u = -max_val
    return np.exp(u)

# ---- Backward-compatible aliases (so you don't have to edit the whole file) ----
_safe_log = _safe_log
# =============================================================================
# 1) Loan kernels: base/kappa + outside option
# =============================================================================

@njit(cache=True, fastmath=True)
def _loan_precompute(deltaL_m, zL_BM, alpha_L, c_L, equity_share):
    """Return base_L[b,m], kappa_L[b,m], and e0_L[b,m]."""
    B, M = zL_BM.shape
    base = np.empty((B, M), dtype=np.float64)
    kappa = np.empty((B, M), dtype=np.float64)
    e0 = np.empty((B, M), dtype=np.float64)

    one_minus_e = 1.0 - equity_share
    for b in range(B):
        for m in range(M):
            z = zL_BM[b, m]
            base[b, m] = _safe_exp_clip(deltaL_m[m] + z * alpha_L * c_L)
            kappa[b, m] = z * alpha_L * one_minus_e
            # outside option: keep same structure you used before
            e0[b, m] = _safe_exp_clip(alpha_L * equity_share * c_L * z)
    return base, kappa, e0

# We can’t use a python-level helper in njit; implement home multiplier inline.

@njit(cache=True, fastmath=True)
def _loan_build_S_all_v2(base_L, kappa_L, zL_BM, rL_old_JM, deltaL_j, home_JM,
                        gammaL0, gammaL1, partL_JM):
    B, M = base_L.shape
    J = deltaL_j.size
    S_all = np.zeros((B, M), dtype=np.float64)

    wj = np.empty(J, dtype=np.float64)
    for j in range(J):
        wj[j] = _safe_exp_clip(deltaL_j[j])

    for b in range(B):
        for m in range(M):
            kb = kappa_L[b, m]
            z = zL_BM[b, m]
            acc = 0.0
            for j in range(J):
                if partL_JM[j, m]:
                    hb = 1.0 if home_JM[j, m] else 0.0
                    # additive intercept in utility -> multiplicative in exp()
                    hm = _safe_exp_clip(hb * (gammaL0 + gammaL1 * z))
                    acc += (wj[j] * hm) * _safe_exp_clip(-kb * rL_old_JM[j, m])
            S_all[b, m] = base_L[b, m] * acc
    return S_all

# =============================================================================
# 2) Deposit kernels: hetero slope + outside option per type
# =============================================================================

@njit(cache=True, fastmath=True)
def _deposit_precompute_slope(zD_DM, alpha_D, beta_D_slope):
    """slope_D[d,m] = alpha_D + beta_D_slope * zD[d,m]."""
    Dn, M = zD_DM.shape
    slope = np.empty((Dn, M), dtype=np.float64)
    for d in range(Dn):
        for m in range(M):
            slope[d, m] = alpha_D + beta_D_slope * zD_DM[d, m]
    return slope

@njit(cache=True, fastmath=True)
def _deposit_build_sum_expu(slope_D, zD_DM, rD_old_JM, deltaD_j, deltaD_m,
                           beta_branch, log_branches_JM, home_JM,
                           gammaD0, gammaD1, partD_JM,
                           R):
    """sum_expu[d,m] = sum_j exp(u_{j,d,m}) evaluated at rD_old_JM."""
    Dn, M = slope_D.shape
    J = deltaD_j.size
    sum_expu = np.zeros((Dn, M), dtype=np.float64)

    for d in range(Dn):
        for m in range(M):
            z = zD_DM[d, m]
            sl = slope_D[d, m]
            acc = 0.0
            for j in range(J):
                if partD_JM[j, m]:
                    hb = 1.0 if home_JM[j, m] else 0.0
                    hm = _safe_exp_clip(hb * (gammaD0 + gammaD1 * z))
                    u = deltaD_j[j] + deltaD_m[m] + beta_branch * log_branches_JM[j, m] + sl * (rD_old_JM[j, m] - 1.0)
                    acc += hm * _safe_exp_clip(u)
            sum_expu[d, m] = acc
    return sum_expu

@njit(cache=True, fastmath=True)
def _deposit_outside_e0D(slope_D, R):
    """Outside option per type: invest at R in the same utility scale."""
    Dn, M = slope_D.shape
    e0 = np.empty((Dn, M), dtype=np.float64)
    for d in range(Dn):
        for m in range(M):
            e0[d, m] = _safe_exp_clip(slope_D[d, m] * (R - 1.0))
    return e0

# =============================================================================
# 3) Per-bank-per-market demand + derivatives (heterogeneity integration)
# =============================================================================

@njit(cache=True, fastmath=True)
def _loan_L_dL_for_jm(j, m, rL_cand, rL_old_jm,
                     base_L, kappa_L, e0_L, S_all_L,
                     zL_BM, wL_BM,
                     deltaL_j, home_jm, gammaL0, gammaL1,
                     sizeL_m, equity_share):
    """Return (L_jm, dL_jm/drL) given others fixed at old prices."""
    B = base_L.shape[0]
    one_minus_e = 1.0 - equity_share
    invB = 1.0 / float(B)

    wj = _safe_exp_clip(deltaL_j[j])
    L = 0.0
    dL = 0.0
    hb = 1.0 if home_jm else 0.0

    for b in range(B):
        z = zL_BM[b, m]
        kb = kappa_L[b, m]
        bb = base_L[b, m]
        hm = _safe_exp_clip(hb * (gammaL0 + gammaL1 * z))

        # self terms
        old_term = bb * (wj * hm) * _safe_exp_clip(-kb * rL_old_jm)
        new_term = bb * (wj * hm) * _safe_exp_clip(-kb * rL_cand)

        denom = e0_L[b, m] + (S_all_L[b, m] - old_term + new_term)
        if denom < 1e-300:
            denom = 1e-300
        s = new_term / denom

        w = wL_BM[b, m]
        L += w * s
        dL += w * (s * (1.0 - s) * (-kb))

    L = sizeL_m * one_minus_e * (L * invB)
    dL = sizeL_m * one_minus_e * (dL * invB)
    return L, dL

@njit(cache=True, fastmath=True)
def _deposit_D_dD_for_jm(j, m, rD_cand, rD_old_jm,
                        slope_D, e0D_DM, sum_expu_D, zD_DM, wD_DM,
                        deltaD_j, deltaD_m, beta_branch, log_branches_JM,
                        home_jm, gammaD0, gammaD1,
                        sizeD_m):
    """Return (D_jm, dD_jm/drD) given others fixed at old prices."""
    Dn = slope_D.shape[0]
    invD = 1.0 / float(Dn)

    D = 0.0
    dD = 0.0
    hb = 1.0 if home_jm else 0.0

    for d in range(Dn):
        z = zD_DM[d, m]
        sl = slope_D[d, m]
        hm = _safe_exp_clip(hb * (gammaD0 + gammaD1 * z))

        # self exp utilities (note: hm multiplicative)
        u_old = deltaD_j[j] + deltaD_m[m] + beta_branch * log_branches_JM[j, m] + sl * (rD_old_jm - 1.0)
        u_new = deltaD_j[j] + deltaD_m[m] + beta_branch * log_branches_JM[j, m] + sl * (rD_cand - 1.0)
        exp_old = hm * _safe_exp_clip(u_old)
        exp_new = hm * _safe_exp_clip(u_new)

        denom = e0D_DM[d, m] + (sum_expu_D[d, m] - exp_old + exp_new)
        if denom < 1e-300:
            denom = 1e-300
        s = exp_new / denom

        w = wD_DM[d, m]
        D += w * s
        dD += w * (s * (1.0 - s) * sl)

    D = sizeD_m * (D * invD)
    dD = sizeD_m * (dD * invD)
    return D, dD

# =============================================================================
# 4) 1D market solvers given a bank-wide liquidity wedge (mu or soft x)
# =============================================================================

@njit(cache=True, fastmath=True)
def _solve_rL_market_fp(j, m, r_init, rL_old_jm,
                       base_L, kappa_L, e0_L, S_all_L, zL_BM, wL_BM,
                       deltaL_j, home_jm, gammaL0, gammaL1,
                       sizeL_m, equity_share,
                       R, kappaL_j, mu_j,
                       r_min, r_max,
                       max_it=60, tol=1e-10):
    """Solve loan FOC in market (j,m) with hard liquidity multiplier mu_j.

    FOC: L + dL * (r - (R + kappaL + mu)) = 0
         => r = (R+kappaL+mu) - L/dL
    """
    r = r_init
    cost = R + kappaL_j + mu_j

    for _ in range(max_it):
        L, dL = _loan_L_dL_for_jm(j, m, r, rL_old_jm,
                                 base_L, kappa_L, e0_L, S_all_L,
                                 zL_BM, wL_BM,
                                 deltaL_j, home_jm, gammaL0, gammaL1,
                                 sizeL_m, equity_share)
        if dL >= -1e-14:
            # demand slope wrong sign or tiny -> nudge up a bit
            r = min(r_max, max(r_min, r + 1e-3))
            continue
        r_tgt = cost - (L / dL)
        if r_tgt < r_min:
            r_tgt = r_min
        elif r_tgt > r_max:
            r_tgt = r_max

        diff = r_tgt - r
        r = r + 0.7 * diff
        if abs(diff) < tol:
            break

    # FOC residual at final r
    L, dL = _loan_L_dL_for_jm(j, m, r, rL_old_jm,
                             base_L, kappa_L, e0_L, S_all_L,
                             zL_BM, wL_BM,
                             deltaL_j, home_jm, gammaL0, gammaL1,
                             sizeL_m, equity_share)
    foc = L + dL * (r - cost)
    return r, L, foc

@njit(cache=True, fastmath=True)
def _solve_rD_market_fp(j, m, r_init, rD_old_jm,
                       slope_D, e0D_DM, sum_expu_D, zD_DM, wD_DM,
                       deltaD_j, deltaD_m, beta_branch, log_branches_JM,
                       home_jm, gammaD0, gammaD1,
                       sizeD_m,
                       R, kappaD_j, mu_j, lambda_liq,
                       r_min, r_max,
                       max_it=60, tol=1e-10):
    """Solve deposit FOC in market (j,m) with hard liquidity multiplier mu_j.

    Profit term in D gets + mu*(1-lambda) * D from constraint.

    Deposit FOC: dD * (R - r - kappaD + mu*(1-lambda)) - D = 0
                 => r = (R - kappaD + mu*(1-lambda)) - D/dD
    """
    r = r_init
    benefit = R - kappaD_j + mu_j * (1.0 - lambda_liq)

    for _ in range(max_it):
        D, dD = _deposit_D_dD_for_jm(j, m, r, rD_old_jm,
                                    slope_D, e0D_DM, sum_expu_D, zD_DM, wD_DM,
                                    deltaD_j, deltaD_m, beta_branch, log_branches_JM,
                                    home_jm, gammaD0, gammaD1,
                                    sizeD_m)
        if dD <= 1e-14:
            # tiny slope -> nudge slightly
            r = min(r_max, max(r_min, r - 1e-3))
            continue
        r_tgt = benefit - (D / dD)
        if r_tgt < r_min:
            r_tgt = r_min
        elif r_tgt > r_max:
            r_tgt = r_max

        diff = r_tgt - r
        r = r + 0.7 * diff
        if abs(diff) < tol:
            break

    D, dD = _deposit_D_dD_for_jm(j, m, r, rD_old_jm,
                                slope_D, e0D_DM, sum_expu_D, zD_DM, wD_DM,
                                deltaD_j, deltaD_m, beta_branch, log_branches_JM,
                                home_jm, gammaD0, gammaD1,
                                sizeD_m)
    foc = dD * (benefit - r) - D
    return r, D, foc

# =============================================================================
# 5) Bank best response: solve mu_j (hard) or x_j (soft), then market rates
# =============================================================================

@njit(cache=True, fastmath=True)
def _bank_best_response_hard(
    j,
    # old rates (others fixed)
    rL_old_JM, rD_old_JM,
    # participation
    partL_JM, partD_JM,
    # loan environment
    base_L, kappa_L, e0_L, S_all_L, zL_BM, wL_BM,
    # deposit environment
    slope_D, e0D_DM, sum_expu_D, zD_DM, wD_DM,
    # shifters
    deltaL_j, deltaD_j, deltaD_m,
    beta_branch, log_branches_JM,
    home_JM, gammaL0, gammaL1, gammaD0, gammaD1,
    sizeL_M, sizeD_M,
    # costs / balance sheet
    R, kappaL_j, kappaD_j, E_j, lambda_liq,
    # bounds
    rL_min, rL_max, rD_min, rD_max,
    # solver controls
    max_mu_iter=35, max_market_iter=60,
):
    """Hard-liquidity best response for bank j.

    Finds mu >= 0 such that slack = E + (1-lambda)*D - L >= 0.
    If slack(mu=0) >= 0, uses mu=0.
    Otherwise, bisection on mu until slack ~= 0.

    Returns:
      rL_row(M), rD_row(M), L_row(M), D_row(M),
      mu, slack, max_abs_foc
    """
    M = sizeL_M.size

    # working arrays
    rL_row = np.empty(M, dtype=np.float64)
    rD_row = np.empty(M, dtype=np.float64)
    L_row = np.zeros(M, dtype=np.float64)
    D_row = np.zeros(M, dtype=np.float64)

    # initialise at old
    for m in range(M):
        rL_row[m] = rL_old_JM[j, m]
        rD_row[m] = rD_old_JM[j, m]

    # helper: given mu, solve all markets and compute slack
    def solve_given_mu(mu, rL_start, rD_start):
        # Numba cannot use nested def; we inline using a block below.
        return 0.0

    # --- compute slack at mu=0 ---
    mu0 = 0.0
    max_foc0 = 0.0
    Ltot0 = 0.0
    Dtot0 = 0.0

    for m in range(M):
        # loans
        if partL_JM[j, m]:
            r, Lm, foc = _solve_rL_market_fp(
                j, m,
                rL_row[m], rL_old_JM[j, m],
                base_L, kappa_L, e0_L, S_all_L, zL_BM, wL_BM,
                deltaL_j, home_JM[j, m], gammaL0, gammaL1,
                sizeL_M[m], 1.0 - (1.0 - 1.0),  # dummy, overwritten below
                R, kappaL_j, mu0,
                rL_min, rL_max,
                max_it=max_market_iter
            )
            # The equity_share is needed; but we only have it outside.
            # We pass it correctly by writing a wrapper below (see _bank_best_response_hard_v2).
            rL_row[m] = r
            L_row[m] = Lm
            if abs(foc) > max_foc0:
                max_foc0 = abs(foc)
            Ltot0 += Lm
        else:
            L_row[m] = 0.0

        # deposits
        if partD_JM[j, m]:
            r, Dm, foc = _solve_rD_market_fp(
                j, m,
                rD_row[m], rD_old_JM[j, m],
                slope_D, e0D_DM, sum_expu_D, zD_DM, wD_DM,
                deltaD_j, deltaD_m, beta_branch, log_branches_JM,
                home_JM[j, m], gammaD0, gammaD1,
                sizeD_M[m],
                R, kappaD_j, mu0, lambda_liq,
                rD_min, rD_max,
                max_it=max_market_iter
            )
            rD_row[m] = r
            D_row[m] = Dm
            if abs(foc) > max_foc0:
                max_foc0 = abs(foc)
            Dtot0 += Dm
        else:
            D_row[m] = 0.0

    # slack = E + (1-lam)*D - L
    slack0 = E_j + (1.0 - lambda_liq) * Dtot0 - Ltot0
    if slack0 >= 0.0:
        return rL_row, rD_row, L_row, D_row, 0.0, slack0, max_foc0

    # --- find an upper bound for mu such that slack >= 0 ---
    mu_lo = 0.0
    mu_hi = 1.0

    # restore starts to old for stability
    for m in range(M):
        rL_row[m] = rL_old_JM[j, m]
        rD_row[m] = rD_old_JM[j, m]

    for _ in range(20):
        # compute slack at mu_hi
        Ltot = 0.0
        Dtot = 0.0
        max_foc = 0.0
        for m in range(M):
            if partL_JM[j, m]:
                r, Lm, foc = _solve_rL_market_fp(
                    j, m,
                    rL_row[m], rL_old_JM[j, m],
                    base_L, kappa_L, e0_L, S_all_L, zL_BM, wL_BM,
                    deltaL_j, home_JM[j, m], gammaL0, gammaL1,
                    sizeL_M[m], 0.0,  # equity_share to be supplied in v2 wrapper
                    R, kappaL_j, mu_hi,
                    rL_min, rL_max,
                    max_it=max_market_iter
                )
                rL_row[m] = r
                L_row[m] = Lm
                if abs(foc) > max_foc:
                    max_foc = abs(foc)
                Ltot += Lm
            else:
                L_row[m] = 0.0

            if partD_JM[j, m]:
                r, Dm, foc = _solve_rD_market_fp(
                    j, m,
                    rD_row[m], rD_old_JM[j, m],
                    slope_D, e0D_DM, sum_expu_D, zD_DM, wD_DM,
                    deltaD_j, deltaD_m, beta_branch, log_branches_JM,
                    home_JM[j, m], gammaD0, gammaD1,
                    sizeD_M[m],
                    R, kappaD_j, mu_hi, lambda_liq,
                    rD_min, rD_max,
                    max_it=max_market_iter
                )
                rD_row[m] = r
                D_row[m] = Dm
                if abs(foc) > max_foc:
                    max_foc = abs(foc)
                Dtot += Dm
            else:
                D_row[m] = 0.0

        slack_hi = E_j + (1.0 - lambda_liq) * Dtot - Ltot
        if slack_hi >= 0.0:
            break
        mu_hi *= 2.0

    # --- bisection ---
    mu = mu_hi
    slack = 0.0
    max_foc = 0.0

    # keep last computed at mu_hi as starting point
    rL_start = rL_row.copy()
    rD_start = rD_row.copy()

    for _ in range(max_mu_iter):
        mu_mid = 0.5 * (mu_lo + mu_hi)

        # start from last solution (warm start)
        for m in range(M):
            rL_row[m] = rL_start[m]
            rD_row[m] = rD_start[m]

        Ltot = 0.0
        Dtot = 0.0
        max_foc = 0.0
        for m in range(M):
            if partL_JM[j, m]:
                r, Lm, foc = _solve_rL_market_fp(
                    j, m,
                    rL_row[m], rL_old_JM[j, m],
                    base_L, kappa_L, e0_L, S_all_L, zL_BM, wL_BM,
                    deltaL_j, home_JM[j, m], gammaL0, gammaL1,
                    sizeL_M[m], 0.0,  # equity_share injected in wrapper
                    R, kappaL_j, mu_mid,
                    rL_min, rL_max,
                    max_it=max_market_iter
                )
                rL_row[m] = r
                L_row[m] = Lm
                if abs(foc) > max_foc:
                    max_foc = abs(foc)
                Ltot += Lm
            else:
                L_row[m] = 0.0

            if partD_JM[j, m]:
                r, Dm, foc = _solve_rD_market_fp(
                    j, m,
                    rD_row[m], rD_old_JM[j, m],
                    slope_D, e0D_DM, sum_expu_D, zD_DM, wD_DM,
                    deltaD_j, deltaD_m, beta_branch, log_branches_JM,
                    home_JM[j, m], gammaD0, gammaD1,
                    sizeD_M[m],
                    R, kappaD_j, mu_mid, lambda_liq,
                    rD_min, rD_max,
                    max_it=max_market_iter
                )
                rD_row[m] = r
                D_row[m] = Dm
                if abs(foc) > max_foc:
                    max_foc = abs(foc)
                Dtot += Dm
            else:
                D_row[m] = 0.0

        slack_mid = E_j + (1.0 - lambda_liq) * Dtot - Ltot

        # update bracket
        if slack_mid >= 0.0:
            mu_hi = mu_mid
            mu = mu_mid
            slack = slack_mid
            # warm start update
            for m in range(M):
                rL_start[m] = rL_row[m]
                rD_start[m] = rD_row[m]
        else:
            mu_lo = mu_mid

        # stopping on slack
        if abs(slack_mid) < 1e-10:
            mu = mu_mid
            slack = slack_mid
            break

    return rL_start, rD_start, L_row, D_row, mu, slack, max_foc

# The function above has a placeholder for equity_share inside _solve_rL_market_fp.
# To keep everything njit-friendly without rewriting a huge amount of code, we
# provide a second version that correctly passes equity_share.

@njit(cache=True, fastmath=True)
def _bank_best_response_hard_v2(
    j,
    rL_old_JM, rD_old_JM,
    partL_JM, partD_JM,
    base_L, kappa_L, e0_L, S_all_L, zL_BM, wL_BM,
    slope_D, e0D_DM, sum_expu_D, zD_DM, wD_DM,
    deltaL_j, deltaD_j, deltaD_m,
    beta_branch, log_branches_JM,
    home_JM, gammaL0, gammaL1, gammaD0, gammaD1,
    sizeL_M, sizeD_M,
    R, kappaL_j, kappaD_j, E_j, lambda_liq,
    equity_share,
    rL_min, rL_max, rD_min, rD_max,
    max_mu_iter=35, max_market_iter=60,
):
    M = sizeL_M.size

    rL_row = np.empty(M, dtype=np.float64)
    rD_row = np.empty(M, dtype=np.float64)
    L_row = np.zeros(M, dtype=np.float64)
    D_row = np.zeros(M, dtype=np.float64)

    for m in range(M):
        rL_row[m] = rL_old_JM[j, m]
        rD_row[m] = rD_old_JM[j, m]

    # mu=0
    mu0 = 0.0
    max_foc0 = 0.0
    Ltot0 = 0.0
    Dtot0 = 0.0

    for m in range(M):
        if partL_JM[j, m]:
            r, Lm, foc = _solve_rL_market_fp(
                j, m,
                rL_row[m], rL_old_JM[j, m],
                base_L, kappa_L, e0_L, S_all_L, zL_BM, wL_BM,
                deltaL_j, home_JM[j, m], gammaL0, gammaL1,
                sizeL_M[m], equity_share,
                R, kappaL_j, mu0,
                rL_min, rL_max,
                max_it=max_market_iter
            )
            rL_row[m] = r
            L_row[m] = Lm
            if abs(foc) > max_foc0:
                max_foc0 = abs(foc)
            Ltot0 += Lm
        else:
            L_row[m] = 0.0

        if partD_JM[j, m]:
            r, Dm, foc = _solve_rD_market_fp(
                j, m,
                rD_row[m], rD_old_JM[j, m],
                slope_D, e0D_DM, sum_expu_D, zD_DM, wD_DM,
                deltaD_j, deltaD_m, beta_branch, log_branches_JM,
                home_JM[j, m], gammaD0, gammaD1,
                sizeD_M[m],
                R, kappaD_j, mu0, lambda_liq,
                rD_min, rD_max,
                max_it=max_market_iter
            )
            rD_row[m] = r
            D_row[m] = Dm
            if abs(foc) > max_foc0:
                max_foc0 = abs(foc)
            Dtot0 += Dm
        else:
            D_row[m] = 0.0

    slack0 = E_j + (1.0 - lambda_liq) * Dtot0 - Ltot0
    if slack0 >= 0.0:
        return rL_row, rD_row, L_row, D_row, 0.0, slack0, max_foc0

    # bracket mu
    mu_lo = 0.0
    mu_hi = 1.0

    for m in range(M):
        rL_row[m] = rL_old_JM[j, m]
        rD_row[m] = rD_old_JM[j, m]

    slack_hi = -1.0
    for _ in range(25):
        Ltot = 0.0
        Dtot = 0.0
        for m in range(M):
            if partL_JM[j, m]:
                r, Lm, _ = _solve_rL_market_fp(
                    j, m,
                    rL_row[m], rL_old_JM[j, m],
                    base_L, kappa_L, e0_L, S_all_L, zL_BM, wL_BM,
                    deltaL_j, home_JM[j, m], gammaL0, gammaL1,
                    sizeL_M[m], equity_share,
                    R, kappaL_j, mu_hi,
                    rL_min, rL_max,
                    max_it=max_market_iter
                )
                rL_row[m] = r
                L_row[m] = Lm
                Ltot += Lm
            else:
                L_row[m] = 0.0

            if partD_JM[j, m]:
                r, Dm, _ = _solve_rD_market_fp(
                    j, m,
                    rD_row[m], rD_old_JM[j, m],
                    slope_D, e0D_DM, sum_expu_D, zD_DM, wD_DM,
                    deltaD_j, deltaD_m, beta_branch, log_branches_JM,
                    home_JM[j, m], gammaD0, gammaD1,
                    sizeD_M[m],
                    R, kappaD_j, mu_hi, lambda_liq,
                    rD_min, rD_max,
                    max_it=max_market_iter
                )
                rD_row[m] = r
                D_row[m] = Dm
                Dtot += Dm
            else:
                D_row[m] = 0.0

        slack_hi = E_j + (1.0 - lambda_liq) * Dtot - Ltot
        if slack_hi >= 0.0:
            break
        mu_hi *= 2.0

    # bisection
    rL_best = rL_row.copy()
    rD_best = rD_row.copy()
    mu_best = mu_hi
    slack_best = slack_hi
    max_foc_best = 1e100

    for _ in range(max_mu_iter):
        mu_mid = 0.5 * (mu_lo + mu_hi)

        # warm start from best
        for m in range(M):
            rL_row[m] = rL_best[m]
            rD_row[m] = rD_best[m]

        Ltot = 0.0
        Dtot = 0.0
        max_foc = 0.0
        for m in range(M):
            if partL_JM[j, m]:
                r, Lm, foc = _solve_rL_market_fp(
                    j, m,
                    rL_row[m], rL_old_JM[j, m],
                    base_L, kappa_L, e0_L, S_all_L, zL_BM, wL_BM,
                    deltaL_j, home_JM[j, m], gammaL0, gammaL1,
                    sizeL_M[m], equity_share,
                    R, kappaL_j, mu_mid,
                    rL_min, rL_max,
                    max_it=max_market_iter
                )
                rL_row[m] = r
                L_row[m] = Lm
                if abs(foc) > max_foc:
                    max_foc = abs(foc)
                Ltot += Lm
            else:
                L_row[m] = 0.0

            if partD_JM[j, m]:
                r, Dm, foc = _solve_rD_market_fp(
                    j, m,
                    rD_row[m], rD_old_JM[j, m],
                    slope_D, e0D_DM, sum_expu_D, zD_DM, wD_DM,
                    deltaD_j, deltaD_m, beta_branch, log_branches_JM,
                    home_JM[j, m], gammaD0, gammaD1,
                    sizeD_M[m],
                    R, kappaD_j, mu_mid, lambda_liq,
                    rD_min, rD_max,
                    max_it=max_market_iter
                )
                rD_row[m] = r
                D_row[m] = Dm
                if abs(foc) > max_foc:
                    max_foc = abs(foc)
                Dtot += Dm
            else:
                D_row[m] = 0.0

        slack_mid = E_j + (1.0 - lambda_liq) * Dtot - Ltot

        if slack_mid >= 0.0:
            mu_hi = mu_mid
            mu_best = mu_mid
            slack_best = slack_mid
            max_foc_best = max_foc
            for m in range(M):
                rL_best[m] = rL_row[m]
                rD_best[m] = rD_row[m]
        else:
            mu_lo = mu_mid

        if abs(slack_mid) < 1e-10:
            break

    # recompute L_row/D_row at best rates for outputs
    for m in range(M):
        if partL_JM[j, m]:
            Lm, _ = _loan_L_dL_for_jm(j, m, rL_best[m], rL_old_JM[j, m],
                                     base_L, kappa_L, e0_L, S_all_L,
                                     zL_BM, wL_BM,
                                     deltaL_j, home_JM[j, m], gammaL0, gammaL1,
                                     sizeL_M[m], equity_share)
            L_row[m] = Lm
        else:
            L_row[m] = 0.0

        if partD_JM[j, m]:
            Dm, _ = _deposit_D_dD_for_jm(j, m, rD_best[m], rD_old_JM[j, m],
                                        slope_D, e0D_DM, sum_expu_D, zD_DM, wD_DM,
                                        deltaD_j, deltaD_m, beta_branch, log_branches_JM,
                                        home_JM[j, m], gammaD0, gammaD1,
                                        sizeD_M[m])
            D_row[m] = Dm
        else:
            D_row[m] = 0.0

    return rL_best, rD_best, L_row, D_row, mu_best, slack_best, max_foc_best

# =============================================================================
# 6) Parallel best-response update across banks
# =============================================================================

@njit(cache=True, parallel=True, fastmath=True)
def _update_all_banks_hard(
    rL_old_JM, rD_old_JM,
    partL_JM, partD_JM,
    base_L, kappa_L, e0_L, S_all_L, zL_BM, wL_BM,
    slope_D, e0D_DM, sum_expu_D, zD_DM, wD_DM,
    deltaL_j, deltaD_j, deltaD_m,
    beta_branch, log_branches_JM,
    home_JM, gammaL0, gammaL1, gammaD0, gammaD1,
    sizeL_M, sizeD_M,
    R, kappaL_J, kappaD_J, E_J, lambda_liq,
    equity_share,
    rL_min, rL_max, rD_min, rD_max,
):
    J, M = rL_old_JM.shape
    rL_br = np.empty((J, M), dtype=np.float64)
    rD_br = np.empty((J, M), dtype=np.float64)
    mu_J = np.empty(J, dtype=np.float64)
    slack_J = np.empty(J, dtype=np.float64)
    foc_J = np.empty(J, dtype=np.float64)
    L_JM = np.zeros((J, M), dtype=np.float64)
    D_JM = np.zeros((J, M), dtype=np.float64)

    for j in prange(J):
        rL_row, rD_row, L_row, D_row, mu, slack, max_foc = _bank_best_response_hard_v2(
            j,
            rL_old_JM, rD_old_JM,
            partL_JM, partD_JM,
            base_L, kappa_L, e0_L, S_all_L, zL_BM, wL_BM,
            slope_D, e0D_DM, sum_expu_D, zD_DM, wD_DM,
            deltaL_j, deltaD_j, deltaD_m,
            beta_branch, log_branches_JM,
            home_JM, gammaL0, gammaL1, gammaD0, gammaD1,
            sizeL_M, sizeD_M,
            R, kappaL_J[j], kappaD_J[j], E_J[j], lambda_liq,
            equity_share,
            rL_min, rL_max, rD_min, rD_max,
            35, 60
        )
        for m in range(M):
            rL_br[j, m] = rL_row[m]
            rD_br[j, m] = rD_row[m]
            L_JM[j, m] = L_row[m]
            D_JM[j, m] = D_row[m]
        mu_J[j] = mu
        slack_J[j] = slack
        foc_J[j] = max_foc

    return rL_br, rD_br, L_JM, D_JM, mu_J, slack_J, foc_J

# =============================================================================
# 7) Shares and welfare (post-processing)
# =============================================================================

@njit(cache=True, fastmath=True)
def _loan_shares_MJ(rL_JM, base_L, kappa_L, e0_L, deltaL_j, home_JM, gammaL0, gammaL1,
                   zL_BM, partL_JM):
    J, M = rL_JM.shape
    B = base_L.shape[0]

    # build S_all at equilibrium rates
    S_all = _loan_build_S_all_v2(base_L, kappa_L, zL_BM, rL_JM, deltaL_j, home_JM, gammaL0, gammaL1, partL_JM)

    wj = np.empty(J, dtype=np.float64)
    for j in range(J):
        wj[j] = _safe_exp_clip(deltaL_j[j])

    invB = 1.0 / float(B)
    S_L_MJ = np.zeros((M, J), dtype=np.float64)

    for m in range(M):
        for j in range(J):
            if not partL_JM[j, m]:
                continue
            r = rL_JM[j, m]
            acc = 0.0
            for b in range(B):
                z = zL_BM[b, m]
                hb = 1.0 if home_JM[j, m] else 0.0
                hm = _safe_exp_clip(hb * (gammaL0 + gammaL1 * z))
                num = base_L[b, m] * (wj[j] * hm) * _safe_exp_clip(-kappa_L[b, m] * r)
                den = e0_L[b, m] + S_all[b, m]
                if den < 1e-300:
                    den = 1e-300
                acc += num / den
            S_L_MJ[m, j] = acc * invB

    return S_L_MJ

@njit(cache=True, fastmath=True)
def _deposit_shares_MJ(rD_JM, slope_D, e0D_DM, deltaD_j, deltaD_m,
                      beta_branch, log_branches_JM, home_JM, gammaD0, gammaD1,
                      zD_DM, partD_JM):
    J, M = rD_JM.shape
    Dn = slope_D.shape[0]

    sum_expu = _deposit_build_sum_expu(slope_D, zD_DM, rD_JM, deltaD_j, deltaD_m,
                                      beta_branch, log_branches_JM, home_JM,
                                      gammaD0, gammaD1, partD_JM,
                                      R=1.0)  # R unused inside

    S_D_MJ = np.zeros((M, J), dtype=np.float64)
    invD = 1.0 / float(Dn)

    for m in range(M):
        for j in range(J):
            if not partD_JM[j, m]:
                continue
            acc = 0.0
            for d in range(Dn):
                z = zD_DM[d, m]
                hb = 1.0 if home_JM[j, m] else 0.0
                hm = _safe_exp_clip(hb * (gammaD0 + gammaD1 * z))
                sl = slope_D[d, m]
                u = deltaD_j[j] + deltaD_m[m] + beta_branch * log_branches_JM[j, m] + sl * (rD_JM[j, m] - 1.0)
                num = hm * _safe_exp_clip(u)
                den = e0D_DM[d, m] + sum_expu[d, m]
                if den < 1e-300:
                    den = 1e-300
                acc += num / den
            S_D_MJ[m, j] = acc * invD

    return S_D_MJ

@njit(cache=True, fastmath=True)
def _welfare(rL_JM, rD_JM,
            base_L, kappa_L, e0_L, deltaL_j, home_JM, gammaL0, gammaL1, zL_BM,
            slope_D, e0D_DM, deltaD_j, deltaD_m, beta_branch, log_branches_JM, gammaD0, gammaD1, zD_DM,
            partL_JM, partD_JM,
            sizeL_M, sizeD_M,
            alpha_L, alpha_D):
    """Simple aggregate surplus measures (same spirit as your earlier code)."""
    J, M = rL_JM.shape
    B = base_L.shape[0]
    Dn = slope_D.shape[0]

    # Borrower surplus: average logsum over loan types
    loan = 0.0
    wj = np.empty(J, dtype=np.float64)
    for j in range(J):
        wj[j] = _safe_exp_clip(deltaL_j[j])

    for m in range(M):
        acc_b = 0.0
        for b in range(B):
            den = e0_L[b, m]
            z = zL_BM[b, m]
            kb = kappa_L[b, m]
            bb = base_L[b, m]
            for j in range(J):
                if partL_JM[j, m]:
                    hb = 1.0 if home_JM[j, m] else 0.0
                    hm = _safe_exp_clip(hb * (gammaL0 + gammaL1 * z))
                    den += bb * (wj[j] * hm) * _safe_exp_clip(-kb * rL_JM[j, m])
            acc_b += _safe_log(den)
        loan += sizeL_M[m] * (acc_b / float(B))

    # Depositor surplus: average logsum over deposit types, divided by alpha_D (approx)
    dep = 0.0
    for m in range(M):
        acc_d = 0.0
        for d in range(Dn):
            den = e0D_DM[d, m]
            z = zD_DM[d, m]
            sl = slope_D[d, m]
            for j in range(J):
                if partD_JM[j, m]:
                    hb = 1.0 if home_JM[j, m] else 0.0
                    hm = _safe_exp_clip(hb * (gammaD0 + gammaD1 * z))
                    u = deltaD_j[j] + deltaD_m[m] + beta_branch * log_branches_JM[j, m] + sl * (rD_JM[j, m] - 1.0)
                    den += hm * _safe_exp_clip(u)
            acc_d += _safe_log(den)
        dep += sizeD_M[m] * (acc_d / float(Dn))

    # scale depositor surplus back by a representative alpha_D (rough)
    dep = dep / alpha_D

    return loan, dep

# =============================================================================
# 8) Public solver
# =============================================================================

def solve_joint_eqm(
    params,
    deltaL_j, deltaL_m,
    deltaD_j, deltaD_m,
    branches_JM,
    partL_JM, partD_JM,
    home_JM,
    zL_BM, wL_BM,
    zD_DM, wD_DM,
    sizeL_M, sizeD_M,
    kappaL_J, kappaD_J, E_J,
    # bounds
    rL_min=1.0, rL_max=3.0,
    rD_min=0.5, rD_max=2.0,
    # iteration
    max_iter=400, tol=1e-6, tol_foc=1e-6, damp_fp=0.5,
    init_rL=None, init_rD=None,
):
    """Outer fixed-point: iterate bank best responses until convergence."""

    # unpack
    alpha_L = float(params["alpha_L"])
    alpha_D = float(params["alpha_D"])
    beta_branch = float(params.get("beta_branch", 0.0))
    c_L = float(params.get("c_L", 1.0))
    r = float(params.get("r", 0.0))
    R = 1.0 + r
    equity_share = float(params.get("equity_share", 0.0))
    lambda_liq = float(params.get("lambda_liq", 0.0))

    # home bias parameters
    gammaL0 = float(params.get("gammaL0", 0.0))
    gammaL1 = float(params.get("gammaL1", 0.0))
    gammaD0 = float(params.get("gammaD0", 0.0))
    gammaD1 = float(params.get("gammaD1", 0.0))

    beta_D_slope = float(params.get("beta_D_slope", 0.0))

    deltaL_j = np.asarray(deltaL_j, dtype=np.float64)
    deltaL_m = np.asarray(deltaL_m, dtype=np.float64)
    deltaD_j = np.asarray(deltaD_j, dtype=np.float64)
    deltaD_m = np.asarray(deltaD_m, dtype=np.float64)

    partL_JM = np.asarray(partL_JM, dtype=np.bool_)
    partD_JM = np.asarray(partD_JM, dtype=np.bool_)
    home_JM = np.asarray(home_JM, dtype=np.bool_)

    branches_JM = np.asarray(branches_JM, dtype=np.float64)
    log_branches_JM = np.log(np.maximum(branches_JM, 1.0))

    zL_BM = np.asarray(zL_BM, dtype=np.float64)
    wL_BM = np.asarray(wL_BM, dtype=np.float64)
    zD_DM = np.asarray(zD_DM, dtype=np.float64)
    wD_DM = np.asarray(wD_DM, dtype=np.float64)

    sizeL_M = np.asarray(sizeL_M, dtype=np.float64)
    sizeD_M = np.asarray(sizeD_M, dtype=np.float64)
    kappaL_J = np.asarray(kappaL_J, dtype=np.float64)
    kappaD_J = np.asarray(kappaD_J, dtype=np.float64)
    E_J = np.asarray(E_J, dtype=np.float64)

    J, M = partL_JM.shape

    # loan precomputes
    base_L, kappa_L, e0_L = _loan_precompute(deltaL_m, zL_BM, alpha_L, c_L, equity_share)

    # deposit precomputes
    slope_D = _deposit_precompute_slope(zD_DM, alpha_D, beta_D_slope)
    e0D_DM = _deposit_outside_e0D(slope_D, R)

    # initial rates
    if init_rL is None:
        rL_old = np.full((J, M), R + 0.02, dtype=np.float64)
    else:
        rL_old = np.asarray(init_rL, dtype=np.float64).copy()
    if init_rD is None:
        rD_old = np.full((J, M), max(rD_min, R - 0.01), dtype=np.float64)
    else:
        rD_old = np.asarray(init_rD, dtype=np.float64).copy()

    # enforce bounds & participation
    rL_old = np.clip(rL_old, rL_min, rL_max)
    rD_old = np.clip(rD_old, rD_min, rD_max)

    print(f"Starting per-market solver: J={J}, M={M}, B_L={zL_BM.shape[0]}, B_D={zD_DM.shape[0]}")

    info = {"converged": False, "iterations": max_iter, "gap": np.nan, "max_foc": np.nan}

    for it in range(1, max_iter + 1):
        # environments at old prices
        S_all_L = _loan_build_S_all_v2(base_L, kappa_L, zL_BM, rL_old, deltaL_j, home_JM, gammaL0, gammaL1, partL_JM)
        sum_expu_D = _deposit_build_sum_expu(slope_D, zD_DM, rD_old, deltaD_j, deltaD_m,
                                            beta_branch, log_branches_JM, home_JM,
                                            gammaD0, gammaD1, partD_JM,
                                            R)

        # best responses in parallel
        rL_br, rD_br, L_JM, D_JM, mu_J, slack_J, foc_J = _update_all_banks_hard(
            rL_old, rD_old,
            partL_JM, partD_JM,
            base_L, kappa_L, e0_L, S_all_L, zL_BM, wL_BM,
            slope_D, e0D_DM, sum_expu_D, zD_DM, wD_DM,
            deltaL_j, deltaD_j, deltaD_m,
            beta_branch, log_branches_JM,
            home_JM, gammaL0, gammaL1, gammaD0, gammaD1,
            sizeL_M, sizeD_M,
            R, kappaL_J, kappaD_J, E_J, lambda_liq,
            equity_share,
            rL_min, rL_max, rD_min, rD_max,
        )

        # damped update
        rL_new = (1.0 - damp_fp) * rL_old + damp_fp * rL_br
        rD_new = (1.0 - damp_fp) * rD_old + damp_fp * rD_br

        # keep non-participation fixed (or set to NaN)
        for j in range(J):
            for m in range(M):
                if not partL_JM[j, m]:
                    rL_new[j, m] = rL_old[j, m]
                if not partD_JM[j, m]:
                    rD_new[j, m] = rD_old[j, m]

        gap = max(float(np.max(np.abs(rL_new - rL_old))), float(np.max(np.abs(rD_new - rD_old))))
        max_foc = float(np.max(foc_J))

        if it == 1 or it % 10 == 0:
            print(f"Iter {it:03d}: gap={gap:.3e} | max bank FOC={max_foc:.3e}")

        rL_old, rD_old = rL_new, rD_new

        if gap < tol and max_foc < tol_foc:
            info = {"converged": True, "iterations": it, "gap": gap, "max_foc": max_foc}
            break

    # post-processing
    # volumes and liquidity ratio
    L_j = np.sum(L_JM, axis=1)
    D_j = np.sum(D_JM, axis=1)
    I_j = E_J + D_j - L_j
    with np.errstate(divide="ignore", invalid="ignore"):
        I_over_D = I_j / np.maximum(D_j, 1e-12)

    # shares
    S_L_MJ = _loan_shares_MJ(rL_old, base_L, kappa_L, e0_L, deltaL_j, home_JM, gammaL0, gammaL1, zL_BM, partL_JM)
    S_D_MJ = _deposit_shares_MJ(rD_old, slope_D, e0D_DM, deltaD_j, deltaD_m,
                               beta_branch, log_branches_JM, home_JM, gammaD0, gammaD1,
                               zD_DM, partD_JM)

    # profits (consistent with bank accounting)
    # Π = Σ (rL - R - kL) L + Σ (R - rD - kD) D + R*E
    # (hard constraint multiplier already handled in choices; not subtracted from Π)
    profit_j = np.zeros(J, dtype=np.float64)
    for j in range(J):
        pl = 0.0
        pd = 0.0
        for m in range(M):
            if partL_JM[j, m]:
                pl += (rL_old[j, m] - (R + kappaL_J[j])) * L_JM[j, m]
            if partD_JM[j, m]:
                pd += (R - rD_old[j, m] - kappaD_J[j]) * D_JM[j, m]
        profit_j[j] = pl + pd + R * E_J[j]

    borrower_surplus, depositor_surplus = _welfare(
        rL_old, rD_old,
        base_L, kappa_L, e0_L, deltaL_j, home_JM, gammaL0, gammaL1, zL_BM,
        slope_D, e0D_DM, deltaD_j, deltaD_m, beta_branch, log_branches_JM, gammaD0, gammaD1, zD_DM,
        partL_JM, partD_JM,
        sizeL_M, sizeD_M,
        alpha_L, alpha_D,
    )

    return {
        "rL_JM": rL_old,
        "rD_JM": rD_old,
        "L_JM": L_JM,
        "D_JM": D_JM,
        "L_j": L_j,
        "D_j": D_j,
        "I_j": I_j,
        "I_over_D": I_over_D,
        "mu_J": mu_J,
        "slack_J": slack_J,
        "bank_FOC_errors": foc_J,
        "S_L_MJ": S_L_MJ,
        "S_D_MJ": S_D_MJ,
        "profit_by_bank": profit_j,
        "borrower_surplus": borrower_surplus,
        "depositor_surplus": depositor_surplus,
        "info": info,
        "sim_params": params,
    }

# =============================================================================
# 9) Tiny simulator / smoke test
# =============================================================================

def create_params_for_smoke_test():
    return {
        "alpha_L": 23.0,
        "alpha_D": 90.0,
        "beta_branch": 1.3,
        "beta_D_slope": 0.0,
        "c_L": 1.18,
        "r": 0.024,
        "equity_share": 0.17,
        "lambda_liq": 0.39,
        # home bias
        "gammaL0": 0.3,
        "gammaL1": 0.0,
        "gammaD0": 0.2,
        "gammaD1": 0.0,
    }

def simulate_inputs(params, J=10, M=5, B_L=200, B_D=200, seed=42):
    rng = np.random.default_rng(seed)

    deltaL_j = rng.normal(5.5, 0.9, size=J)
    deltaL_m = rng.normal(0.5, 0.5, size=M)
    deltaD_j = rng.normal(-1.5, 1.1, size=J)
    deltaD_m = rng.normal(-1.5, 2.2, size=M)

    partL = rng.random((J, M)) < 0.8
    partD = rng.random((J, M)) < 0.8
    for m in range(M):
        if not partL[:, m].any(): partL[rng.integers(J), m] = True
        if not partD[:, m].any(): partD[rng.integers(J), m] = True
    for j in range(J):
        if not partL[j].any(): partL[j, rng.integers(M)] = True
        if not partD[j].any(): partD[j, rng.integers(M)] = True

    # home indicator (country match) – for smoke test just random
    home = rng.random((J, M)) < 0.5

    sizeL = 75.0 * (1.1 + 0.4 * rng.random(M))
    sizeD = 100.0 * (1.5 + 0.6 * rng.random(M))

    branches = (1.0 + rng.integers(1, 20, size=(J, M))).astype(np.float64)

    kappaL = 0.005 + 0.002 * rng.random(J)
    kappaD = 0.005 + 0.002 * rng.random(J)
    E = 100.0 * np.clip(0.05 + 0.10 * rng.random(J), 0.05, 0.1)

    # heterogeneity draws (log-normal wealth proxy) + weights = wealth
    zL = rng.lognormal(mean=0.0, sigma=0.7, size=(B_L, M))
    wL = zL.copy()

    zD = rng.lognormal(mean=0.0, sigma=0.7, size=(B_D, M))
    wD = zD.copy()

    return (deltaL_j, deltaL_m, deltaD_j, deltaD_m,
            branches, partL, partD, home,
            zL, wL, zD, wD,
            sizeL, sizeD, kappaL, kappaD, E)


def simulate_and_solve(seed=42):
    params = create_params_for_smoke_test()
    J, M, B_L, B_D = 10, 5, 200, 200

    (deltaL_j, deltaL_m, deltaD_j, deltaD_m,
     branches, partL, partD, home,
     zL, wL, zD, wD,
     sizeL, sizeD, kL, kD, E) = simulate_inputs(params, J, M, B_L, B_D, seed)

    out = solve_joint_eqm(
        params,
        deltaL_j, deltaL_m,
        deltaD_j, deltaD_m,
        branches,
        partL, partD,
        home,
        zL, wL,
        zD, wD,
        sizeL, sizeD,
        kL, kD, E,
        rL_min=1.0, rL_max=3.0,
        rD_min=0.5, rD_max=2.0,
        max_iter=200, tol=1e-6, tol_foc=1e-6, damp_fp=0.5,
    )
    return out


if __name__ == "__main__":
    out = simulate_and_solve()

    print(out["info"])
    print("mean rL", float(np.mean(out["rL_JM"])) )
    print("mean rD", float(np.mean(out["rD_JM"])) )
    print("bank FOC errors quantiles [0,.5,.9,.99,1]:", np.quantile(out["bank_FOC_errors"], [0, .5, .9, .99, 1]))

    # liquidity ratio check
    print("I/D quantiles [0,.5,.9,.99,1]:", np.quantile(out["I_over_D"], [0, .5, .9, .99, 1]))

    # share sums by market
    loan_sum = np.sum(out["S_L_MJ"], axis=1)
    dep_sum = np.sum(out["S_D_MJ"], axis=1)
    print("Loan share sum by market (min/median/max):", float(np.min(loan_sum)), float(np.median(loan_sum)), float(np.max(loan_sum)))
    print("Deposit share sum by market (min/median/max):", float(np.min(dep_sum)), float(np.median(dep_sum)), float(np.max(dep_sum)))
