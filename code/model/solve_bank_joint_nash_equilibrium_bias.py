#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint loan–deposit Nash equilibrium with balance-sheet liquidity wedge.

LOANS (firms):
 u^F_{i,j} = -alpha_F * rL_{j,m}
             + (gammaF + beta_c * L) * 1{home_{j,m}}
             + xiF_j + eps
 Outside option: exp(-alpha_F * r_nonbank_m)
 qL_{j,m} = ∫ s_j(L) * L * h(L) dL

DEPOSITS (households):
 u^D_{i,j} = (alpha_D + beta_w * D) * rD_{j,m}
             + gammaD * 1{home_{j,m}}
             + xiD_j + eps
 Outside option utility normalized to 0 -> denominator includes 1
 qD_{j,m} = ∫ s_j(D) * D * g(D) dD

BANK PRICING FOCs (markets m):
 rL_{j,m} = rF + phi*(lambda_target - I/D) - qL / (dqL/drL)
 rD_{j,m} = rF - 0.5*phi*(lambda_target - I/D)*(lambda_target + I/D) - qD / (dqD/drD)

Balance sheet:
 I + L = D + E  -> I = D + E - L  -> x := I/D
We solve bank best response by finding x consistent with quantities implied by the market FOCs.
"""
import numpy as np
import math
try:
    from numba import njit, prange  # type: ignore
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        def _wrap(func):
            return func
        return _wrap
    prange = range


# =============================================================================
# 0) Numerics + Helpers
# =============================================================================

@njit(cache=True)
def _safe_exp_clip(u, max_val=1000.0):
    if u > max_val:
        u = max_val
    elif u < -max_val:
        u = -max_val
    return np.exp(u)


# ---------------------- helpers: outside option for loans --------------------
@njit(cache=True, fastmath=True)
def _compute_outside_e0_BM(zL_BM, r_nonbank_M, alpha_F,
                           vary_outside_L=False, eta_outside_L=0.0):
    """
    Return e0_BM of shape (B,M) for the loan outside option (non-bank lender).

    Baseline (constant outside across borrower types within a market):
        e0[b,m] = exp( -alpha_F * r_nonbank_M[m] )

    Optional (type-dependent outside utility shift):
        e0[b,m] = exp( -alpha_F * r_nonbank_M[m] ) * exp( eta_outside_L * zL_BM[b,m] )
    """
    B, M = zL_BM.shape
    e0 = np.empty((B, M), dtype=np.float64)
    for b in range(B):
        for m in range(M):
            val = _safe_exp_clip(-alpha_F * r_nonbank_M[m])
            if vary_outside_L:
                val *= _safe_exp_clip(eta_outside_L * zL_BM[b, m])
            e0[b, m] = val
    return e0


# =============================================================================
# 1.1) Loan side precomputation (fast home/foreign split)
# =============================================================================

@njit(cache=True, fastmath=True)
def _loan_precompute_home_factor(zL_BM, gammaF, beta_c):
    """
    home_factor_BM[b,m] = exp(gammaF + beta_c * L_bm)
    This is the multiplicative factor applied to HOME banks in market m.
    """
    B, M = zL_BM.shape
    home_factor = np.empty((B, M), dtype=np.float64)
    for b in range(B):
        for m in range(M):
            home_factor[b, m] = _safe_exp_clip(gammaF + beta_c * zL_BM[b, m])
    return home_factor


@njit(cache=True, fastmath=True)
def _loan_build_S_all_home_foreign(rL_old_JM, xiF_J, home_JM, partL_JM, alpha_F):
    """
    Build two arrays over (m): sums of exp(xiF - alpha_F * rL) separately for:
      - foreign banks in market m
      - home banks in market m
    """
    J, M = rL_old_JM.shape
    S_home_M = np.zeros(M, dtype=np.float64)
    S_for_M  = np.zeros(M, dtype=np.float64)

    for m in range(M):
        sh = 0.0
        sf = 0.0
        for k in range(J):
            if partL_JM[k, m]:
                term = _safe_exp_clip(xiF_J[k] - alpha_F * rL_old_JM[k, m])
                if home_JM[k, m]:
                    sh += term
                else:
                    sf += term
        S_home_M[m] = sh
        S_for_M[m]  = sf

    return S_home_M, S_for_M


@njit(cache=True, fastmath=True)
def _loan_build_sumexp_with_price(zL_BM, rL_old_JM, xiF_J, home_JM,
                                  gammaF, beta_c, partL_JM, alpha_F):
    """
    sumexp_L[b,m] = Σ_k exp( xiF_k + 1{home}*(gammaF+beta_c*L_bm) - alpha_F*rL_{k,m} )
    computed at rL_old_JM.
    """
    B, M = zL_BM.shape
    out = np.empty((B, M), dtype=np.float64)

    home_factor_BM = _loan_precompute_home_factor(zL_BM, gammaF, beta_c)
    S_home_M, S_for_M = _loan_build_S_all_home_foreign(
        rL_old_JM, xiF_J, home_JM, partL_JM, alpha_F
    )

    for b in range(B):
        for m in range(M):
            out[b, m] = S_for_M[m] + home_factor_BM[b, m] * S_home_M[m]

    return out


# =============================================================================
# 1.2) Deposit side precomputation (base + slope-by-type)
# =============================================================================

@njit(cache=True, fastmath=True)
def _deposit_precompute_base_JM(xiD_J, home_JM, gammaD, partD_JM):
    """
    base_JM[k,m] = 1{part} * exp( xiD_k + 1{home_{k,m}} * gammaD )
    """
    J = xiD_J.size
    M = home_JM.shape[1]
    base_JM = np.zeros((J, M), dtype=np.float64)

    for k in range(J):
        xk = xiD_J[k]
        for m in range(M):
            if partD_JM[k, m]:
                hb = 1.0 if home_JM[k, m] else 0.0
                base_JM[k, m] = _safe_exp_clip(xk + hb * gammaD)
            else:
                base_JM[k, m] = 0.0
    return base_JM


@njit(cache=True, fastmath=True)
def _deposit_build_S_all_DM(zD_DM, rD_old_JM, base_JM, alpha_D, beta_w):
    """
    S_all[d,m] = Σ_k base_JM[k,m] * exp( (alpha_D + beta_w*D_{d,m}) * rD_{k,m} )
    """
    Dn, M = zD_DM.shape
    J = rD_old_JM.shape[0]
    S_all = np.zeros((Dn, M), dtype=np.float64)

    for d in range(Dn):
        for m in range(M):
            Dval = zD_DM[d, m]
            slope = alpha_D + beta_w * Dval
            acc = 0.0
            for k in range(J):
                bkm = base_JM[k, m]
                if bkm != 0.0:
                    acc += bkm * _safe_exp_clip(slope * rD_old_JM[k, m])
            S_all[d, m] = acc
    return S_all


@njit(cache=True, fastmath=True)
def _deposit_build_sumexp_D(zD_DM, rD_old_JM, xiD_J, home_JM,
                            gammaD, alpha_D, beta_w, partD_JM):
    """
    sumexp_D[d,m] = Σ_k exp( xiD_k + 1{home}*gammaD + (alpha_D+beta_w*D_dm)*rD_{k,m} )
    computed at rD_old_JM.
    """
    base_JM = _deposit_precompute_base_JM(xiD_J, home_JM, gammaD, partD_JM)
    out = _deposit_build_S_all_DM(zD_DM, rD_old_JM, base_JM, alpha_D, beta_w)
    return out


# =============================================================================
# 2) q and dq for one (j,m) holding rivals fixed at old profile
# =============================================================================

@njit(cache=True, fastmath=True)
def _deposit_q_dq_for_jm(
    j, m,
    rj,                 # candidate r^D_{j,m}
    sumexp_D_DM,        # (Dn, M) computed at rD_old
    zD_DM,              # (Dn, M)
    wD_DM,              # (Dn, M) weights for g(D)dD
    rD_old_JM,          # (J, M)
    xiD_J,
    home_JM,
    gammaD, alpha_D, beta_w,
    partD_JM
):
    """
    q^D_{j,m} = ∫ s_j(D) * D * g(D) dD
    dq/dr     = ∫ slope(D) * s(1-s) * D * g(D) dD,  slope=alpha_D+beta_w*D
    Outside option => exp(u0)=1 in denominator.
    """
    if partD_JM[j, m] == 0:
        return 0.0, 0.0

    Dn, _ = zD_DM.shape
    q = 0.0
    dq = 0.0

    hb_j = 1.0 if home_JM[j, m] else 0.0
    base_j = xiD_J[j] + hb_j * gammaD

    for d in range(Dn):
        D = zD_DM[d, m]
        w = wD_DM[d, m]
        slope = alpha_D + beta_w * D

        exp_old_j = _safe_exp_clip(base_j + slope * rD_old_JM[j, m])
        sumexp_without_j = sumexp_D_DM[d, m] - exp_old_j

        exp_new_j = _safe_exp_clip(base_j + slope * rj)

        denom = 1.0 + sumexp_without_j + exp_new_j
        if denom < 1e-300:
            denom = 1e-300
        s_j = exp_new_j / denom

        q  += w * s_j * D
        dq += w * slope * s_j * (1.0 - s_j) * D

    return q, dq


@njit(cache=True, fastmath=True)
def _loan_q_dq_for_jm(
    j, m,
    rj,                 # candidate r^L_{j,m}
    sumexp_L_BM,        # (Bn, M) computed at rL_old
    e0_BM,              # (Bn, M) outside option term exp(u0) for loans
    zL_BM,              # (Bn, M)
    wL_BM,              # (Bn, M) weights for h(L)dL
    rL_old_JM,          # (J, M) old profile for subtracting exp_old_j
    xiF_J,
    home_JM,
    gammaF, beta_c,
    alpha_F,
    partL_JM
):
    """
    q^L_{j,m} = ∫ s_j(L) * L * h(L) dL
    dq/dr     = -alpha_F ∫ s(1-s) * L * h(L) dL
    Outside option for each (b,m) enters as e0_BM[b,m] in denominator.
    """
    if partL_JM[j, m] == 0:
        return 0.0, 0.0

    Bn, _ = zL_BM.shape
    q = 0.0
    dq = 0.0

    hb_j = 1.0 if home_JM[j, m] else 0.0

    for b in range(Bn):
        L = zL_BM[b, m]
        w = wL_BM[b, m]

        u_old = (-alpha_F * rL_old_JM[j, m]
                 + hb_j * (gammaF + beta_c * L)
                 + xiF_J[j])
        exp_old_j = _safe_exp_clip(u_old)
        sumexp_without_j = sumexp_L_BM[b, m] - exp_old_j

        u_new = (-alpha_F * rj
                 + hb_j * (gammaF + beta_c * L)
                 + xiF_J[j])
        exp_new_j = _safe_exp_clip(u_new)

        denom = e0_BM[b, m] + sumexp_without_j + exp_new_j
        if denom < 1e-300:
            denom = 1e-300
        s_j = exp_new_j / denom

        q  += w * s_j * L
        dq += w * (-alpha_F) * s_j * (1.0 - s_j) * L

    return q, dq


# =============================================================================
# 3) Market-level solvers given liquidity ratio x = I/D
# =============================================================================

@njit(cache=True, fastmath=True)
def _solve_rL_market(
    j, m,
    r_init,
    x,
    rF, phi, lambda_target,
    alpha_F, gammaF, beta_c,
    sumexp_L_BM, e0_BM, zL_BM, wL_BM, rL_old_JM, xiF_J, home_JM, partL_JM,
    sizeL_m,
    r_min, r_max,
    max_it=80, tol=1e-10
):
    cost = rF + phi * (lambda_target - x)
    r = r_init

    for _ in range(max_it):
        q, dq = _loan_q_dq_for_jm(
            j, m, r,
            sumexp_L_BM, e0_BM,
            zL_BM, wL_BM,
            rL_old_JM, xiF_J, home_JM,
            gammaF, beta_c, alpha_F,
            partL_JM
        )
        q  *= sizeL_m
        dq *= sizeL_m

        if dq >= -1e-14:
            r = min(r_max, max(r_min, r + 1e-3))
            continue

        r_tgt = cost - (q / dq)
        if r_tgt < r_min:
            r_tgt = r_min
        elif r_tgt > r_max:
            r_tgt = r_max

        diff = r_tgt - r
        r = r + 0.7 * diff
        if abs(diff) < tol:
            break

    q, dq = _loan_q_dq_for_jm(
        j, m, r,
        sumexp_L_BM, e0_BM,
        zL_BM, wL_BM,
        rL_old_JM, xiF_J, home_JM,
        gammaF, beta_c, alpha_F,
        partL_JM
    )
    q  *= sizeL_m
    dq *= sizeL_m
    foc = q + dq * (r - cost)  # should be ~0
    return r, q, foc


@njit(cache=True, fastmath=True)
def _solve_rD_market(
    j, m,
    r_init,
    x,
    rF, phi, lambda_target,
    alpha_D, beta_w, gammaD,
    sumexp_D_DM, zD_DM, wD_DM, rD_old_JM, xiD_J, home_JM, partD_JM,
    sizeD_m,
    r_min, r_max,
    max_it=80, tol=1e-10
):
    benefit = rF - 0.5 * phi * (lambda_target - x) * (lambda_target + x)
    r = r_init

    for _ in range(max_it):
        q, dq = _deposit_q_dq_for_jm(
            j, m, r,
            sumexp_D_DM,
            zD_DM, wD_DM,
            rD_old_JM,
            xiD_J, home_JM,
            gammaD, alpha_D, beta_w,
            partD_JM
        )
        q  *= sizeD_m
        dq *= sizeD_m

        if dq <= 1e-14:
            r = min(r_max, max(r_min, r - 1e-3))
            continue

        r_tgt = benefit - (q / dq)
        if r_tgt < r_min:
            r_tgt = r_min
        elif r_tgt > r_max:
            r_tgt = r_max

        diff = r_tgt - r
        r = r + 0.7 * diff
        if abs(diff) < tol:
            break

    q, dq = _deposit_q_dq_for_jm(
        j, m, r,
        sumexp_D_DM,
        zD_DM, wD_DM,
        rD_old_JM,
        xiD_J, home_JM,
        gammaD, alpha_D, beta_w,
        partD_JM
    )
    q  *= sizeD_m
    dq *= sizeD_m
    foc = dq * (benefit - r) - q
    return r, q, foc


# =============================================================================
# 4) Evaluate bank j given x, then bank best response (bisection in x)
# =============================================================================

@njit(cache=True, fastmath=True)
def _eval_bank_given_x(
    j, x,
    rL_old_JM, rD_old_JM,
    partL_JM, partD_JM,
    sumexp_L_BM, e0_BM, zL_BM, wL_BM,
    sumexp_D_DM, zD_DM, wD_DM,
    sizeL_M, sizeD_M,
    xiF_J, xiD_J, home_JM,
    rF, phi, lambda_target,
    alpha_F, gammaF, beta_c,
    alpha_D, beta_w, gammaD,
    E_j,
    rL_min, rL_max, rD_min, rD_max,
    max_market_iter
):
  
    M = sizeL_M.size
    rL_row = np.empty(M, dtype=np.float64)
    rD_row = np.empty(M, dtype=np.float64)
    qL_row = np.zeros(M, dtype=np.float64)
    qD_row = np.zeros(M, dtype=np.float64)

    for m in range(M):
        rL_row[m] = rL_old_JM[j, m]
        rD_row[m] = rD_old_JM[j, m]

    Ltot = 0.0
    Dtot = 0.0

    max_foc   = 0.0
    max_foc_L = 0.0
    max_foc_D = 0.0
    worst_m   = -1
    worst_side = 0  # 1=loan, 2=deposit

    for m in range(M):
        if partL_JM[j, m]:
            r, q, foc = _solve_rL_market(
                j, m,
                rL_row[m],
                x,
                rF, phi, lambda_target,
                alpha_F, gammaF, beta_c,
                sumexp_L_BM, e0_BM, zL_BM, wL_BM, rL_old_JM, xiF_J, home_JM, partL_JM,
                sizeL_M[m],
                rL_min, rL_max,
                max_it=max_market_iter
            )
            rL_row[m] = r
            qL_row[m] = q
            Ltot += q

            af = abs(foc)
            if af > max_foc_L:
                max_foc_L = af
            if af > max_foc:
                max_foc = af
                worst_m = m
                worst_side = 1
        else:
            qL_row[m] = 0.0

        if partD_JM[j, m]:
            r, q, foc = _solve_rD_market(
                j, m,
                rD_row[m],
                x,
                rF, phi, lambda_target,
                alpha_D, beta_w, gammaD,
                sumexp_D_DM, zD_DM, wD_DM, rD_old_JM, xiD_J, home_JM, partD_JM,
                sizeD_M[m],
                rD_min, rD_max,
                max_it=max_market_iter
            )
            rD_row[m] = r
            qD_row[m] = q
            Dtot += q

            af = abs(foc)
            if af > max_foc_D:
                max_foc_D = af
            if af > max_foc:
                max_foc = af
                worst_m = m
                worst_side = 2
        else:
            qD_row[m] = 0.0

    if Dtot < 1e-12:
        implied_x = 1e12
    else:
        I = Dtot + E_j - Ltot
        implied_x = I / Dtot

    f = implied_x - x

    return (
        f, implied_x,
        max_foc, max_foc_L, max_foc_D, worst_m, worst_side,
        Ltot, Dtot, rL_row, rD_row, qL_row, qD_row
    )


@njit(cache=True, fastmath=True)
def _bank_best_response(
    j,
    rL_old_JM, rD_old_JM,
    partL_JM, partD_JM,
    zL_BM, wL_BM, e0_BM,
    zD_DM, wD_DM,
    sizeL_M, sizeD_M,
    xiF_J, xiD_J,
    home_JM,
    rF, phi, lambda_target,
    alpha_F, gammaF, beta_c,
    alpha_D, beta_w, gammaD,
    E_j,
    rL_min, rL_max, rD_min, rD_max,
    max_x_iter=40, max_market_iter=80,
    x_tol=1e-10
):
    sumexp_L_BM = _loan_build_sumexp_with_price(
        zL_BM, rL_old_JM, xiF_J, home_JM, gammaF, beta_c, partL_JM, alpha_F
    )
    sumexp_D_DM = _deposit_build_sumexp_D(
        zD_DM, rD_old_JM, xiD_J, home_JM, gammaD, alpha_D, beta_w, partD_JM
    )

    x_lo = -2.0
    x_hi =  5.0

    f_lo, _, foc_lo, focL_lo, focD_lo, wm_lo, ws_lo, _, _, rL_lo, rD_lo, qL_lo, qD_lo = _eval_bank_given_x(
        j, x_lo,
        rL_old_JM, rD_old_JM, partL_JM, partD_JM,
        sumexp_L_BM, e0_BM, zL_BM, wL_BM,
        sumexp_D_DM, zD_DM, wD_DM,
        sizeL_M, sizeD_M,
        xiF_J, xiD_J, home_JM,
        rF, phi, lambda_target,
        alpha_F, gammaF, beta_c,
        alpha_D, beta_w, gammaD,
        E_j,
        rL_min, rL_max, rD_min, rD_max,
        max_market_iter
    )

    f_hi, _, foc_hi, focL_hi, focD_hi, wm_hi, ws_hi, _, _, rL_hi, rD_hi, qL_hi, qD_hi = _eval_bank_given_x(
        j, x_hi,
        rL_old_JM, rD_old_JM, partL_JM, partD_JM,
        sumexp_L_BM, e0_BM, zL_BM, wL_BM,
        sumexp_D_DM, zD_DM, wD_DM,
        sizeL_M, sizeD_M,
        xiF_J, xiD_J, home_JM,
        rF, phi, lambda_target,
        alpha_F, gammaF, beta_c,
        alpha_D, beta_w, gammaD,
        E_j,
        rL_min, rL_max, rD_min, rD_max,
        max_market_iter
    )

    have_bracket = (f_lo * f_hi <= 0.0)
    for _ in range(25):
        if have_bracket:
            break
        x_lo *= 2.0
        x_hi *= 2.0

        f_lo, _, foc_lo, focL_lo, focD_lo, wm_lo, ws_lo, _, _, rL_lo, rD_lo, qL_lo, qD_lo = _eval_bank_given_x(
            j, x_lo,
            rL_old_JM, rD_old_JM, partL_JM, partD_JM,
            sumexp_L_BM, e0_BM, zL_BM, wL_BM,
            sumexp_D_DM, zD_DM, wD_DM,
            sizeL_M, sizeD_M,
            xiF_J, xiD_J, home_JM,
            rF, phi, lambda_target,
            alpha_F, gammaF, beta_c,
            alpha_D, beta_w, gammaD,
            E_j,
            rL_min, rL_max, rD_min, rD_max,
            max_market_iter
        )

        f_hi, _, foc_hi, focL_hi, focD_hi, wm_hi, ws_hi, _, _, rL_hi, rD_hi, qL_hi, qD_hi = _eval_bank_given_x(
            j, x_hi,
            rL_old_JM, rD_old_JM, partL_JM, partD_JM,
            sumexp_L_BM, e0_BM, zL_BM, wL_BM,
            sumexp_D_DM, zD_DM, wD_DM,
            sizeL_M, sizeD_M,
            xiF_J, xiD_J, home_JM,
            rF, phi, lambda_target,
            alpha_F, gammaF, beta_c,
            alpha_D, beta_w, gammaD,
            E_j,
            rL_min, rL_max, rD_min, rD_max,
            max_market_iter
        )

        have_bracket = (f_lo * f_hi <= 0.0)

    best_abs_f = 1e100
    x_best = 1.0

    foc_best  = 1e100
    focL_best = 1e100
    focD_best = 1e100
    wm_best   = -1
    ws_best   = 0

    rL_best = rL_lo
    rD_best = rD_lo
    qL_best = qL_lo
    qD_best = qD_lo

    af = abs(f_lo)
    if af < best_abs_f:
        best_abs_f = af
        x_best = x_lo
        foc_best  = foc_lo
        focL_best = focL_lo
        focD_best = focD_lo
        wm_best   = wm_lo
        ws_best   = ws_lo
        rL_best = rL_lo
        rD_best = rD_lo
        qL_best = qL_lo
        qD_best = qD_lo

    af = abs(f_hi)
    if af < best_abs_f:
        best_abs_f = af
        x_best = x_hi
        foc_best  = foc_hi
        focL_best = focL_hi
        focD_best = focD_hi
        wm_best   = wm_hi
        ws_best   = ws_hi
        rL_best = rL_hi
        rD_best = rD_hi
        qL_best = qL_hi
        qD_best = qD_hi

    if have_bracket:
        for _ in range(max_x_iter):
            x_mid = 0.5 * (x_lo + x_hi)
            f_mid, _, foc_mid, focL_mid, focD_mid, wm_mid, ws_mid, _, _, rL_mid, rD_mid, qL_mid, qD_mid = _eval_bank_given_x(
                j, x_mid,
                rL_old_JM, rD_old_JM, partL_JM, partD_JM,
                sumexp_L_BM, e0_BM, zL_BM, wL_BM,
                sumexp_D_DM, zD_DM, wD_DM,
                sizeL_M, sizeD_M,
                xiF_J, xiD_J, home_JM,
                rF, phi, lambda_target,
                alpha_F, gammaF, beta_c,
                alpha_D, beta_w, gammaD,
                E_j,
                rL_min, rL_max, rD_min, rD_max,
                max_market_iter
            )

            af = abs(f_mid)
            if af < best_abs_f:
                best_abs_f = af
                x_best = x_mid
                foc_best  = foc_mid
                focL_best = focL_mid
                focD_best = focD_mid
                wm_best   = wm_mid
                ws_best   = ws_mid
                rL_best = rL_mid
                rD_best = rD_mid
                qL_best = qL_mid
                qD_best = qD_mid

            if af < x_tol:
                break

            if f_lo * f_mid <= 0.0:
                x_hi = x_mid
                f_hi = f_mid
            else:
                x_lo = x_mid
                f_lo = f_mid
    else:
        x = 1.0
        for _ in range(max_x_iter):
            f_mid, implied_mid, foc_mid, focL_mid, focD_mid, wm_mid, ws_mid, _, _, rL_mid, rD_mid, qL_mid, qD_mid = _eval_bank_given_x(
                j, x,
                rL_old_JM, rD_old_JM, partL_JM, partD_JM,
                sumexp_L_BM, e0_BM, zL_BM, wL_BM,
                sumexp_D_DM, zD_DM, wD_DM,
                sizeL_M, sizeD_M,
                xiF_J, xiD_J, home_JM,
                rF, phi, lambda_target,
                alpha_F, gammaF, beta_c,
                alpha_D, beta_w, gammaD,
                E_j,
                rL_min, rL_max, rD_min, rD_max,
                max_market_iter
            )
            x_new = 0.5 * x + 0.5 * implied_mid
            if abs(x_new - x) < x_tol:
                best_abs_f = abs(f_mid)
                x_best = x_new
                foc_best  = foc_mid
                focL_best = focL_mid
                focD_best = focD_mid
                wm_best   = wm_mid
                ws_best   = ws_mid
                rL_best = rL_mid
                rD_best = rD_mid
                qL_best = qL_mid
                qD_best = qD_mid
                break
            x = x_new

    return rL_best, rD_best, qL_best, qD_best, x_best, foc_best, focL_best, focD_best, wm_best, ws_best


# =============================================================================
# 5) Parallel best-response update across banks
# =============================================================================

@njit(cache=True, parallel=True, fastmath=True)
def _update_all_banks(
    rL_old_JM, rD_old_JM,
    partL_JM, partD_JM,
    zL_BM, wL_BM, e0_BM,
    zD_DM, wD_DM,
    sizeL_M, sizeD_M,
    xiF_J, xiD_J,
    home_JM,
    rF, phi, lambda_target,
    alpha_F, gammaF, beta_c,
    alpha_D, beta_w, gammaD,
    E_J,
    rL_min, rL_max, rD_min, rD_max
):
    J, M = rL_old_JM.shape
    rL_br = np.empty((J, M), dtype=np.float64)
    rD_br = np.empty((J, M), dtype=np.float64)
    qL_JM = np.zeros((J, M), dtype=np.float64)
    qD_JM = np.zeros((J, M), dtype=np.float64)

    x_J   = np.empty(J, dtype=np.float64)
    foc_J = np.empty(J, dtype=np.float64)

    focL_J = np.empty(J, dtype=np.float64)
    focD_J = np.empty(J, dtype=np.float64)
    worst_m_J = np.empty(J, dtype=np.int64)
    worst_side_J = np.empty(J, dtype=np.int64)

    for j in prange(J):
        rL_row, rD_row, qL_row, qD_row, x, max_foc, max_foc_L, max_foc_D, wm, ws = _bank_best_response(
            j,
            rL_old_JM, rD_old_JM,
            partL_JM, partD_JM,
            zL_BM, wL_BM, e0_BM,
            zD_DM, wD_DM,
            sizeL_M, sizeD_M,
            xiF_J, xiD_J,
            home_JM,
            rF, phi, lambda_target,
            alpha_F, gammaF, beta_c,
            alpha_D, beta_w, gammaD,
            E_J[j],
            rL_min, rL_max, rD_min, rD_max
        )

        for m in range(M):
            rL_br[j, m] = rL_row[m]
            rD_br[j, m] = rD_row[m]
            qL_JM[j, m] = qL_row[m]
            qD_JM[j, m] = qD_row[m]

        x_J[j] = x
        foc_J[j] = max_foc
        focL_J[j] = max_foc_L
        focD_J[j] = max_foc_D
        worst_m_J[j] = wm
        worst_side_J[j] = ws

    return rL_br, rD_br, qL_JM, qD_JM, x_J, foc_J, focL_J, focD_J, worst_m_J, worst_side_J


# =============================================================================
# 6) Public solver (outer Nash fixed point)
# =============================================================================

def solve_joint_eqm(
    params,
    xiF_J, xiD_J,
    partL_JM, partD_JM, home_JM,
    L_draws_BM, L_weights_BM,
    D_draws_DM, D_weights_DM,
    sizeL_M, sizeD_M,
    E_J,
    rL_min=0.0, rL_max=10.0,
    rD_min=-5.0, rD_max=10.0,
    max_iter=400, tol=1e-6, tol_foc=1e-6, damp_fp=0.5,
    init_rL=None, init_rD=None,
    verbose=True,
    track_foc_history=False,
    history_stride=1,
    print_bank_diag_every=10,
    print_top_k_banks=5,
):
    alpha_F = float(params["alpha_F"])
    alpha_D = float(params["alpha_D"])
    beta_w  = float(params.get("beta_w", 0.0))
    gammaF  = float(params.get("gammaF", 0.0))
    beta_c  = float(params.get("beta_c", 0.0))
    gammaD  = float(params.get("gammaD", 0.0))
    rF      = float(params["rF"])
    phi     = float(params["phi"])
    lambda_target = float(params["lambda"])

    r_nonbank = params.get("r_nonbank", rF)
    vary_outside_L = bool(params.get("vary_outside_L", False))
    eta_outside_L  = float(params.get("eta_outside_L", 0.0))

    xiF_J = np.asarray(xiF_J, dtype=np.float64)
    xiD_J = np.asarray(xiD_J, dtype=np.float64)

    partL_JM = np.asarray(partL_JM, dtype=np.bool_)
    partD_JM = np.asarray(partD_JM, dtype=np.bool_)
    home_JM  = np.asarray(home_JM, dtype=np.bool_)

    L_draws_BM = np.asarray(L_draws_BM, dtype=np.float64)
    L_weights_BM = np.asarray(L_weights_BM, dtype=np.float64)
    D_draws_DM = np.asarray(D_draws_DM, dtype=np.float64)
    D_weights_DM = np.asarray(D_weights_DM, dtype=np.float64)

    sizeL_M = np.asarray(sizeL_M, dtype=np.float64)
    sizeD_M = np.asarray(sizeD_M, dtype=np.float64)
    E_J = np.asarray(E_J, dtype=np.float64)

    J, M = partL_JM.shape
    assert partD_JM.shape == (J, M)
    assert home_JM.shape == (J, M)
    assert sizeL_M.shape == (M,)
    assert sizeD_M.shape == (M,)

    if np.isscalar(r_nonbank):
        r_nonbank_M = np.full(M, float(r_nonbank), dtype=np.float64)
    else:
        r_nonbank_M = np.asarray(r_nonbank, dtype=np.float64).reshape(M)

    e0_BM = _compute_outside_e0_BM(
        L_draws_BM, r_nonbank_M, alpha_F,
        vary_outside_L=vary_outside_L,
        eta_outside_L=eta_outside_L
    )

    if init_rL is None:
        rL_old = np.full((J, M), rF + 0.02, dtype=np.float64)
    else:
        rL_old = np.asarray(init_rL, dtype=np.float64).copy()

    if init_rD is None:
        rD_old = np.full((J, M), rF - 0.01, dtype=np.float64)
    else:
        rD_old = np.asarray(init_rD, dtype=np.float64).copy()

    rL_old = np.clip(rL_old, rL_min, rL_max)
    rD_old = np.clip(rD_old, rD_min, rD_max)

    qL_JM = np.zeros((J, M), dtype=np.float64)
    qD_JM = np.zeros((J, M), dtype=np.float64)

    x_J   = np.ones(J, dtype=np.float64)
    foc_J = np.ones(J, dtype=np.float64)

    focL_J = np.ones(J, dtype=np.float64)
    focD_J = np.ones(J, dtype=np.float64)
    worst_m_J = np.full(J, -1, dtype=np.int64)
    worst_side_J = np.zeros(J, dtype=np.int64)

    info = {"converged": False, "iterations": max_iter, "gap": np.nan, "max_foc": np.nan}

    # history
    foc_hist = None
    focL_hist = None
    focD_hist = None
    wm_hist = None
    ws_hist = None
    x_hist = None
    gap_hist = None
    maxfoc_hist = None
    iter_hist = None

    brgap_hist = None
    brgap_by_bank_hist = None

    t_ix = 0

    stride = int(max(1, history_stride))

    if track_foc_history:
        T = (max_iter // stride) + 2
        foc_hist  = np.full((T, J), np.nan, dtype=np.float64)
        focL_hist = np.full((T, J), np.nan, dtype=np.float64)
        focD_hist = np.full((T, J), np.nan, dtype=np.float64)
        wm_hist   = np.full((T, J), -1, dtype=np.int64)
        ws_hist   = np.full((T, J),  0, dtype=np.int64)

        x_hist   = np.full((T, J), np.nan, dtype=np.float64)
        gap_hist = np.full(T, np.nan, dtype=np.float64)
        maxfoc_hist = np.full(T, np.nan, dtype=np.float64)
        iter_hist = np.full(T, -1, dtype=np.int64)
        brgap_hist = np.full(T, np.nan, dtype=np.float64)
        brgap_by_bank_hist = np.full((T, J), np.nan, dtype=np.float64)

    prev_print_foc = None
    
    def _side(ws):
        return "L" if ws == 1 else ("D" if ws == 2 else "?")

    for it in range(1, max_iter + 1):
        (rL_br, rD_br, qL_JM, qD_JM, x_J,
         foc_J, focL_J, focD_J, worst_m_J, worst_side_J) = _update_all_banks(
            rL_old, rD_old,
            partL_JM, partD_JM,
            L_draws_BM, L_weights_BM, e0_BM,
            D_draws_DM, D_weights_DM,
            sizeL_M, sizeD_M,
            xiF_J, xiD_J,
            home_JM,
            rF, phi, lambda_target,
            alpha_F, gammaF, beta_c,
            alpha_D, beta_w, gammaD,
            E_J,
            rL_min, rL_max, rD_min, rD_max
        )

        # --- Nash fixed-point diagnostic: per-bank best-response distance
        br_gap_L = np.max(np.abs(rL_br - rL_old), axis=1)  # (J,)
        br_gap_D = np.max(np.abs(rD_br - rD_old), axis=1)  # (J,)
        br_gap_J = np.maximum(br_gap_L, br_gap_D)          # (J,)
        max_br_gap = float(np.max(br_gap_J))

        rL_new = (1.0 - damp_fp) * rL_old + damp_fp * rL_br
        rD_new = (1.0 - damp_fp) * rD_old + damp_fp * rD_br

        rL_new[~partL_JM] = rL_old[~partL_JM]
        rD_new[~partD_JM] = rD_old[~partD_JM]

        gap = max(float(np.max(np.abs(rL_new - rL_old))),
                  float(np.max(np.abs(rD_new - rD_old))))
        max_foc = float(np.max(foc_J))

        if track_foc_history:
            if (it == 1) or ((it % stride) == 0):
                foc_hist[t_ix, :]  = foc_J
                focL_hist[t_ix, :] = focL_J
                focD_hist[t_ix, :] = focD_J
                wm_hist[t_ix, :]   = worst_m_J
                ws_hist[t_ix, :]   = worst_side_J

                x_hist[t_ix, :]    = x_J
                gap_hist[t_ix]     = gap
                maxfoc_hist[t_ix]  = max_foc
                iter_hist[t_ix]    = it
                brgap_hist[t_ix] = max_br_gap
                brgap_by_bank_hist[t_ix, :] = br_gap_J
                t_ix += 1

        if verbose and (it == 1 or it % int(max(1, print_bank_diag_every)) == 0):
            p50 = float(np.median(foc_J))
            p90 = float(np.percentile(foc_J, 90))
            mx  = float(np.max(foc_J))

            order = np.argsort(foc_J)[::-1]
            K = min(int(max(1, print_top_k_banks)), J)
            top = order[:K]

            top_str = ", ".join([
                f"j={int(j)}:{foc_J[j]:.2e} "
                f"(L={focL_J[j]:.2e}, D={focD_J[j]:.2e}, worst={_side(int(worst_side_J[j]))}@m={int(worst_m_J[j])}, x={x_J[j]:.3f})"
                for j in top
            ])

            improve_msg = ""
            if prev_print_foc is not None:
                jw = int(top[0])
                improve_msg = f" | worst-bank ΔFOC={float(foc_J[jw] - prev_print_foc[jw]):+.2e}"
            prev_print_foc = foc_J.copy()

            print(
                f"Iter {it:03d}: gap={gap:.3e} | maxBRgap={max_br_gap:.3e} | maxFOC(all)={max_foc:.3e} "
                f"| bankFOC p50={p50:.2e}, p90={p90:.2e}, max={mx:.2e} "
                f"| worst: {top_str}{improve_msg}"
            )


        rL_old, rD_old = rL_new, rD_new

        if gap < tol and max_foc < tol_foc:
            info = {"converged": True, "iterations": it, "gap": gap, "max_foc": max_foc}
            if track_foc_history and (t_ix > 0) and (iter_hist[t_ix-1] != it):
                foc_hist[t_ix, :]  = foc_J
                focL_hist[t_ix, :] = focL_J
                focD_hist[t_ix, :] = focD_J
                wm_hist[t_ix, :]   = worst_m_J
                ws_hist[t_ix, :]   = worst_side_J

                x_hist[t_ix, :]    = x_J
                gap_hist[t_ix]     = gap
                maxfoc_hist[t_ix]  = max_foc
                iter_hist[t_ix]    = it
                t_ix += 1
            break

    L_j = np.sum(qL_JM, axis=1)
    D_j = np.sum(qD_JM, axis=1)
    I_j = D_j + E_J - L_j
    with np.errstate(divide="ignore", invalid="ignore"):
        I_over_D = I_j / np.maximum(D_j, 1e-12)

    out = {
        "rL_JM": rL_old,
        "rD_JM": rD_old,
        "qL_JM": qL_JM,
        "qD_JM": qD_JM,
        "L_j": L_j,
        "D_j": D_j,
        "I_j": I_j,
        "I_over_D": I_over_D,
        "x_J": x_J,
        "bank_FOC_errors": foc_J,
        "bank_FOC_L": focL_J,
        "bank_FOC_D": focD_J,
        "bank_worst_m": worst_m_J,
        "bank_worst_side": worst_side_J,
        "info": info,
        "sim_params": params,
    }

    if track_foc_history:
        out["history"] = {
            "iter": iter_hist[:t_ix],
            "foc_by_bank": foc_hist[:t_ix],
            "focL_by_bank": focL_hist[:t_ix],
            "focD_by_bank": focD_hist[:t_ix],
            "worst_m_by_bank": wm_hist[:t_ix],
            "worst_side_by_bank": ws_hist[:t_ix],
            "x_by_bank": x_hist[:t_ix],
            "gap": gap_hist[:t_ix],
            "max_foc": maxfoc_hist[:t_ix],
            "stride": stride,
            "max_br_gap": brgap_hist[:t_ix],
             "br_gap_by_bank": brgap_by_bank_hist[:t_ix],
        }

    return out


# =============================================================================
# 7) Test
# =============================================================================

def create_params_for_test():
    return {
        "alpha_F": 25.0,
        "alpha_D":  2.0,
        "beta_w":   0.01,
        "gammaF":   0.4,
        "beta_c":  -0.01,
        "gammaD":   0.2,
        "rF":       1.02,
        "phi":      0.5,
        "lambda":   0.2,
        "r_nonbank": 1.05,
    }


def simulate_inputs(params, J=10, M=5, B_L=200, B_D=200, seed=42):
    rng = np.random.default_rng(seed)

    partL = rng.random((J, M)) < 0.85
    partD = rng.random((J, M)) < 0.85
    for m in range(M):
        if not partL[:, m].any(): partL[rng.integers(J), m] = True
        if not partD[:, m].any(): partD[rng.integers(J), m] = True
    for j in range(J):
        if not partL[j, :].any(): partL[j, rng.integers(M)] = True
        if not partD[j, :].any(): partD[j, rng.integers(M)] = True

    home = rng.random((J, M)) < 0.5

    sizeL = 75.0 * (1.1 + 0.4 * rng.random(M))
    sizeD = 100.0 * (1.5 + 0.6 * rng.random(M))

    xiF = rng.normal(0.0, 0.7, size=J)
    xiD = rng.normal(0.0, 0.7, size=J)

    E = 100.0 * np.clip(0.05 + 0.10 * rng.random(J), 0.05, 0.2)

    L_draws = rng.lognormal(mean=0.0, sigma=0.7, size=(B_L, M))
    D_draws = rng.lognormal(mean=0.0, sigma=0.7, size=(B_D, M))

    L_weights = np.full((B_L, M), 1.0 / B_L, dtype=float)
    D_weights = np.full((B_D, M), 1.0 / B_D, dtype=float)

    return xiF, xiD, partL, partD, home, L_draws, L_weights, D_draws, D_weights, sizeL, sizeD, E


if __name__ == "__main__":
    params = create_params_for_test()
    xiF, xiD, partL, partD, home, Ld, Lw, Dd, Dw, sizeL, sizeD, E = simulate_inputs(params, seed=42)

    out = solve_joint_eqm(
        params,
        xiF, xiD,
        partL, partD, home,
        Ld, Lw,
        Dd, Dw,
        sizeL, sizeD,
        E,
        rL_min=0.5, rL_max=5.0,
        rD_min=-1.0, rD_max=5.0,
        max_iter=200, tol=1e-6, tol_foc=1e-6, damp_fp=0.5,
        track_foc_history=True,
        history_stride=1,
        print_bank_diag_every=10,
        print_top_k_banks=5,
        verbose=True
    )

    hist = out["history"]
    print("\nFirst 10 max_foc values:", hist["max_foc"][:10])

    final_foc  = hist["foc_by_bank"][-1, :]
    final_focL = hist["focL_by_bank"][-1, :]
    final_focD = hist["focD_by_bank"][-1, :]
    final_x    = hist["x_by_bank"][-1, :]
    final_wm   = hist["worst_m_by_bank"][-1, :]
    final_ws   = hist["worst_side_by_bank"][-1, :]
    

    order = np.argsort(final_foc)[::-1]
    print("\nFinal per-bank max|FOC| (worst to best):")
    for j in order:
        side = "L" if int(final_ws[j]) == 1 else ("D" if int(final_ws[j]) == 2 else "?")
        print(
            f"  j={int(j):3d}: {final_foc[j]:.3e}  "
            f"(L={final_focL[j]:.3e}, D={final_focD[j]:.3e}, worst={side}@m={int(final_wm[j])}, x={final_x[j]:.4f})"
        )

    print("\nConvergence info:", out["info"])
    print("Max bank FOC:", float(np.max(out["bank_FOC_errors"])))
    print("Median I/D:", float(np.median(out["I_over_D"])))

    hist = out["history"]
    j = 1

    iters = hist["iter"]
    foc  = hist["foc_by_bank"][:, j]
    focL = hist["focL_by_bank"][:, j]
    focD = hist["focD_by_bank"][:, j]
    x    = hist["x_by_bank"][:, j]
    wm   = hist["worst_m_by_bank"][:, j]
    ws   = hist["worst_side_by_bank"][:, j]
    hist = out["history"]
    for t in range(len(hist["iter"])):
        print(
            f"it={int(hist['iter'][t]):3d}  "
            f"FOC={hist['foc_by_bank'][t, j]:.3e}  "
            f"BRgap={hist['br_gap_by_bank'][t, j]:.3e}  "
            f"x={hist['x_by_bank'][t, j]:.4f}"
        )


    def side(ws):
        return "L" if ws == 1 else ("D" if ws == 2 else "?")

    for t in range(len(iters)):
        print(f"it={int(iters[t]):3d}  j={j:2d}  FOC={foc[t]:.3e}  "
            f"(L={focL[t]:.3e}, D={focD[t]:.3e})  "
            f"worst={side(int(ws[t]))}@m={int(wm[t])}  x={x[t]:.4f}")

