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
def _safe_exp_clip(u, max_val=1000.0):
    if u > max_val:
        u = max_val
    elif u < -max_val:
        u = -max_val
    return np.exp(u)


# =============================================================================
# 1) Build sums of exp utilities at FIXED "others" prices
#    These are used for O(1) replacement of bank j's term.
# =============================================================================

@njit(cache=True, fastmath=True)
def _loan_build_sumexp_with_price(zL_BM, rL_old_JM, xiF_J, home_JM,
                                  gammaF, beta_c, partL_JM, alpha_F):
    """
    sumexp_L[b,m] = Σ_k exp( xiF_k + home*(gammaF+beta_c*L_bm) - alpha_F*rL_{k,m} )
    computed at the "others fixed" price profile rL_old_JM.
    """
    B, M = zL_BM.shape
    J = xiF_J.size
    out = np.zeros((B, M), dtype=np.float64)

    for b in range(B):
        for m in range(M):
            L = zL_BM[b, m]
            acc = 0.0
            for k in range(J):
                if partL_JM[k, m]:
                    hb = 1.0 if home_JM[k, m] else 0.0
                    u = xiF_J[k] + hb * (gammaF + beta_c * L) - alpha_F * rL_old_JM[k, m]
                    acc += _safe_exp_clip(u)
            out[b, m] = acc
    return out

@njit(cache=True, fastmath=True)
def _deposit_build_sumexp_D(zD_DM, rD_old_JM, xiD_J, home_JM,
                            gammaD, alpha_D, beta_w, partD_JM):
    """
    sumexp_D[d,m] = Σ_k exp( xiD_k + home*gammaD + (alpha_D+beta_w*D_dm)*rD_{k,m} )
    computed at the "others fixed" price profile rD_old_JM.
    """
    Dn, M = zD_DM.shape
    J = xiD_J.size
    out = np.zeros((Dn, M), dtype=np.float64)

    for d in range(Dn):
        for m in range(M):
            D = zD_DM[d, m]
            slope = alpha_D + beta_w * D
            acc = 0.0
            for k in range(J):
                if partD_JM[k, m]:
                    hb = 1.0 if home_JM[k, m] else 0.0
                    u = xiD_J[k] + hb * gammaD + slope * rD_old_JM[k, m]
                    acc += _safe_exp_clip(u)
            out[d, m] = acc
    return out

# =============================================================================
# 2) Per-bank-per-market quantities + derivatives (PDF exact)
# =============================================================================

@njit(cache=True, fastmath=True)
def _loan_q_dq_for_jm(
    j, m, r_cand, r_old_jm,
    alpha_F, r_nonbank_m,
    zL_BM, wL_BM, sizeL_m,
    xiF_J, home_JM, gammaF, beta_c,
    partL_JM,
    sumexp_L_with_price
):
    """
    qL_{j,m} = ∫ s_j(L) * L h(L) dL  (discrete approx with weights wL_BM)
    dqL/dr  = -alpha_F ∫ s(1-s) * L h(L) dL
    """
    B = zL_BM.shape[0]
    invB = 1.0 / float(B)

    q = 0.0
    dq = 0.0

    for b in range(B):
        L = zL_BM[b, m]
        w = wL_BM[b, m]

        hb_j = 1.0 if home_JM[j, m] else 0.0
        a_j = xiF_J[j] + hb_j * (gammaF + beta_c * L)

        exp_old = _safe_exp_clip(a_j - alpha_F * r_old_jm)
        exp_new = _safe_exp_clip(a_j - alpha_F * r_cand)

        denom = _safe_exp_clip(-alpha_F * r_nonbank_m) + (sumexp_L_with_price[b, m] - exp_old + exp_new)
        if denom < 1e-300:
            denom = 1e-300

        s = exp_new / denom
        q += w * s
        dq += w * (-alpha_F) * s * (1.0 - s)

    q = sizeL_m * (q * invB)
    dq = sizeL_m * (dq * invB)
    return q, dq

@njit(cache=True, fastmath=True)
def _deposit_q_dq_for_jm(
    j, m, r_cand, r_old_jm,
    alpha_D, beta_w,
    zD_DM, wD_DM, sizeD_m,
    xiD_J, home_JM, gammaD,
    partD_JM,
    sumexp_D
):
    """
    qD_{j,m} = ∫ s_j(D) * D g(D) dD
    dqD/dr  = ∫ slope(D) * s(1-s) * D g(D) dD
    Outside option: 1 in denominator
    """
    Dn = zD_DM.shape[0]
    invD = 1.0 / float(Dn)

    q = 0.0
    dq = 0.0

    for d in range(Dn):
        D = zD_DM[d, m]
        w = wD_DM[d, m]
        slope = alpha_D + beta_w * D

        hb_j = 1.0 if home_JM[j, m] else 0.0
        a_j = xiD_J[j] + hb_j * gammaD

        exp_old = _safe_exp_clip(a_j + slope * r_old_jm)
        exp_new = _safe_exp_clip(a_j + slope * r_cand)

        denom = 1.0 + (sumexp_D[d, m] - exp_old + exp_new)
        if denom < 1e-300:
            denom = 1e-300

        s = exp_new / denom
        q += w * s
        dq += w * slope * s * (1.0 - s)

    q = sizeD_m * (q * invD)
    dq = sizeD_m * (dq * invD)
    return q, dq

# =============================================================================
# 3) 1D market solvers given liquidity ratio x = I/D (PDF exact)
# =============================================================================

@njit(cache=True, fastmath=True)
def _solve_rL_market_pdf(
    j, m,
    r_init, r_old_jm,
    alpha_F, rF, phi, lambda_target, x,
    r_nonbank_m,
    zL_BM, wL_BM, sizeL_m,
    xiF_J, home_JM, gammaF, beta_c, partL_JM,
    sumexp_L_with_price,
    r_min, r_max,
    max_it=80, tol=1e-10
):
    cost = rF + phi * (lambda_target - x)
    r = r_init

    for _ in range(max_it):
        q, dq = _loan_q_dq_for_jm(
            j, m, r, r_old_jm,
            alpha_F, r_nonbank_m,
            zL_BM, wL_BM, sizeL_m,
            xiF_J, home_JM, gammaF, beta_c,
            partL_JM,
            sumexp_L_with_price
        )
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
        j, m, r, r_old_jm,
        alpha_F, r_nonbank_m,
        zL_BM, wL_BM, sizeL_m,
        xiF_J, home_JM, gammaF, beta_c,
        partL_JM,
        sumexp_L_with_price
    )
    foc = q + dq * (r - cost)  # should be ~0
    return r, q, foc

@njit(cache=True, fastmath=True)
def _solve_rD_market_pdf(
    j, m,
    r_init, r_old_jm,
    alpha_D, beta_w, rF, phi, lambda_target, x,
    zD_DM, wD_DM, sizeD_m,
    xiD_J, home_JM, gammaD, partD_JM,
    sumexp_D,
    r_min, r_max,
    max_it=80, tol=1e-10
):
    benefit = rF - 0.5 * phi * (lambda_target - x) * (lambda_target + x)
    r = r_init

    for _ in range(max_it):
        q, dq = _deposit_q_dq_for_jm(
            j, m, r, r_old_jm,
            alpha_D, beta_w,
            zD_DM, wD_DM, sizeD_m,
            xiD_J, home_JM, gammaD,
            partD_JM,
            sumexp_D
        )
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
        j, m, r, r_old_jm,
        alpha_D, beta_w,
        zD_DM, wD_DM, sizeD_m,
        xiD_J, home_JM, gammaD,
        partD_JM,
        sumexp_D
    )
    foc = dq * (benefit - r) - q  # should be ~0
    return r, q, foc

# =============================================================================
# 4) Evaluate bank j quantities given x (Numba-safe helper)
# =============================================================================

@njit(cache=True, fastmath=True)
def _eval_bank_given_x(
    j, x,
    rL_old_JM, rD_old_JM,
    partL_JM, partD_JM,
    zL_BM, wL_BM, zD_DM, wD_DM,
    sizeL_M, sizeD_M,
    xiF_J, xiD_J,
    home_JM, gammaF, beta_c, gammaD,
    alpha_F, alpha_D, beta_w,
    rF, phi, lambda_target, E_j,
    r_nonbank_M,
    sumexp_L, sumexp_D,
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
    max_foc = 0.0

    for m in range(M):
        if partL_JM[j, m]:
            r, q, foc = _solve_rL_market_pdf(
                j, m,
                rL_row[m], rL_old_JM[j, m],
                alpha_F, rF, phi, lambda_target, x,
                r_nonbank_M[m],
                zL_BM, wL_BM, sizeL_M[m],
                xiF_J, home_JM, gammaF, beta_c, partL_JM,
                sumexp_L,
                rL_min, rL_max,
                max_it=max_market_iter
            )
            rL_row[m] = r
            qL_row[m] = q
            if abs(foc) > max_foc:
                max_foc = abs(foc)
            Ltot += q
        else:
            qL_row[m] = 0.0

        if partD_JM[j, m]:
            r, q, foc = _solve_rD_market_pdf(
                j, m,
                rD_row[m], rD_old_JM[j, m],
                alpha_D, beta_w, rF, phi, lambda_target, x,
                zD_DM, wD_DM, sizeD_M[m],
                xiD_J, home_JM, gammaD, partD_JM,
                sumexp_D,
                rD_min, rD_max,
                max_it=max_market_iter
            )
            rD_row[m] = r
            qD_row[m] = q
            if abs(foc) > max_foc:
                max_foc = abs(foc)
            Dtot += q
        else:
            qD_row[m] = 0.0

    if Dtot < 1e-12:
        implied_x = 1e12
    else:
        I = Dtot + E_j - Ltot
        implied_x = I / Dtot

    f = implied_x - x
    return f, implied_x, max_foc, Ltot, Dtot, rL_row, rD_row, qL_row, qD_row

# =============================================================================
# 5) Bank best response: bisection in x = I/D (PDF exact, Numba-safe)
# =============================================================================

@njit(cache=True, fastmath=True)
def _bank_best_response_pdf(
    j,
    rL_old_JM, rD_old_JM,
    partL_JM, partD_JM,
    zL_BM, wL_BM, zD_DM, wD_DM,
    sizeL_M, sizeD_M,
    xiF_J, xiD_J,
    home_JM, gammaF, beta_c, gammaD,
    alpha_F, alpha_D, beta_w,
    rF, phi, lambda_target, E_j,
    r_nonbank_M,
    rL_min, rL_max, rD_min, rD_max,
    max_x_iter=40, max_market_iter=80,
    x_tol=1e-10
):
    # Precompute denominators at "others fixed" prices (including bank j at its old price)
    sumexp_L = _loan_build_sumexp_with_price(zL_BM, rL_old_JM, xiF_J, home_JM, gammaF, beta_c, partL_JM, alpha_F)
    sumexp_D = _deposit_build_sumexp_D(zD_DM, rD_old_JM, xiD_J, home_JM, gammaD, alpha_D, beta_w, partD_JM)

    # Bracket search for f(x)=implied_x-x
    x_lo = -2.0
    x_hi =  5.0

    f_lo, implied_lo, foc_lo, _, _, rL_lo, rD_lo, qL_lo, qD_lo = _eval_bank_given_x(
        j, x_lo,
        rL_old_JM, rD_old_JM, partL_JM, partD_JM,
        zL_BM, wL_BM, zD_DM, wD_DM,
        sizeL_M, sizeD_M,
        xiF_J, xiD_J,
        home_JM, gammaF, beta_c, gammaD,
        alpha_F, alpha_D, beta_w,
        rF, phi, lambda_target, E_j,
        r_nonbank_M,
        sumexp_L, sumexp_D,
        rL_min, rL_max, rD_min, rD_max,
        max_market_iter
    )
    f_hi, implied_hi, foc_hi, _, _, rL_hi, rD_hi, qL_hi, qD_hi = _eval_bank_given_x(
        j, x_hi,
        rL_old_JM, rD_old_JM, partL_JM, partD_JM,
        zL_BM, wL_BM, zD_DM, wD_DM,
        sizeL_M, sizeD_M,
        xiF_J, xiD_J,
        home_JM, gammaF, beta_c, gammaD,
        alpha_F, alpha_D, beta_w,
        rF, phi, lambda_target, E_j,
        r_nonbank_M,
        sumexp_L, sumexp_D,
        rL_min, rL_max, rD_min, rD_max,
        max_market_iter
    )

    # Expand bracket if needed
    have_bracket = (f_lo * f_hi <= 0.0)
    for _ in range(25):
        if have_bracket:
            break
        x_lo *= 2.0
        x_hi *= 2.0
        f_lo, implied_lo, foc_lo, _, _, rL_lo, rD_lo, qL_lo, qD_lo = _eval_bank_given_x(
            j, x_lo,
            rL_old_JM, rD_old_JM, partL_JM, partD_JM,
            zL_BM, wL_BM, zD_DM, wD_DM,
            sizeL_M, sizeD_M,
            xiF_J, xiD_J,
            home_JM, gammaF, beta_c, gammaD,
            alpha_F, alpha_D, beta_w,
            rF, phi, lambda_target, E_j,
            r_nonbank_M,
            sumexp_L, sumexp_D,
            rL_min, rL_max, rD_min, rD_max,
            max_market_iter
        )
        f_hi, implied_hi, foc_hi, _, _, rL_hi, rD_hi, qL_hi, qD_hi = _eval_bank_given_x(
            j, x_hi,
            rL_old_JM, rD_old_JM, partL_JM, partD_JM,
            zL_BM, wL_BM, zD_DM, wD_DM,
            sizeL_M, sizeD_M,
            xiF_J, xiD_J,
            home_JM, gammaF, beta_c, gammaD,
            alpha_F, alpha_D, beta_w,
            rF, phi, lambda_target, E_j,
            r_nonbank_M,
            sumexp_L, sumexp_D,
            rL_min, rL_max, rD_min, rD_max,
            max_market_iter
        )
        have_bracket = (f_lo * f_hi <= 0.0)

    # Best-so-far (min |f|)
    best_abs_f = 1e100
    x_best = 1.0
    max_foc_best = 1e100
    rL_best = rL_lo
    rD_best = rD_lo
    qL_best = qL_lo
    qD_best = qD_lo

    def_abs = abs(f_lo)
    if def_abs < best_abs_f:
        best_abs_f = def_abs
        x_best = x_lo
        max_foc_best = foc_lo
        rL_best = rL_lo
        rD_best = rD_lo
        qL_best = qL_lo
        qD_best = qD_lo

    def_abs = abs(f_hi)
    if def_abs < best_abs_f:
        best_abs_f = def_abs
        x_best = x_hi
        max_foc_best = foc_hi
        rL_best = rL_hi
        rD_best = rD_hi
        qL_best = qL_hi
        qD_best = qD_hi

    if have_bracket:
        # Bisection
        for _ in range(max_x_iter):
            x_mid = 0.5 * (x_lo + x_hi)
            f_mid, implied_mid, max_foc, _, _, rL_mid, rD_mid, qL_mid, qD_mid = _eval_bank_given_x(
                j, x_mid,
                rL_old_JM, rD_old_JM, partL_JM, partD_JM,
                zL_BM, wL_BM, zD_DM, wD_DM,
                sizeL_M, sizeD_M,
                xiF_J, xiD_J,
                home_JM, gammaF, beta_c, gammaD,
                alpha_F, alpha_D, beta_w,
                rF, phi, lambda_target, E_j,
                r_nonbank_M,
                sumexp_L, sumexp_D,
                rL_min, rL_max, rD_min, rD_max,
                max_market_iter
            )

            af = abs(f_mid)
            if af < best_abs_f:
                best_abs_f = af
                x_best = x_mid
                max_foc_best = max_foc
                rL_best = rL_mid
                rD_best = rD_mid
                qL_best = qL_mid
                qD_best = qD_mid

            if af < x_tol:
                break

            # bracket update
            if f_lo * f_mid <= 0.0:
                x_hi = x_mid
                f_hi = f_mid
            else:
                x_lo = x_mid
                f_lo = f_mid
    else:
        # Fallback: damped fixed point x <- implied_x
        x = 1.0
        for _ in range(max_x_iter):
            f_mid, implied_mid, max_foc, _, _, rL_mid, rD_mid, qL_mid, qD_mid = _eval_bank_given_x(
                j, x,
                rL_old_JM, rD_old_JM, partL_JM, partD_JM,
                zL_BM, wL_BM, zD_DM, wD_DM,
                sizeL_M, sizeD_M,
                xiF_J, xiD_J,
                home_JM, gammaF, beta_c, gammaD,
                alpha_F, alpha_D, beta_w,
                rF, phi, lambda_target, E_j,
                r_nonbank_M,
                sumexp_L, sumexp_D,
                rL_min, rL_max, rD_min, rD_max,
                max_market_iter
            )
            x_new = 0.5 * x + 0.5 * implied_mid
            if abs(x_new - x) < x_tol:
                x = x_new
                best_abs_f = abs(f_mid)
                x_best = x
                max_foc_best = max_foc
                rL_best = rL_mid
                rD_best = rD_mid
                qL_best = qL_mid
                qD_best = qD_mid
                break
            x = x_new

    return rL_best, rD_best, qL_best, qD_best, x_best, max_foc_best

# =============================================================================
# 6) Parallel best-response update across banks
# =============================================================================

@njit(cache=True, parallel=True, fastmath=True)
def _update_all_banks_pdf(
    rL_old_JM, rD_old_JM,
    partL_JM, partD_JM,
    zL_BM, wL_BM, zD_DM, wD_DM,
    sizeL_M, sizeD_M,
    xiF_J, xiD_J,
    home_JM, gammaF, beta_c, gammaD,
    alpha_F, alpha_D, beta_w,
    rF, phi, lambda_target, E_J,
    r_nonbank_M,
    rL_min, rL_max, rD_min, rD_max
):
    J, M = rL_old_JM.shape
    rL_br = np.empty((J, M), dtype=np.float64)
    rD_br = np.empty((J, M), dtype=np.float64)
    qL_JM = np.zeros((J, M), dtype=np.float64)
    qD_JM = np.zeros((J, M), dtype=np.float64)
    x_J = np.empty(J, dtype=np.float64)
    foc_J = np.empty(J, dtype=np.float64)

    for j in prange(J):
        rL_row, rD_row, qL_row, qD_row, x, max_foc = _bank_best_response_pdf(
            j,
            rL_old_JM, rD_old_JM,
            partL_JM, partD_JM,
            zL_BM, wL_BM, zD_DM, wD_DM,
            sizeL_M, sizeD_M,
            xiF_J, xiD_J,
            home_JM, gammaF, beta_c, gammaD,
            alpha_F, alpha_D, beta_w,
            rF, phi, lambda_target, E_J[j],
            r_nonbank_M,
            rL_min, rL_max, rD_min, rD_max
        )

        for m in range(M):
            rL_br[j, m] = rL_row[m]
            rD_br[j, m] = rD_row[m]
            qL_JM[j, m] = qL_row[m]
            qD_JM[j, m] = qD_row[m]

        x_J[j] = x
        foc_J[j] = max_foc

    return rL_br, rD_br, qL_JM, qD_JM, x_J, foc_J

# =============================================================================
# 7) SANITY CHECKS (PDF exact)
# =============================================================================

def _frac_at_bounds(X, lo, hi, tol=1e-10):
    X = np.asarray(X)
    at_lo = np.mean(np.isfinite(X) & (X <= lo + tol))
    at_hi = np.mean(np.isfinite(X) & (X >= hi - tol))
    return float(at_lo), float(at_hi)

@njit(cache=True, fastmath=True)
def _loan_inside_share_by_market(rL_JM, zL_BM, xiF_J, home_JM,
                                 gammaF, beta_c, partL_JM, alpha_F, r_nonbank_M):
    J, M = rL_JM.shape
    B = zL_BM.shape[0]
    out = np.zeros(M, dtype=np.float64)

    for m in range(M):
        acc_b = 0.0
        out0 = _safe_exp_clip(-alpha_F * r_nonbank_M[m])
        for b in range(B):
            L = zL_BM[b, m]
            ssum = 0.0
            for j in range(J):
                if partL_JM[j, m]:
                    hb = 1.0 if home_JM[j, m] else 0.0
                    u = xiF_J[j] + hb * (gammaF + beta_c * L) - alpha_F * rL_JM[j, m]
                    ssum += _safe_exp_clip(u)
            denom = out0 + ssum
            if denom < 1e-300:
                denom = 1e-300
            acc_b += ssum / denom
        out[m] = acc_b / float(B)
    return out

@njit(cache=True, fastmath=True)
def _deposit_inside_share_by_market(rD_JM, zD_DM, xiD_J, home_JM,
                                    gammaD, partD_JM, alpha_D, beta_w):
    J, M = rD_JM.shape
    Dn = zD_DM.shape[0]
    out = np.zeros(M, dtype=np.float64)

    for m in range(M):
        acc_d = 0.0
        for d in range(Dn):
            D = zD_DM[d, m]
            slope = alpha_D + beta_w * D
            ssum = 0.0
            for j in range(J):
                if partD_JM[j, m]:
                    hb = 1.0 if home_JM[j, m] else 0.0
                    u = xiD_J[j] + hb * gammaD + slope * rD_JM[j, m]
                    ssum += _safe_exp_clip(u)
            denom = 1.0 + ssum
            if denom < 1e-300:
                denom = 1e-300
            acc_d += ssum / denom
        out[m] = acc_d / float(Dn)
    return out

def sanity_checks_pdf(
    out,
    params,
    xiF_J, xiD_J,
    partL_JM, partD_JM, home_JM,
    L_draws_BM, L_weights_BM,
    D_draws_DM, D_weights_DM,
    sizeL_M, sizeD_M,
    E_J,
    rL_min, rL_max, rD_min, rD_max,
    verbose=True,
    sample_max_markets=None,
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
    sizeL_M = np.asarray(sizeL_M, dtype=np.float64)
    M = sizeL_M.size
    if np.isscalar(r_nonbank):
        r_nonbank_M = np.full(M, float(r_nonbank), dtype=np.float64)
    else:
        r_nonbank_M = np.asarray(r_nonbank, dtype=np.float64).reshape(M)

    rL = np.asarray(out["rL_JM"])
    rD = np.asarray(out["rD_JM"])
    qL = np.asarray(out["qL_JM"])
    qD = np.asarray(out["qD_JM"])
    I  = np.asarray(out["I_j"])
    D  = np.asarray(out["D_j"])
    L  = np.asarray(out["L_j"])
    I_over_D = np.asarray(out["I_over_D"])
    xJ = np.asarray(out["x_J"])
    foc_err = np.asarray(out["bank_FOC_errors"])

    # bounds
    rL_at_lo, rL_at_hi = _frac_at_bounds(rL[np.asarray(partL_JM, bool)], rL_min, rL_max)
    rD_at_lo, rD_at_hi = _frac_at_bounds(rD[np.asarray(partD_JM, bool)], rD_min, rD_max)

    # balance sheet identity and x consistency
    I_check = D + np.asarray(E_J) - L
    bs_max_abs = float(np.max(np.abs(I - I_check)))
    x_cons_max = float(np.max(np.abs(xJ - I_over_D)))

    # recompute FOC residuals at equilibrium (sample markets if requested)
    sumexp_L_eq = _loan_build_sumexp_with_price(
        np.asarray(L_draws_BM, float), np.asarray(rL, float), np.asarray(xiF_J, float),
        np.asarray(home_JM, bool), gammaF, beta_c, np.asarray(partL_JM, bool), alpha_F
    )
    sumexp_D_eq = _deposit_build_sumexp_D(
        np.asarray(D_draws_DM, float), np.asarray(rD, float), np.asarray(xiD_J, float),
        np.asarray(home_JM, bool), gammaD, alpha_D, beta_w, np.asarray(partD_JM, bool)
    )

    J, M2 = rL.shape
    markets_to_check = range(M2) if (sample_max_markets is None) else range(min(M2, int(sample_max_markets)))

    max_loan_res = 0.0
    max_dep_res = 0.0
    bad_dqL = 0
    bad_dqD = 0
    checked_L = 0
    checked_D = 0

    for j in range(J):
        x = float(I_over_D[j])
        cost_L = rF + phi * (lambda_target - x)
        benefit_D = rF - 0.5 * phi * (lambda_target - x) * (lambda_target + x)

        for m in markets_to_check:
            if partL_JM[j, m]:
                qjm, dqjm = _loan_q_dq_for_jm(
                    j, m, rL[j, m], rL[j, m],
                    alpha_F, float(r_nonbank_M[m]),
                    np.asarray(L_draws_BM, float), np.asarray(L_weights_BM, float), float(sizeL_M[m]),
                    np.asarray(xiF_J, float), np.asarray(home_JM, bool), gammaF, beta_c,
                    np.asarray(partL_JM, bool),
                    sumexp_L_eq
                )
                checked_L += 1
                if dqjm >= 0.0:
                    bad_dqL += 1
                rhs = cost_L - (qjm / dqjm) if dqjm != 0.0 else rL[j, m]
                max_loan_res = max(max_loan_res, abs(float(rL[j, m] - rhs)))

            if partD_JM[j, m]:
                qjm, dqjm = _deposit_q_dq_for_jm(
                    j, m, rD[j, m], rD[j, m],
                    alpha_D, beta_w,
                    np.asarray(D_draws_DM, float), np.asarray(D_weights_DM, float), float(sizeD_M[m]),
                    np.asarray(xiD_J, float), np.asarray(home_JM, bool), gammaD,
                    np.asarray(partD_JM, bool),
                    sumexp_D_eq
                )
                checked_D += 1
                if dqjm <= 0.0:
                    bad_dqD += 1
                rhs = benefit_D - (qjm / dqjm) if dqjm != 0.0 else rD[j, m]
                max_dep_res = max(max_dep_res, abs(float(rD[j, m] - rhs)))

    inside_L = _loan_inside_share_by_market(
        np.asarray(rL, float), np.asarray(L_draws_BM, float), np.asarray(xiF_J, float),
        np.asarray(home_JM, bool), gammaF, beta_c, np.asarray(partL_JM, bool), alpha_F, r_nonbank_M
    )
    inside_D = _deposit_inside_share_by_market(
        np.asarray(rD, float), np.asarray(D_draws_DM, float), np.asarray(xiD_J, float),
        np.asarray(home_JM, bool), gammaD, np.asarray(partD_JM, bool), alpha_D, beta_w
    )

    diag = {
        "rates": {
            "rL_min": float(np.min(rL[np.asarray(partL_JM, bool)])),
            "rL_med": float(np.median(rL[np.asarray(partL_JM, bool)])),
            "rL_max": float(np.max(rL[np.asarray(partL_JM, bool)])),
            "rD_min": float(np.min(rD[np.asarray(partD_JM, bool)])),
            "rD_med": float(np.median(rD[np.asarray(partD_JM, bool)])),
            "rD_max": float(np.max(rD[np.asarray(partD_JM, bool)])),
            "frac_rL_at_lower": rL_at_lo,
            "frac_rL_at_upper": rL_at_hi,
            "frac_rD_at_lower": rD_at_lo,
            "frac_rD_at_upper": rD_at_hi,
        },
        "balance_sheet": {
            "max_abs_I_minus_(D+E-L)": bs_max_abs,
            "max_abs_x_minus_I_over_D": x_cons_max,
            "I_over_D_quantiles": np.quantile(I_over_D, [0, .5, .9, .99, 1]).tolist(),
        },
        "focs": {
            "solver_reported_max_bank_foc": float(np.max(foc_err)),
            "recomputed_max_loan_rFOC_residual": float(max_loan_res),
            "recomputed_max_deposit_rFOC_residual": float(max_dep_res),
            "bad_dqL_count": int(bad_dqL),
            "bad_dqD_count": int(bad_dqD),
            "checked_L_cells": int(checked_L),
            "checked_D_cells": int(checked_D),
            "markets_checked_for_foc": int(len(list(markets_to_check))),
        },
        "shares": {
            "inside_share_L_min/med/max": (float(np.min(inside_L)), float(np.median(inside_L)), float(np.max(inside_L))),
            "inside_share_D_min/med/max": (float(np.min(inside_D)), float(np.median(inside_D)), float(np.max(inside_D))),
        }
    }

    if verbose:
        print("\n=== SANITY CHECKS (PDF model) ===")
        print(f"Rates rL min/med/max: {diag['rates']['rL_min']:.4g} / {diag['rates']['rL_med']:.4g} / {diag['rates']['rL_max']:.4g}")
        print(f"Rates rD min/med/max: {diag['rates']['rD_min']:.4g} / {diag['rates']['rD_med']:.4g} / {diag['rates']['rD_max']:.4g}")
        print(f"Frac at bounds: rL(lo,hi)=({rL_at_lo:.3%},{rL_at_hi:.3%}) | rD(lo,hi)=({rD_at_lo:.3%},{rD_at_hi:.3%})")
        print(f"Balance sheet max|I-(D+E-L)| = {bs_max_abs:.3e}")
        print(f"x consistency max|x_J - I/D| = {x_cons_max:.3e}")
        print(f"FOC residuals (recomputed): max loan |r - rhs| = {max_loan_res:.3e} ; max dep |r - rhs| = {max_dep_res:.3e}")
        print(f"dq sign violations: loans bad={bad_dqL}/{checked_L}, deposits bad={bad_dqD}/{checked_D}")
        il0, il1, il2 = diag["shares"]["inside_share_L_min/med/max"]
        id0, id1, id2 = diag["shares"]["inside_share_D_min/med/max"]
        print(f"Inside share by market: loans min/med/max = {il0:.3f}/{il1:.3f}/{il2:.3f}")
        print(f"Inside share by market: deps  min/med/max = {id0:.3f}/{id1:.3f}/{id2:.3f}")

        if rL_at_hi > 0.10:
            print("WARNING: >10% of loan rates at upper bound. Increase rL_max or increase alpha_F.")
        if rD_at_lo > 0.10 or rD_at_hi > 0.10:
            print("WARNING: Many deposit rates at a bound. Widen [rD_min, rD_max].")
        if bs_max_abs > 1e-6:
            print("WARNING: Balance sheet identity not tight (numeric/reporting).")
        if x_cons_max > 1e-6:
            print("WARNING: x_J differs from I/D (check best-response return).")
        if max_loan_res > 1e-6 or max_dep_res > 1e-6:
            print("WARNING: FOC residuals not small (solver tol/bounds).")

    return diag

# =============================================================================
# 8) Public solver (outer Nash fixed point)
# =============================================================================

def solve_joint_eqm_pdf(
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
    run_sanity_checks=True,
    sanity_sample_markets=None,
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
    sizeL_M = np.asarray(sizeL_M, dtype=np.float64)
    M = sizeL_M.size
    if np.isscalar(r_nonbank):
        r_nonbank_M = np.full(M, float(r_nonbank), dtype=np.float64)
    else:
        r_nonbank_M = np.asarray(r_nonbank, dtype=np.float64).reshape(M)

    xiF_J = np.asarray(xiF_J, dtype=np.float64)
    xiD_J = np.asarray(xiD_J, dtype=np.float64)

    partL_JM = np.asarray(partL_JM, dtype=np.bool_)
    partD_JM = np.asarray(partD_JM, dtype=np.bool_)
    home_JM  = np.asarray(home_JM, dtype=np.bool_)

    L_draws_BM = np.asarray(L_draws_BM, dtype=np.float64)
    L_weights_BM = np.asarray(L_weights_BM, dtype=np.float64)
    D_draws_DM = np.asarray(D_draws_DM, dtype=np.float64)
    D_weights_DM = np.asarray(D_weights_DM, dtype=np.float64)

    sizeD_M = np.asarray(sizeD_M, dtype=np.float64)
    E_J = np.asarray(E_J, dtype=np.float64)

    J, M2 = partL_JM.shape
    assert M2 == M, "partL_JM M mismatch vs sizeL_M"
    assert partD_JM.shape == (J, M), "partD_JM shape mismatch"
    assert home_JM.shape == (J, M), "home_JM shape mismatch"

    # initial rates
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

    info = {"converged": False, "iterations": max_iter, "gap": np.nan, "max_foc": np.nan}

    qL_JM = np.zeros((J, M), dtype=np.float64)
    qD_JM = np.zeros((J, M), dtype=np.float64)
    x_J = np.ones(J, dtype=np.float64)
    foc_J = np.ones(J, dtype=np.float64)

    for it in range(1, max_iter + 1):
        rL_br, rD_br, qL_JM, qD_JM, x_J, foc_J = _update_all_banks_pdf(
            rL_old, rD_old,
            partL_JM, partD_JM,
            L_draws_BM, L_weights_BM,
            D_draws_DM, D_weights_DM,
            sizeL_M, sizeD_M,
            xiF_J, xiD_J,
            home_JM, gammaF, beta_c, gammaD,
            alpha_F, alpha_D, beta_w,
            rF, phi, lambda_target, E_J,
            r_nonbank_M,
            rL_min, rL_max, rD_min, rD_max
        )

        rL_new = (1.0 - damp_fp) * rL_old + damp_fp * rL_br
        rD_new = (1.0 - damp_fp) * rD_old + damp_fp * rD_br

        # keep non-participation fixed
        for j in range(J):
            for m in range(M):
                if not partL_JM[j, m]:
                    rL_new[j, m] = rL_old[j, m]
                if not partD_JM[j, m]:
                    rD_new[j, m] = rD_old[j, m]

        gap = max(float(np.max(np.abs(rL_new - rL_old))),
                  float(np.max(np.abs(rD_new - rD_old))))
        max_foc = float(np.max(foc_J))

        if it == 1 or it % 10 == 0:
            print(f"Iter {it:03d}: gap={gap:.3e} | max bank FOC={max_foc:.3e}")

        rL_old, rD_old = rL_new, rD_new

        if gap < tol and max_foc < tol_foc:
            info = {"converged": True, "iterations": it, "gap": gap, "max_foc": max_foc}
            break

    # totals and balance sheet
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
        "info": info,
        "sim_params": params,
    }

    if run_sanity_checks:
        diag = sanity_checks_pdf(
            out=out,
            params=params,
            xiF_J=xiF_J, xiD_J=xiD_J,
            partL_JM=partL_JM, partD_JM=partD_JM, home_JM=home_JM,
            L_draws_BM=L_draws_BM, L_weights_BM=L_weights_BM,
            D_draws_DM=D_draws_DM, D_weights_DM=D_weights_DM,
            sizeL_M=sizeL_M, sizeD_M=sizeD_M,
            E_J=E_J,
            rL_min=rL_min, rL_max=rL_max, rD_min=rD_min, rD_max=rD_max,
            verbose=True,
            sample_max_markets=sanity_sample_markets,
        )
        out["diagnostics"] = diag

    return out

# =============================================================================
# 9) Smoke test
# =============================================================================

def create_params_for_smoke_test_pdf():
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

def simulate_inputs_pdf(params, J=10, M=5, B_L=200, B_D=200, seed=42):
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
    L_weights = L_draws.copy()  # weight by L in qL integral

    D_draws = rng.lognormal(mean=0.0, sigma=0.7, size=(B_D, M))
    D_weights = D_draws.copy()  # weight by D in qD integral

    return xiF, xiD, partL, partD, home, L_draws, L_weights, D_draws, D_weights, sizeL, sizeD, E

def simulate_and_solve_pdf(seed=42):
    params = create_params_for_smoke_test_pdf()
    xiF, xiD, partL, partD, home, Ld, Lw, Dd, Dw, sizeL, sizeD, E = simulate_inputs_pdf(params, seed=seed)

    out = solve_joint_eqm_pdf(
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
        run_sanity_checks=True,
        sanity_sample_markets=None,
    )
    return out

def print_run_report_pdf(out, partL_JM, partD_JM, rL_min, rL_max, rD_min, rD_max):
    import numpy as np

    info = out.get("info", {})
    rL = np.asarray(out["rL_JM"])
    rD = np.asarray(out["rD_JM"])
    qL = np.asarray(out["qL_JM"])
    qD = np.asarray(out["qD_JM"])
    L  = np.asarray(out["L_j"])
    D  = np.asarray(out["D_j"])
    I  = np.asarray(out["I_j"])
    x  = np.asarray(out["I_over_D"])
    xJ = np.asarray(out["x_J"])
    foc = np.asarray(out["bank_FOC_errors"])

    partL = np.asarray(partL_JM, dtype=bool)
    partD = np.asarray(partD_JM, dtype=bool)

    def qtls(a):
        a = np.asarray(a, dtype=float)
        return np.quantile(a[np.isfinite(a)], [0, .5, .9, .99, 1])

    def frac_at_bounds(A, lo, hi, tol=1e-10):
        A = np.asarray(A, dtype=float)
        A = A[np.isfinite(A)]
        if A.size == 0:
            return 0.0, 0.0
        return float(np.mean(A <= lo + tol)), float(np.mean(A >= hi - tol))

    rL_part = rL[partL]
    rD_part = rD[partD]
    rL_lo, rL_hi = frac_at_bounds(rL_part, rL_min, rL_max)
    rD_lo, rD_hi = frac_at_bounds(rD_part, rD_min, rD_max)

    # Basic report
    print("\n" + "=" * 78)
    print("RUN REPORT (PDF model)")
    print("=" * 78)
    print(f"Convergence: {info}")
    print(f"Shapes: rL {rL.shape}, rD {rD.shape}, qL {qL.shape}, qD {qD.shape}")
    print()

    # Rates
    print("Rates (participating cells):")
    print(f"  rL quantiles [0,.5,.9,.99,1] = {qtls(rL_part)}")
    print(f"  rD quantiles [0,.5,.9,.99,1] = {qtls(rD_part)}")
    print(f"  Fraction at bounds: rL(lo,hi)=({rL_lo:.2%},{rL_hi:.2%}) | rD(lo,hi)=({rD_lo:.2%},{rD_hi:.2%})")
    print()

    # Volumes
    print("Per-bank totals:")
    print(f"  L_j quantiles [0,.5,.9,.99,1] = {qtls(L)}")
    print(f"  D_j quantiles [0,.5,.9,.99,1] = {qtls(D)}")
    print(f"  I_j quantiles [0,.5,.9,.99,1] = {qtls(I)}")
    print()

    # Liquidity ratio + x consistency
    print("Liquidity ratio:")
    print(f"  I/D quantiles [0,.5,.9,.99,1] = {qtls(x)}")
    print(f"  max |x_J - I/D| = {float(np.max(np.abs(xJ - x))):.3e}")
    print()

    # Solver errors
    print("FOC / errors:")
    print(f"  bank_FOC_errors quantiles [0,.5,.9,.99,1] = {qtls(foc)}")
    print(f"  max bank_FOC_errors = {float(np.max(foc)):.3e}")
    print()

    # Shares / diagnostics (if you ran sanity checks)
    diag = out.get("diagnostics", None)
    if diag is not None:
        print("Diagnostics (from sanity_checks_pdf):")
        print(f"  Balance sheet max|I-(D+E-L)| = {diag['balance_sheet']['max_abs_I_minus_(D+E-L)']:.3e}")
        print(f"  Recomputed max loan |r - rhs| = {diag['focs']['recomputed_max_loan_rFOC_residual']:.3e}")
        print(f"  Recomputed max dep  |r - rhs| = {diag['focs']['recomputed_max_deposit_rFOC_residual']:.3e}")
        il0, il1, il2 = diag["shares"]["inside_share_L_min/med/max"]
        id0, id1, id2 = diag["shares"]["inside_share_D_min/med/max"]
        print(f"  Inside share by market (loans) min/med/max = {il0:.3f}/{il1:.3f}/{il2:.3f}")
        print(f"  Inside share by market (deps)  min/med/max = {id0:.3f}/{id1:.3f}/{id2:.3f}")
        print()

    # Quick leaderboard (useful for debugging)
    order = np.argsort(x)  # smallest to largest I/D
    k = min(5, x.size)
    print("Bottom 5 banks by I/D (tightest liquidity):")
    for idx in order[:k]:
        print(f"  bank {idx:02d}: I/D={x[idx]: .4f}, L={L[idx]: .4g}, D={D[idx]: .4g}, I={I[idx]: .4g}, FOC={foc[idx]:.2e}")
    print("Top 5 banks by I/D (most liquid):")
    for idx in order[-k:][::-1]:
        print(f"  bank {idx:02d}: I/D={x[idx]: .4f}, L={L[idx]: .4g}, D={D[idx]: .4g}, I={I[idx]: .4g}, FOC={foc[idx]:.2e}")

    print("=" * 78 + "\n")

if __name__ == "__main__":
    out = simulate_and_solve_pdf()

    # This already prints the detailed sanity checks because verbose=True in your call.
    # Now also print a compact but comprehensive run report:
    params = out["sim_params"]
    rL_min, rL_max = 0.5, 5.0
    rD_min, rD_max = -1.0, 5.0

    # Recreate the same simulated inputs to pass participation masks to the report:
    # (or, if you prefer, return partL/partD from simulate_and_solve_pdf)
    params2 = create_params_for_smoke_test_pdf()
    xiF, xiD, partL, partD, home, Ld, Lw, Dd, Dw, sizeL, sizeD, E = simulate_inputs_pdf(params2, seed=42)

    print_run_report_pdf(out, partL, partD, rL_min, rL_max, rD_min, rD_max)
