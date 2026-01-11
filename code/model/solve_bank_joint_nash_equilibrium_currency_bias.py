#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solve_bank_joint_nash_equilibrium_currency_bias.py

European banking integration (currency-choice + bank pricing) model.

IMPORTANT:
- This module must be SAFE TO IMPORT (no runs at import time).
- If you want to test it, run it directly: `python solve_bank_joint_nash_equilibrium_currency_bias.py`

Model notes (practical):
- Deposit demand is logit across (bank, currency) + outside options.
- Inner loop: banks set rates given Q.
- Outer loop: Q updates from implied holdings.

Units:
- Rates are DECIMALS (e.g. 0.03 = 3%).
- alpha0/alpha1 are typically calibrated as "per percentage point" sensitivities,
  so we multiply by RATE_SCALE=100 by default.

*** IMPORTANT CHANGE (to match the PDF exactly) ***
Bank FOC update now implements:

    r_{n,b,c} = ( s_{home(n),c} / f_{home(n),c} ) *
                [ (r^L_b - m_{n,b,c}) - ( q_{n,b,c} / (dq/dr)_{n,b,c} ) ]

i.e. the spot/forward conversion factor uses ONLY household-country prices (n,c),
and it multiplies the ENTIRE bracket (including the markup term).
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# Demand system
# =============================================================================

def _ensure_r_out_nc(data: dict) -> np.ndarray:
    """
    Option 1 outside option:
      - If data has r_out_nc (N,C), use it.
      - Else if data has r_out_n (N,), replicate across currencies.
      - Else default zeros.
    """
    N = int(data["N"])
    C = int(data["C"])

    if "r_out_nc" in data and data["r_out_nc"] is not None:
        r_out_nc = np.asarray(data["r_out_nc"], dtype=float)
        if r_out_nc.shape != (N, C):
            raise ValueError(f"r_out_nc must have shape (N,C)=({N},{C}), got {r_out_nc.shape}")
        return r_out_nc

    if "r_out_n" in data and data["r_out_n"] is not None:
        r_out_n = np.asarray(data["r_out_n"], dtype=float)
        if r_out_n.shape != (N,):
            raise ValueError(f"r_out_n must have shape (N,), got {r_out_n.shape}")
        return np.tile(r_out_n[:, None], (1, C))

    return np.zeros((N, C), dtype=float)


def compute_demand_system(
    r_nbc: np.ndarray,              # (N,B,C) deposit rates (decimals)
    Q_c: np.ndarray,                # (C,) currency depth (physical units)
    W_grid_nk: np.ndarray,          # (N,K) wealth grid in home-currency units
    W_wgt_nk: np.ndarray,           # (N,K) weights (sum_k=1 per n)
    alpha0: float,
    alpha1: float,
    delta_b: np.ndarray,            # (B,)
    nu_home: float,
    gamma: float,
    home_c_of_n: np.ndarray,        # (N,)
    spot_home_c: np.ndarray,        # (N,C) price of c in home n
    fwd_home_c: np.ndarray,         # (N,C)
    r_out_nc: np.ndarray,           # (N,C)
    offer_nbc: np.ndarray,          # (N,B,C) bool
    RATE_SCALE: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      q_nbc_home: (N,B,C) demand in *home-currency units*
      dq_nbc:     (N,B,C) derivative dq/dr
    """
    N, B, C = r_nbc.shape
    K = W_grid_nk.shape[1]

    q_nbc_home = np.zeros((N, B, C), dtype=float)
    dq_nbc = np.zeros((N, B, C), dtype=float)

    # I_home(n,c)
    is_home_nc = np.zeros((N, C), dtype=float)
    for n in range(N):
        is_home_nc[n, int(home_c_of_n[n])] = 1.0

    for n in range(N):
        s_nc = np.asarray(spot_home_c[n], dtype=float)   # (C,)
        f_nc = np.asarray(fwd_home_c[n], dtype=float)    # (C,)
        safe_s = np.maximum(s_nc, 1e-12)

        # convert rates in currency c into home-n units by f/s
        # (this matches the PDF's u_{n,b,c} term that uses f/s on r)
        price_mult_c = f_nc / safe_s  # (C,)

        # depth utility: gamma * Q_c * s_{home,c}
        depth_util_c = gamma * np.asarray(Q_c, dtype=float) * s_nc

        home_util_c = nu_home * is_home_nc[n]  # (C,)

        static_util_bc = np.asarray(delta_b, dtype=float)[:, None] + home_util_c[None, :] + depth_util_c[None, :]

        for k in range(K):
            W = float(W_grid_nk[n, k])
            wgt = float(W_wgt_nk[n, k])

            alpha_i = (alpha0 + alpha1 * W) * RATE_SCALE  # per-pp scaling

            u_bc = alpha_i * (r_nbc[n] * price_mult_c[None, :]) + static_util_bc  # (B,C)
            u_out_c = alpha_i * np.asarray(r_out_nc[n], dtype=float)              # (C,)

            max_u = np.max(np.maximum(np.max(u_bc), np.max(u_out_c)))

            exp_bc = np.exp(u_bc - max_u) * offer_nbc[n]
            exp_out = np.exp(u_out_c - max_u)

            denom = float(np.sum(exp_bc) + np.sum(exp_out))
            if denom <= 0.0:
                continue

            probs_bc = exp_bc / denom  # (B,C)

            q_nbc_home[n] += wgt * probs_bc * W

            # dq/dr = W * P*(1-P) * dU/dr   (own derivative in multinomial logit)
            slope_factor = alpha_i * price_mult_c[None, :]
            dq_nbc[n] += wgt * W * probs_bc * (1.0 - probs_bc) * slope_factor

    return q_nbc_home, dq_nbc


# =============================================================================
# Inner loop: pricing given Q
# =============================================================================

def solve_prices_fixed_Q(
    r_init: np.ndarray,
    Q_c: np.ndarray,
    *,
    rL_b: np.ndarray,               # (B,)
    m_nbc: np.ndarray,              # (N,B,C)
    spot_bank_c: np.ndarray,        # (B,C) UNUSED now (kept for backwards compatibility)
    W_grid_nk: np.ndarray,
    W_wgt_nk: np.ndarray,
    alpha0: float,
    alpha1: float,
    delta_b: np.ndarray,
    nu_home: float,
    gamma: float,
    home_c_of_n: np.ndarray,
    spot_home_c: np.ndarray,
    fwd_home_c: np.ndarray,
    r_out_nc: np.ndarray,
    offer_nbc: np.ndarray,
    RATE_SCALE: float = 100.0,
    max_iter: int = 250,
    tol: float = 1e-8,
    damp: float = 0.6,
    r_min: float = -0.02,
    r_max: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterates on rearranged PDF FOC (exact mapping):

      r_{n,b,c} = (s_{n,c}/f_{n,c}) * [ (rL_b - m_{n,b,c}) - q_{n,b,c}/(dq/dr)_{n,b,c} ].

    Notes:
    - Uses household-country spot/forward ONLY (no bank-based spot in the prefactor).
    - The prefactor multiplies the entire bracket, including the markup term.
    """
    _ = spot_bank_c  # explicitly unused

    r = np.asarray(r_init, dtype=float).copy()

    # conv_factor(n,c) = s_{n,c}/f_{n,c}, broadcast over banks b
    s_home = np.asarray(spot_home_c, dtype=float)[:, None, :]  # (N,1,C)
    f_home = np.asarray(fwd_home_c, dtype=float)[:, None, :]   # (N,1,C)
    conv_factor = s_home / np.maximum(f_home, 1e-12)           # (N,1,C) -> broadcasts to (N,B,C)

    rL_b = np.asarray(rL_b, dtype=float)[None, :, None]        # (1,B,1)
    m_nbc = np.asarray(m_nbc, dtype=float)

    for _it in range(max_iter):
        q_home, dq = compute_demand_system(
            r, Q_c,
            np.asarray(W_grid_nk, dtype=float),
            np.asarray(W_wgt_nk, dtype=float),
            float(alpha0), float(alpha1),
            np.asarray(delta_b, dtype=float),
            float(nu_home),
            float(gamma),
            np.asarray(home_c_of_n, dtype=int),
            np.asarray(spot_home_c, dtype=float),
            np.asarray(fwd_home_c, dtype=float),
            np.asarray(r_out_nc, dtype=float),
            np.asarray(offer_nbc, dtype=bool),
            RATE_SCALE=float(RATE_SCALE),
        )

        # markup term = q / (dq/dr)
        safe_dq = np.where(dq > 1e-14, dq, np.inf)
        markup = q_home / safe_dq

        # exact PDF mapping: r = conv * ( (rL - m) - markup )
        bracket = (rL_b - m_nbc) - markup
        r_target = np.clip(conv_factor * bracket, r_min, r_max)

        mask = offer_nbc & (dq > 1e-14)
        r_new = r.copy()
        r_new[mask] = (1.0 - damp) * r[mask] + damp * r_target[mask]

        diff = float(np.max(np.abs(r_new - r)))
        r = r_new
        if diff < tol:
            break

    return r, q_home, dq


# =============================================================================
# Outer loop: equilibrium in (r, Q)
# =============================================================================

def solve_equilibrium(params: dict, data: dict, settings: dict | None = None) -> dict:
    if settings is None:
        settings = {}

    N = int(data["N"]); B = int(data["B"]); C = int(data["C"])

    # settings
    max_outer = int(settings.get("max_outer", 60))
    tol_Q = float(settings.get("tol_Q", 1e-6))
    damp_Q = float(settings.get("damp_Q", 0.5))

    max_inner = int(settings.get("max_inner", 250))
    tol_inner = float(settings.get("tol_inner", 1e-8))
    damp_inner = float(settings.get("damp_inner", 0.6))

    r_min = float(settings.get("r_min", -0.02))
    r_max = float(settings.get("r_max", 0.10))

    RATE_SCALE = float(params.get("RATE_SCALE", data.get("RATE_SCALE", 100.0)))

    # initialise
    Q_c = np.asarray(data.get("Q_init", np.ones(C) * 50.0), dtype=float).copy()
    r_nbc = np.asarray(data.get("r_init", np.full((N, B, C), 0.01)), dtype=float).copy()

    # required objects
    spot_home_c = np.asarray(data["spot_home_c"], dtype=float)
    r_out_nc = _ensure_r_out_nc(data)

    # helpful print
    if bool(settings.get("verbose_outer", True)):
        print(f"{'Iter':<5} | {'relDiff(Q)':<12} | {'max(Q)':<12} | {'avg r':<10}")
        print("-" * 52)

    rel_diff_Q = np.inf
    offer_nbc = np.asarray(data["offer_nbc"], dtype=bool)

    for outer_it in range(max_outer):
        Q_prev = Q_c.copy()

        r_nbc, q_home, dq = solve_prices_fixed_Q(
            r_nbc, Q_c,
            rL_b=np.asarray(data["rL_b"], dtype=float),
            m_nbc=np.asarray(data["m_nbc"], dtype=float),
            spot_bank_c=np.asarray(data["spot_bank_c"], dtype=float),  # kept but unused
            W_grid_nk=np.asarray(data["W_grid_nk"], dtype=float),
            W_wgt_nk=np.asarray(data["W_wgt_nk"], dtype=float),
            alpha0=float(params["alpha0"]),
            alpha1=float(params["alpha1"]),
            delta_b=np.asarray(params["delta_b"], dtype=float),
            nu_home=float(params["nu_home"]),
            gamma=float(params["gamma"]),
            home_c_of_n=np.asarray(data["home_c_of_n"], dtype=int),
            spot_home_c=spot_home_c,
            fwd_home_c=np.asarray(data["fwd_home_c"], dtype=float),
            r_out_nc=r_out_nc,
            offer_nbc=offer_nbc,
            RATE_SCALE=RATE_SCALE,
            max_iter=max_inner,
            tol=tol_inner,
            damp=damp_inner,
            r_min=r_min,
            r_max=r_max,
        )

        # update Q in PHYSICAL units: sum_{n,b} q_home / spot_home
        Q_new = np.zeros(C, dtype=float)
        for c in range(C):
            denom = np.maximum(spot_home_c[:, c], 1e-12)   # (N,)
            phys = q_home[:, :, c] / denom[:, None]        # (N,B)
            Q_new[c] = float(np.sum(phys))

        Q_c = (1.0 - damp_Q) * Q_prev + damp_Q * Q_new
        rel_diff_Q = float(np.max(np.abs(Q_c - Q_prev) / (np.abs(Q_prev) + 1e-12)))

        avg_r = float(np.mean(r_nbc[offer_nbc])) if np.any(offer_nbc) else float("nan")

        if bool(settings.get("verbose_outer", True)):
            print(f"{outer_it+1:<5} | {rel_diff_Q:<12.2e} | {np.max(Q_c):<12.4f} | {avg_r:<10.5f}")

        if rel_diff_Q < tol_Q:
            break

    return {
        "r_nbc": r_nbc,
        "Q_c": Q_c,
        "q_nbc_home": q_home,
        "dq_nbc": dq,
        "info": {
            "converged": bool(rel_diff_Q < tol_Q),
            "outer_iters": outer_it + 1,
            "rel_diff_Q": rel_diff_Q,
        },
    }


# =============================================================================
# Optional: quick self-test (ONLY runs when executed directly)
# =============================================================================

def _make_toy_data(N=8, B=30, C=3, K=25, seed=123, offer_prob=0.85) -> dict:
    rng = np.random.default_rng(seed)

    home_c_of_n = rng.integers(0, C, size=N)
    bank_home_c = rng.integers(0, C, size=B)

    spot_home_c = np.exp(rng.normal(0.0, 0.25, size=(N, C)))
    for n in range(N):
        spot_home_c[n, int(home_c_of_n[n])] = 1.0
    fwd_home_c = spot_home_c.copy()

    # still created for compatibility with existing callers (even though unused in pricing now)
    spot_bank_c = np.zeros((B, C), dtype=float)
    for b in range(B):
        candidates = np.where(home_c_of_n == bank_home_c[b])[0]
        if candidates.size > 0:
            spot_bank_c[b, :] = spot_home_c[int(candidates[0]), :]
        else:
            v = np.exp(rng.normal(0.0, 0.25, size=C))
            v[int(bank_home_c[b])] = 1.0
            spot_bank_c[b, :] = v

    W_grid = np.exp(np.linspace(0.0, 2.0, K))
    W_wgt = np.ones(K) / K
    W_grid_nk = np.tile(W_grid, (N, 1))
    W_wgt_nk = np.tile(W_wgt, (N, 1))

    rL_b = 0.05 + 0.03 * rng.random(B)
    m_nbc = 0.0015 + 0.0015 * rng.random((N, B, C))

    offer_nbc = rng.random((N, B, C)) < float(offer_prob)
    for n in range(N):
        for c in range(C):
            if not offer_nbc[n, :, c].any():
                offer_nbc[n, rng.integers(0, B), c] = True

    # Option 1 outside option
    r_out_n = np.full(N, 0.01, dtype=float)  # 1% outside option

    return {
        "N": N, "B": B, "C": C,
        "home_c_of_n": home_c_of_n,
        "bank_home_c": bank_home_c,
        "spot_home_c": spot_home_c,
        "fwd_home_c": fwd_home_c,
        "spot_bank_c": spot_bank_c,  # kept but unused by pricing now
        "W_grid_nk": W_grid_nk,
        "W_wgt_nk": W_wgt_nk,
        "rL_b": rL_b,
        "m_nbc": m_nbc,
        "offer_nbc": offer_nbc,
        "r_out_n": r_out_n,
        "Q_init": np.ones(C) * 50.0,
        "r_init": np.full((N, B, C), 0.02),
    }


def run_test():
    data = _make_toy_data(N=8, B=30, C=3, K=25, seed=123)

    rng = np.random.default_rng(123)
    params = {
        "alpha0": 5.0,
        "alpha1": 0.10,
        "nu_home": 1.5,
        "gamma": 0.01,
        "delta_b": rng.normal(0.0, 0.35, size=int(data["B"])),
        "RATE_SCALE": 100.0,
    }

    res = solve_equilibrium(params, data, settings={"verbose_outer": True})
    print("\nTest done.")
    print("C =", data["C"], "| Q_c =", res["Q_c"], "| avg r =", np.mean(res["r_nbc"][data["offer_nbc"]]))


if __name__ == "__main__":
    run_test()
