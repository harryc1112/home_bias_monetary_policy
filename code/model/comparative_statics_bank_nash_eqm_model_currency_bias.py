#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparative statics for the "European banking integration" currency-depth model.

Works with:
    solve_bank_joint_nash_equilibrium_currency_bias.py

Expected solver signature:
    solve_equilibrium(params: dict, data: dict, settings: dict) -> dict

Saves PNGs into:
    ../../output/currency_depth_comp_statics
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Import solver
# ---------------------------------------------------------------------
try:
    import solve_bank_joint_nash_equilibrium_currency_bias as new_model
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Could not import solve_bank_joint_nash_equilibrium_currency_bias.\n"
    ) from e

# ---------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (SCRIPT_DIR / ".." / ".." / "output" / "currency_depth_comp_statics").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Utilities
# =============================================================================
def masked_mean(x: np.ndarray, mask: np.ndarray) -> float:
    x = np.asarray(x)
    mask = np.asarray(mask, dtype=bool)
    vals = x[mask]
    return float(np.nan) if vals.size == 0 else float(np.mean(vals))


def safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return a / np.maximum(b, eps)


def _plot_series(
    x: np.ndarray,
    ys: list[np.ndarray],
    *,
    labels: list[str] | None,
    xlabel: str,
    ylabel: str,
    title: str,
    outpath: Path,
) -> None:
    plt.figure()
    for i, y in enumerate(ys):
        if labels is None:
            plt.plot(x, y)
        else:
            plt.plot(x, y, label=labels[i])
    if labels is not None:
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# =============================================================================
# Synthetic data (Option 1 outside option)
# =============================================================================
def make_synthetic_data(
    *,
    N: int = 3,
    B: int = 6,
    C: int = 3,
    K: int = 50,
    seed: int = 123,
    offer_prob: float = 1.0,
) -> dict:
    rng = np.random.default_rng(seed)

    home_c_of_n = rng.integers(0, C, size=N)
    bank_home_c = rng.integers(0, C, size=B)

    spot_home_c = np.exp(rng.normal(0.0, 0.25, size=(N, C)))
    for n in range(N):
        spot_home_c[n, int(home_c_of_n[n])] = 1.0

    fwd_home_c = spot_home_c.copy()

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

    # bank investment return (in bank domestic currency)
    rL_b = 0.05 + 0.03 * rng.random(B)

    # marginal cost in each (n,b,c)
    m_nbc = 0.0015 + 0.0015 * rng.random((N, B, C))

    # outside option: one per country n
    r_out_n = np.zeros(N, dtype=float)
    r_out_nc = np.zeros((N, C), dtype=float)

    offer_nbc = rng.random((N, B, C)) < float(offer_prob)
    for n in range(N):
        for c in range(C):
            if not offer_nbc[n, :, c].any():
                offer_nbc[n, rng.integers(0, B), c] = True

    return {
        "N": N,
        "B": B,
        "C": C,
        "home_c_of_n": home_c_of_n,
        "bank_home_c": bank_home_c,
        "spot_home_c": spot_home_c,
        "fwd_home_c": fwd_home_c,
        "spot_bank_c": spot_bank_c,
        "W_grid_nk": W_grid_nk,
        "W_wgt_nk": W_wgt_nk,
        "rL_b": rL_b,
        "m_nbc": m_nbc,
        "offer_nbc": offer_nbc,
        "r_out_n": r_out_n,
        "r_out_nc": r_out_nc,
    }


# =============================================================================
# Baseline parameters
# =============================================================================
def make_base_params(B: int, *, seed: int = 123) -> dict:
    rng = np.random.default_rng(seed)
    delta_b = rng.normal(0.0, 0.35, size=B)
    return {
        "alpha0": 1.0,
        "alpha1": 0.05,
        "nu_home": 1.5,
        "gamma": 0.01,
        "delta_b": delta_b,
    }


# =============================================================================
# Solver wrapper
# =============================================================================
def solve_equilibrium_wrapper(
    params: dict,
    data: dict,
    *,
    max_outer: int = 60,
    tol_Q: float = 1e-6,
) -> dict:
    settings = {"max_outer": int(max_outer), "tol_Q": float(tol_Q)}
    return new_model.solve_equilibrium(params, data, settings=settings)


# =============================================================================
# Summaries (key diagnostic part)
# =============================================================================
def summarise_equilibrium(res: dict, data: dict) -> dict:
    """
    Returns:
      - Q_c padded to length C
      - Q shares
      - avg rate (unweighted over offers)
      - avg rate (q-weighted over offers, in physical units)
      - avg home-currency share (physical units)
      - avg rate by currency (unweighted + q-weighted)
      - rate dispersion diagnostics
    """
    out = {}
    C = int(data["C"])

    # ----- Q_c (pad if solver returns fewer currencies)
    Q_c_raw = np.asarray(res.get("Q_c", np.zeros(C)), dtype=float).reshape(-1)
    if Q_c_raw.size < C:
        Q_c = np.zeros(C, dtype=float)
        Q_c[: Q_c_raw.size] = Q_c_raw
    else:
        Q_c = Q_c_raw[:C]
    out["Q_c"] = Q_c
    out["Q_sum"] = float(np.sum(Q_c))
    out["Q_max"] = float(np.max(Q_c)) if C > 0 else float("nan")
    out["Q_share_c"] = Q_c / np.maximum(np.sum(Q_c), 1e-12)

    # ----- Rates & offers
    r_nbc = np.asarray(res.get("r_nbc", np.nan))
    offer = np.asarray(data.get("offer_nbc", np.ones_like(r_nbc, dtype=bool)), dtype=bool)

    if r_nbc.ndim != 3 or offer.shape != r_nbc.shape:
        out["avg_r_offered"] = float("nan")
        out["avg_r_offered_qw"] = float("nan")
        out["std_r_offered"] = float("nan")
        return out

    N, B, C_data = r_nbc.shape
    if C_data != C:
        # still proceed; just be careful with currency loops
        C_eff = min(C, C_data)
    else:
        C_eff = C

    out["avg_r_offered"] = masked_mean(r_nbc, offer)
    out["std_r_offered"] = float(np.std(r_nbc[offer])) if np.any(offer) else float("nan")

    # ----- q-weights from q_nbc_home converted to physical units
    q_home = res.get("q_nbc_home", None)
    if q_home is None:
        out["avg_r_offered_qw"] = float("nan")
        out["avg_r_by_c"] = np.full(C_eff, np.nan)
        out["avg_r_by_c_qw"] = np.full(C_eff, np.nan)
    else:
        q_home = np.asarray(q_home, dtype=float)
        spot_home_c = np.asarray(data["spot_home_c"], dtype=float)

        # Convert q from home-currency units into physical currency units: divide by spot_home_c[n,c]
        q_phys = np.zeros_like(q_home)
        for n in range(N):
            denom = np.maximum(spot_home_c[n, :C_eff], 1e-12)
            q_phys[n, :, :C_eff] = q_home[n, :, :C_eff] / denom

        w = np.zeros_like(r_nbc, dtype=float)
        w[:, :, :C_eff] = q_phys[:, :, :C_eff]
        w = np.where(offer, w, 0.0)

        total_w = float(np.sum(w))
        if total_w > 0:
            out["avg_r_offered_qw"] = float(np.sum(r_nbc * w) / total_w)
        else:
            out["avg_r_offered_qw"] = float("nan")

        # currency-specific averages
        avg_r_by_c = np.full(C_eff, np.nan)
        avg_r_by_c_qw = np.full(C_eff, np.nan)

        for c in range(C_eff):
            mask_c = offer[:, :, c]
            avg_r_by_c[c] = masked_mean(r_nbc[:, :, c], mask_c)

            wc = w[:, :, c]
            tot_wc = float(np.sum(wc))
            if tot_wc > 0:
                avg_r_by_c_qw[c] = float(np.sum(r_nbc[:, :, c] * wc) / tot_wc)

        out["avg_r_by_c"] = avg_r_by_c
        out["avg_r_by_c_qw"] = avg_r_by_c_qw

    # ----- Home-currency share (physical units), averaged over countries
    if q_home is not None:
        home_c_of_n = np.asarray(data["home_c_of_n"], dtype=int)
        # aggregate physical by (n,c)
        q_phys_nc = np.zeros((N, C_eff), dtype=float)
        for n in range(N):
            denom = np.maximum(spot_home_c[n, :C_eff], 1e-12)
            q_phys_nc[n, :C_eff] = np.sum(q_home[n, :, :C_eff], axis=0) / denom

        total_phys_n = np.sum(q_phys_nc, axis=1)
        home_share_n = np.zeros(N, dtype=float)
        for n in range(N):
            hc = int(home_c_of_n[n])
            if hc < C_eff:
                home_share_n[n] = q_phys_nc[n, hc] / np.maximum(total_phys_n[n], 1e-12)
            else:
                home_share_n[n] = np.nan

        out["home_currency_share_avg"] = float(np.nanmean(home_share_n))
    else:
        out["home_currency_share_avg"] = float("nan")

    return out


# =============================================================================
# Comparative statics
# =============================================================================
def comp_static_nu_home(
    data: dict,
    params_base: dict,
    *,
    N_grid: int = 20,
    low: float = 0.0,
    high: float = 3.0,
    max_outer: int = 60,
    tol_Q: float = 1e-6,
):
    xs = np.linspace(low, high, N_grid)

    avg_r = np.full(N_grid, np.nan)
    avg_r_qw = np.full(N_grid, np.nan)
    Q_max = np.full(N_grid, np.nan)
    home_share = np.full(N_grid, np.nan)
    std_r = np.full(N_grid, np.nan)

    C = int(data["C"])
    Qshare = np.full((N_grid, C), np.nan)

    for i, nu in enumerate(xs):
        params = dict(params_base)
        params["nu_home"] = float(nu)

        res = solve_equilibrium_wrapper(params, data, max_outer=max_outer, tol_Q=tol_Q)
        summ = summarise_equilibrium(res, data)

        avg_r[i] = summ["avg_r_offered"]
        avg_r_qw[i] = summ.get("avg_r_offered_qw", np.nan)
        std_r[i] = summ.get("std_r_offered", np.nan)
        Q_max[i] = summ["Q_max"]
        home_share[i] = summ.get("home_currency_share_avg", np.nan)

        qsh = summ.get("Q_share_c", None)
        if isinstance(qsh, np.ndarray) and qsh.ndim == 1 and qsh.size == C:
            Qshare[i, :] = qsh

        print(
            f"nu_home={nu:.3f} | avg_r={avg_r[i]:.8f} | avg_r_qw={avg_r_qw[i]:.8f} "
            f"| std_r={std_r[i]:.8f} | Q_max={Q_max[i]:.3f} | home_share={home_share[i]:.3f}"
        )

    _plot_series(
        xs, [avg_r, avg_r_qw],
        labels=["Avg offered rate (unweighted)", "Avg offered rate (q-weighted)"],
        xlabel="nu_home",
        ylabel="Avg rate",
        title="nu_home and equilibrium rates",
        outpath=OUTPUT_DIR / "cs_nu_home_avg_rate.png",
    )
    _plot_series(
        xs, [std_r],
        labels=["Std dev of offered rates (over offers)"],
        xlabel="nu_home",
        ylabel="Std dev",
        title="nu_home and rate dispersion",
        outpath=OUTPUT_DIR / "cs_nu_home_rate_dispersion.png",
    )
    _plot_series(
        xs, [Q_max],
        labels=["max(Q_c)"],
        xlabel="nu_home",
        ylabel="max currency depth",
        title="nu_home and currency depth (max Q)",
        outpath=OUTPUT_DIR / "cs_nu_home_Qmax.png",
    )
    _plot_series(
        xs, [home_share],
        labels=["Avg home-currency share (physical units)"],
        xlabel="nu_home",
        ylabel="share",
        title="nu_home and home-currency share",
        outpath=OUTPUT_DIR / "cs_nu_home_home_share.png",
    )

    if C <= 6:
        labels = [f"Q share currency {c}" for c in range(C)]
        _plot_series(
            xs, [Qshare[:, c] for c in range(C)],
            labels=labels,
            xlabel="nu_home",
            ylabel="share of global depth",
            title="nu_home and currency depth shares",
            outpath=OUTPUT_DIR / "cs_nu_home_Qshares.png",
        )

    return xs, avg_r, avg_r_qw, Q_max, home_share, Qshare


def comp_static_gamma(
    data: dict,
    params_base: dict,
    *,
    N_grid: int = 20,
    low: float = 0.0,
    high: float = 0.05,
    max_outer: int = 60,
    tol_Q: float = 1e-6,
):
    xs = np.linspace(low, high, N_grid)

    avg_r = np.full(N_grid, np.nan)
    avg_r_qw = np.full(N_grid, np.nan)
    std_r = np.full(N_grid, np.nan)
    Q_sum = np.full(N_grid, np.nan)
    Q_max = np.full(N_grid, np.nan)

    C = int(data["C"])
    Qshare = np.full((N_grid, C), np.nan)

    for i, g in enumerate(xs):
        params = dict(params_base)
        params["gamma"] = float(g)

        res = solve_equilibrium_wrapper(params, data, max_outer=max_outer, tol_Q=tol_Q)
        summ = summarise_equilibrium(res, data)

        avg_r[i] = summ["avg_r_offered"]
        avg_r_qw[i] = summ.get("avg_r_offered_qw", np.nan)
        std_r[i] = summ.get("std_r_offered", np.nan)
        Q_sum[i] = summ["Q_sum"]
        Q_max[i] = summ["Q_max"]

        qsh = summ.get("Q_share_c", None)
        if isinstance(qsh, np.ndarray) and qsh.ndim == 1 and qsh.size == C:
            Qshare[i, :] = qsh

        print(
            f"gamma={g:.5f} | avg_r={avg_r[i]:.8f} | avg_r_qw={avg_r_qw[i]:.8f} "
            f"| std_r={std_r[i]:.8f} | Q_sum={Q_sum[i]:.3f} | Q_max={Q_max[i]:.3f}"
        )

    _plot_series(
        xs, [avg_r, avg_r_qw],
        labels=["Avg offered rate (unweighted)", "Avg offered rate (q-weighted)"],
        xlabel="gamma",
        ylabel="Avg rate",
        title="gamma and equilibrium rates",
        outpath=OUTPUT_DIR / "cs_gamma_avg_rate.png",
    )
    _plot_series(
        xs, [std_r],
        labels=["Std dev of offered rates (over offers)"],
        xlabel="gamma",
        ylabel="Std dev",
        title="gamma and rate dispersion",
        outpath=OUTPUT_DIR / "cs_gamma_rate_dispersion.png",
    )
    _plot_series(
        xs, [Q_sum, Q_max],
        labels=["sum(Q_c)", "max(Q_c)"],
        xlabel="gamma",
        ylabel="depth",
        title="gamma and currency depth levels",
        outpath=OUTPUT_DIR / "cs_gamma_Qlevels.png",
    )

    if C <= 6:
        labels = [f"Q share currency {c}" for c in range(C)]
        _plot_series(
            xs, [Qshare[:, c] for c in range(C)],
            labels=labels,
            xlabel="gamma",
            ylabel="share of global depth",
            title="gamma and currency depth shares",
            outpath=OUTPUT_DIR / "cs_gamma_Qshares.png",
        )

    return xs, avg_r, avg_r_qw, Q_sum, Q_max, Qshare


def comp_static_alpha1(
    data: dict,
    params_base: dict,
    *,
    N_grid: int = 20,
    low: float = 0.0,
    high: float = 0.25,
    max_outer: int = 60,
    tol_Q: float = 1e-6,
):
    xs = np.linspace(low, high, N_grid)

    avg_r = np.full(N_grid, np.nan)
    avg_r_qw = np.full(N_grid, np.nan)
    std_r = np.full(N_grid, np.nan)
    Q_max = np.full(N_grid, np.nan)

    for i, a1 in enumerate(xs):
        params = dict(params_base)
        params["alpha1"] = float(a1)

        res = solve_equilibrium_wrapper(params, data, max_outer=max_outer, tol_Q=tol_Q)
        summ = summarise_equilibrium(res, data)

        avg_r[i] = summ["avg_r_offered"]
        avg_r_qw[i] = summ.get("avg_r_offered_qw", np.nan)
        std_r[i] = summ.get("std_r_offered", np.nan)
        Q_max[i] = summ["Q_max"]

        print(
            f"alpha1={a1:.4f} | avg_r={avg_r[i]:.8f} | avg_r_qw={avg_r_qw[i]:.8f} "
            f"| std_r={std_r[i]:.8f} | Q_max={Q_max[i]:.3f}"
        )

    _plot_series(
        xs, [avg_r, avg_r_qw],
        labels=["Avg offered rate (unweighted)", "Avg offered rate (q-weighted)"],
        xlabel="alpha1 (wealth-dependent slope)",
        ylabel="Avg rate",
        title="alpha1 and equilibrium rates",
        outpath=OUTPUT_DIR / "cs_alpha1_avg_rate.png",
    )
    _plot_series(
        xs, [std_r],
        labels=["Std dev of offered rates (over offers)"],
        xlabel="alpha1 (wealth-dependent slope)",
        ylabel="Std dev",
        title="alpha1 and rate dispersion",
        outpath=OUTPUT_DIR / "cs_alpha1_rate_dispersion.png",
    )
    _plot_series(
        xs, [Q_max],
        labels=["max(Q_c)"],
        xlabel="alpha1 (wealth-dependent slope)",
        ylabel="max currency depth",
        title="alpha1 and currency depth (max Q)",
        outpath=OUTPUT_DIR / "cs_alpha1_Qmax.png",
    )

    return xs, avg_r, avg_r_qw, Q_max


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    data = make_synthetic_data(N=8, B=30, C=3, K=25, seed=123, offer_prob=0.85)
    params_base = make_base_params(B=int(data["B"]), seed=123)

    print("\nBaseline solve...")
    res0 = solve_equilibrium_wrapper(params_base, data, max_outer=60, tol_Q=1e-6)
    s0 = summarise_equilibrium(res0, data)
    print("Baseline summary:")
    print("  C =", data["C"])
    print("  avg_r_offered (unweighted):", s0.get("avg_r_offered"))
    print("  avg_r_offered (q-weighted):", s0.get("avg_r_offered_qw"))
    print("  std_r_offered:", s0.get("std_r_offered"))
    print("  Q_c:", s0.get("Q_c"))
    print("  Q_share_c:", s0.get("Q_share_c"))
    print("  avg home-currency share:", s0.get("home_currency_share_avg"))

    print("\nRunning comp statics: nu_home ...")
    comp_static_nu_home(data, params_base, N_grid=20, low=0.0, high=3.0, max_outer=60, tol_Q=1e-6)

    print("\nRunning comp statics: gamma ...")
    comp_static_gamma(data, params_base, N_grid=20, low=0.0, high=0.05, max_outer=60, tol_Q=1e-6)

    print("\nRunning comp statics: alpha1 ...")
    comp_static_alpha1(data, params_base, N_grid=20, low=0.0, high=0.25, max_outer=60, tol_Q=1e-6)

    print(f"\nSaved figures to:\n  {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
