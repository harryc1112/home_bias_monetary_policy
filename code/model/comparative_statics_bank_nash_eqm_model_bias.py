#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------------------------------------------------
# Import solver module
# ---------------------------------------------------------------------
try:
    import solve_bank_joint_nash_equilibrium_bias as bank_nash_eqm
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Could not import solve_bank_joint_nash_equilibrium_bias_new.\n"
        "Make sure the solver file is named exactly 'solve_bank_joint_nash_equilibrium_bias_new.py' "
        "and is on your Python path."
    ) from e

# ---------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
output_root = os.path.abspath(os.path.join(script_dir, "..", "..", "output"))
comp_stats_dir = os.path.join(output_root, "nash_model_comp_statics")
os.makedirs(comp_stats_dir, exist_ok=True)

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

# =============================================================================
def get_sim_data(params, J=35, M=18, B_L=200, B_D=200, seed=123):
    return bank_nash_eqm.simulate_inputs(params, J=J, M=M, B_L=B_L, B_D=B_D, seed=seed)


def masked_mean(x, mask):
    x = np.asarray(x)
    mask = np.asarray(mask, dtype=bool)
    vals = x[mask]
    return float(np.nan) if vals.size == 0 else float(np.mean(vals))


def masked_var(x, mask):
    x = np.asarray(x)
    mask = np.asarray(mask, dtype=bool)
    vals = x[mask]
    return float(np.nan) if vals.size == 0 else float(np.var(vals))


def _replace_E(primitives, E_new):
    """
    primitives layout:
      (xiF, xiD, partL, partD, home, Ld, Lw, Dd, Dw, sizeL, sizeD, E)
    """
    prim = list(primitives)
    prim[11] = np.asarray(E_new, dtype=float)
    return tuple(prim)


def solve_from_primitives(
    params,
    primitives,
    *,
    max_iter=250,
    tol=1e-6,
    tol_foc=1e-6,
    rL_min=0.5,
    rL_max=5.0,
    rD_min=-1.0,
    rD_max=5.0,
    damp_fp=0.5,
    init_rL=None,
    init_rD=None,
    verbose=False,
):
    (xiF, xiD, partL, partD, home,
     Ld, Lw, Dd, Dw,
     sizeL, sizeD, E, mcL, mcD) = primitives

    out = bank_nash_eqm.solve_joint_eqm(
        params,
        xiF, xiD,
        partL, partD, home,
        Ld, Lw,
        Dd, Dw,
        sizeL, sizeD,
        E, mcL, mcD, 
        rL_min=rL_min, rL_max=rL_max,
        rD_min=rD_min, rD_max=rD_max,
        max_iter=max_iter,
        tol=tol,
        tol_foc=tol_foc,
        damp_fp=damp_fp,
        init_rL=init_rL,
        init_rD=init_rD,
        verbose=verbose,
    )
    return out


def solve_with_retries(
    params,
    primitives,
    *,
    init_rL=None,
    init_rD=None,
    rL_min=0.5,
    rL_max=5.0,
    rD_min=-1.0,
    rD_max=5.0,
    verbose=False,
):
    """
    Small schedule of solver settings. Returns (out, used_settings_dict).
    """
    schedule = [
        dict(max_iter=250, damp_fp=0.60, tol=1e-6, tol_foc=1e-6),
        dict(max_iter=400, damp_fp=0.50, tol=1e-6, tol_foc=1e-6),
        dict(max_iter=700, damp_fp=0.35, tol=1e-6, tol_foc=1e-6),
    ]

    last_out = None
    used = schedule[-1]

    cur_rL = init_rL
    cur_rD = init_rD

    for s in schedule:
        used = s
        out = solve_from_primitives(
            params, primitives,
            max_iter=s["max_iter"],
            tol=s["tol"],
            tol_foc=s["tol_foc"],
            damp_fp=s["damp_fp"],
            rL_min=rL_min, rL_max=rL_max,
            rD_min=rD_min, rD_max=rD_max,
            init_rL=cur_rL,
            init_rD=cur_rD,
            verbose=verbose,
        )
        last_out = out

        if bool(out.get("info", {}).get("converged", False)):
            return out, used

        cur_rL = out.get("rL_JM", cur_rL)
        cur_rD = out.get("rD_JM", cur_rD)

    return last_out, used


def _plot_converged_series(x, series_list, converged, *, xlabel, ylabel, title, path, labels=None, extras=None):
    x = np.asarray(x)
    conv = np.asarray(converged, dtype=bool)

    plt.figure()
    if np.any(conv):
        for y in series_list:
            y = np.asarray(y)
            plt.plot(x[conv], y[conv])
        if extras is not None:
            for ex in extras:
                if len(ex) == 4:
                    plt.plot(ex[0], ex[1], ex[2], label=ex[3])
                elif len(ex) == 3:
                    plt.plot(ex[0], ex[1], ex[2])
                else:
                    plt.plot(ex[0], ex[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if labels is not None:
        plt.legend(labels)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# =============================================================================
# Baseline params
# =============================================================================
def make_base_params():
    if hasattr(bank_nash_eqm, "create_params_for_smoke_test"):
        p = bank_nash_eqm.create_params_for_smoke_test()
    else:
        p = {
            "alpha_F": 35.0,
            "alpha_D": 20.0,
            "beta_w": 0.0001,
            "gammaF": 0.4,
            "beta_c": -0.0001,
            "gammaD": 0.2,
            "rF": 1.02,
            "phi": 0.005,
            "lambda": 0.3,
            "r_nonbank": 1.05,
        }
    p.setdefault("equity_share", 0.17) 
    return p


# =============================================================================
# 1) Comparative Static: net risk-free rate r  (maps to rF = 1 + r)
# =============================================================================
def comp_static_r(N_r=20, r_low=0.005, r_high=0.075, *, J=35, M=15, seed=42):
    print(f"\nRunning Comparative Static on net r (N={N_r})...")
    r_net_grid = np.linspace(r_low, r_high, N_r)

    avg_rL = np.full(N_r, np.nan)
    avg_rD = np.full(N_r, np.nan)
    var_rL = np.full(N_r, np.nan)

    converged = np.zeros(N_r, dtype=bool)

    params_base = make_base_params()
    params_base["rF"] = 1.0 + float(r_net_grid[0])

    # keep nonbank spread constant
    base_spread = float(params_base.get("r_nonbank", params_base["rF"]) - params_base["rF"])

    primitives = get_sim_data(params_base, J=J, M=M, B_L=200, B_D=200, seed=seed)
    (_, _, partL, partD, _, *_) = primitives
    maskL = partL.astype(bool)
    maskD = partD.astype(bool)

    last_good_rL = None
    last_good_rD = None

    for idx, r_net in enumerate(r_net_grid):
        params = dict(params_base)
        params["rF"] = 1.0 + float(r_net)
        params["r_nonbank"] = float(params["rF"] + base_spread)

        out, used = solve_with_retries(
            params, primitives,
            init_rL=last_good_rL, init_rD=last_good_rD,
            rL_min=0.5, rL_max=5.0, rD_min=-1.0, rD_max=5.0,
            verbose=False,
        )

        info = out.get("info", {})
        converged[idx] = bool(info.get("converged", False))

        if converged[idx]:
            last_good_rL = out["rL_JM"]
            last_good_rD = out["rD_JM"]

            avg_rL[idx] = masked_mean(out["rL_JM"], maskL)
            avg_rD[idx] = masked_mean(out["rD_JM"], maskD)
            var_rL[idx] = masked_var(out["rL_JM"], maskL)

        print(
            f"r(net)={r_net:.3f} | rF={params['rF']:.3f} | conv={converged[idx]} "
            f"| avg rL={avg_rL[idx]:.4f} | avg rD={avg_rD[idx]:.4f} | (damp={used['damp_fp']})"
        )

    conv = np.asarray(converged, bool)
    _plot_converged_series(
        r_net_grid,
        [avg_rL, avg_rD],
        converged,
        xlabel="net risk-free rate r",
        ylabel="gross rates",
        title="Pass-through of r to equilibrium bank rates",
        path=os.path.join(comp_stats_dir, f"{RUN_TAG}_comp_static_r_levels.png"),
        labels=["Avg loan rate (active j,m)", "Avg deposit rate (active j,m)", "Risk-free gross (1+r)"],
        extras=[(r_net_grid[conv], 1.0 + r_net_grid[conv], "k--", "Risk-free gross (1+r)")] if np.any(conv) else None
    )

    _plot_converged_series(
        r_net_grid,
        [np.sqrt(var_rL)],
        converged,
        xlabel="net risk-free rate r",
        ylabel="Dispersion σ(r_L) across active (j,m)",
        title="Loan-rate dispersion vs level of r",
        path=os.path.join(comp_stats_dir, f"{RUN_TAG}_comp_static_r_dispersion.png"),
    )

    return r_net_grid, avg_rL, avg_rD, var_rL, converged


# =============================================================================
# 2) Comparative Static: equity_share (implemented as scaling E_J)
# =============================================================================
def comp_static_equity_share(N_e=20, e_low=0.01, e_high=0.35, *, J=35, M=15, seed=101):
    print(f"\nRunning Comparative Static on equity_share (via scaling E_J) (N={N_e})...")
    es = np.linspace(e_low, e_high, N_e)

    avg_rL = np.full(N_e, np.nan)
    avg_spread = np.full(N_e, np.nan)
    avg_I_D = np.full(N_e, np.nan)

    converged = np.zeros(N_e, dtype=bool)

    # ---- FIX: baseline label must NOT be es[0]
    params_base = make_base_params()
    params_base["equity_share"] = float(params_base.get("equity_share", 0.17))  # stable baseline label

    primitives0 = get_sim_data(params_base, J=J, M=M, B_L=200, B_D=200, seed=seed)
    (_, _, partL, partD, _, *_) = primitives0
    maskL = partL.astype(bool)
    maskD = partD.astype(bool)

    E0 = np.asarray(primitives0[11], dtype=float)
    e_base = float(params_base.get("equity_share", 0.17))
    if e_base <= 0:
        e_base = 0.17

    last_good_rL = None
    last_good_rD = None

    for idx, e_val in enumerate(es):
        params = dict(params_base)
        params["equity_share"] = float(e_val)

        scale = float(e_val) / e_base
        E_new = E0 * scale
        primitives = _replace_E(primitives0, E_new)

        out, used = solve_with_retries(
            params, primitives,
            init_rL=last_good_rL, init_rD=last_good_rD,
            rL_min=0.5, rL_max=5.0, rD_min=-1.0, rD_max=5.0,
            verbose=False,
        )

        info = out.get("info", {})
        converged[idx] = bool(info.get("converged", False))

        if converged[idx]:
            last_good_rL = out["rL_JM"]
            last_good_rD = out["rD_JM"]

            rL = out["rL_JM"]
            rD = out["rD_JM"]
            avg_rL[idx] = masked_mean(rL, maskL)
            avg_spread[idx] = masked_mean(rL, maskL) - masked_mean(rD, maskD)

            D_safe = np.maximum(out["D_j"], 1e-12)
            avg_I_D[idx] = float(np.mean(out["I_j"] / D_safe))

        print(
            f"equity_share={e_val:.3f} (scale={scale:.3f}) | conv={converged[idx]} "
            f"| avg spread={avg_spread[idx]:.4f} | avg I/D={avg_I_D[idx]:.4f} | (damp={used['damp_fp']})"
        )

    _plot_converged_series(
        es, [avg_I_D], converged,
        xlabel="equity_share (implemented via scaling E_J)",
        ylabel="Avg I/D",
        title="Liquidity ratio vs equity_share (E_J scaling)",
        path=os.path.join(comp_stats_dir, f"{RUN_TAG}_comp_static_equity_compliance.png"),
        labels=["Avg I/D"],
    )

    _plot_converged_series(
        es, [avg_spread], converged,
        xlabel="equity_share (implemented via scaling E_J)",
        ylabel="Avg spread (r_L - r_D) over active markets",
        title="Pricing wedge vs equity_share (E_J scaling)",
        path=os.path.join(comp_stats_dir, f"{RUN_TAG}_comp_static_equity_spread.png"),
    )

    return es, avg_rL, avg_spread, avg_I_D, converged


# =============================================================================
# 3) Comparative Static: lambda (liquidity target)
# =============================================================================
def comp_static_lambda(N_lam=20, l_low=0.01, l_high=0.7, *, J=35, M=15, seed=99):
    print(f"\nRunning Comparative Static on lambda (N={N_lam})...")
    lams = np.linspace(l_low, l_high, N_lam)

    avg_I_D = np.full(N_lam, np.nan)
    total_L = np.full(N_lam, np.nan)

    converged = np.zeros(N_lam, dtype=bool)

    params_base = make_base_params()
    params_base["lambda"] = float(lams[0])

    primitives = get_sim_data(params_base, J=J, M=M, B_L=200, B_D=200, seed=seed)

    last_good_rL = None
    last_good_rD = None

    for idx, lam in enumerate(lams):
        params = dict(params_base)
        params["lambda"] = float(lam)

        out, used = solve_with_retries(
            params, primitives,
            init_rL=last_good_rL, init_rD=last_good_rD,
            rL_min=0.5, rL_max=5.0, rD_min=-1.0, rD_max=5.0,
            verbose=False,
        )

        info = out.get("info", {})
        converged[idx] = bool(info.get("converged", False))

        if converged[idx]:
            last_good_rL = out["rL_JM"]
            last_good_rD = out["rD_JM"]

            D_safe = np.maximum(out["D_j"], 1e-12)
            avg_I_D[idx] = float(np.mean(out["I_j"] / D_safe))
            total_L[idx] = float(np.sum(out["L_j"]))

        print(
            f"lambda={lam:.3f} | conv={converged[idx]} "
            f"| avg I/D={avg_I_D[idx]:.4f} | total L={total_L[idx]:.2f} | (damp={used['damp_fp']})"
        )

    conv = np.asarray(converged, bool)
    _plot_converged_series(
        lams, [avg_I_D], converged,
        xlabel="lambda (liquidity target)",
        ylabel="Avg I/D",
        title="Compliance with liquidity target",
        path=os.path.join(comp_stats_dir, f"{RUN_TAG}_comp_static_lambda_compliance.png"),
        labels=["Actual avg I/D", "45-degree"],
        extras=[(lams[conv], lams[conv], "k--", "45-degree")] if np.any(conv) else None
    )

    _plot_converged_series(
        lams, [total_L], converged,
        xlabel="lambda (liquidity target)",
        ylabel="Total lending volume (sum_j L_j)",
        title="Credit volume vs liquidity target",
        path=os.path.join(comp_stats_dir, f"{RUN_TAG}_comp_static_lambda_credit.png"),
    )

    return lams, avg_I_D, total_L, converged


# =============================================================================
# 4) Comparative Static: home-bias on loans (gammaF)
# =============================================================================
def comp_static_home_bias_gammaF(N_g=20, g_low=0.0, g_high=1.5, *, J=40, M=20, seed=123):
    print(f"\nRunning Comparative Static on loan home-bias gammaF (N={N_g})...")
    gammas = np.linspace(g_low, g_high, N_g)

    avg_rL_home = np.full(N_g, np.nan)
    avg_rL_foreign = np.full(N_g, np.nan)
    gap = np.full(N_g, np.nan)

    converged = np.zeros(N_g, dtype=bool)

    params_base = make_base_params()
    params_base["gammaF"] = float(gammas[0])

    primitives = get_sim_data(params_base, J=J, M=M, B_L=200, B_D=200, seed=seed)
    (_, _, partL, _, home, *_) = primitives
    mask_home = home.astype(bool) & partL.astype(bool)
    mask_foreign = (~home.astype(bool)) & partL.astype(bool)

    last_good_rL = None
    last_good_rD = None

    for idx, g in enumerate(gammas):
        params = dict(params_base)
        params["gammaF"] = float(g)

        out, used = solve_with_retries(
            params, primitives,
            init_rL=last_good_rL, init_rD=last_good_rD,
            rL_min=0.5, rL_max=5.0, rD_min=-1.0, rD_max=5.0,
            verbose=False,
        )

        info = out.get("info", {})
        converged[idx] = bool(info.get("converged", False))

        if converged[idx]:
            last_good_rL = out["rL_JM"]
            last_good_rD = out["rD_JM"]

            rL = out["rL_JM"]
            r_dom = masked_mean(rL, mask_home)
            r_for = masked_mean(rL, mask_foreign)
            avg_rL_home[idx] = r_dom
            avg_rL_foreign[idx] = r_for
            gap[idx] = r_dom - r_for

        print(
            f"gammaF={g:.2f} | conv={converged[idx]} "
            f"| home rL={avg_rL_home[idx]:.4f} | foreign rL={avg_rL_foreign[idx]:.4f} | (damp={used['damp_fp']})"
        )

    _plot_converged_series(
        gammas, [avg_rL_home, avg_rL_foreign], converged,
        xlabel="loan home-bias gammaF",
        ylabel="Equilibrium loan rate r_L (gross)",
        title="Effect of home bias on loan pricing",
        path=os.path.join(comp_stats_dir, f"{RUN_TAG}_comp_static_bias_rates.png"),
        labels=["Domestic rate (home)", "Cross-border rate (foreign)"],
    )

    _plot_converged_series(
        gammas, [gap], converged,
        xlabel="loan home-bias gammaF",
        ylabel="Spread (r_home - r_foreign)",
        title="Segmentation wedge from home bias",
        path=os.path.join(comp_stats_dir, f"{RUN_TAG}_comp_static_bias_wedge.png"),
    )

    return gammas, avg_rL_home, avg_rL_foreign, gap, converged


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # Comment out any block you don’t want to run
    comp_static_r()
    comp_static_equity_share()
    comp_static_lambda()
    comp_static_home_bias_gammaF()

    print(f"\nPNG figures saved to:\n  {comp_stats_dir}\nRun tag:\n  {RUN_TAG}\n")
