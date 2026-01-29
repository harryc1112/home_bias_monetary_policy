# =========================================================
# Latvia — FX-hedged EUR->LVL benchmark + "home premium net of FX"
# Two-plot layout (Wu/Du-style): (1) Real vs Synthetic, (2) Wedge Δ
#
# Robust Table21 selection by Excel column letter (NO rate-filling).
# Includes sanity checks + removes pandas "highly fragmented" warning.
# Includes tenor-robustness check: 3M vs 6M vs 1Y (optional overlays + tables).
#
# Default mapping (change if you want):
#   - HH deposits: NEW business, agreed maturity, up to 1Y  -> col "D"
#   - NFC loans:   OUTSTANDING amounts, maturity up to 1Y   -> col "AW"
#
# Inputs:
#   - latvian_forwards.xlsx
#   - table21_rates_deposits_{lats,euro}.csv
#   - table21_rates_loans_{lats,euro}.csv
#
# Outputs (into ~/Dropbox/home_bias_monetary_policy/output/motivating_facts/):
#   Baseline (single tenor):
#     - dep_hh_upto1y_real_vs_synth_{TENOR}.png
#     - dep_hh_upto1y_delta_{TENOR}.png
#     - loan_nfc_upto1y_real_vs_synth_{TENOR}.png
#     - loan_nfc_upto1y_delta_{TENOR}.png
#     - latvia_dep_hh_upto1y_real_vs_synth_{TENOR}.csv
#     - latvia_loan_nfc_upto1y_real_vs_synth_{TENOR}.csv
#
#   Robustness (tenor overlay):
#     - hh_deposits_up_to_1y_real_vs_synth_tenor_robust.png
#     - hh_deposits_up_to_1y_delta_tenor_robust.png
#     - hh_deposits_up_to_1y_tenor_stats.csv
#     - hh_deposits_up_to_1y_tenor_pairwise.csv
#     - nfc_loans_up_to_1y_real_vs_synth_tenor_robust.png
#     - nfc_loans_up_to_1y_delta_tenor_robust.png
#     - nfc_loans_up_to_1y_tenor_stats.csv
#     - nfc_loans_up_to_1y_tenor_pairwise.csv
#
# Toggle:
#   RUN_TENOR_ROBUSTNESS = True/False
# =========================================================

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------
# 0) Paths & constants
# -------------------------------------------------
BASE_DIR = Path(os.path.expanduser("~")) / "Dropbox" / "home_bias_monetary_policy"
DATA_DIR = BASE_DIR / "data"
ALT_DATA_DIR = Path("/mnt/data")

OUTPUT_DIR = BASE_DIR / "output" / "motivating_facts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EURO_ADOPTION = pd.Timestamp("2014-01-01")

plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.size": 11,
    "grid.alpha": 0.2,
    "figure.dpi": 200
})

# -----------------------
# Choose baseline hedge tenor
# -----------------------
BASELINE_TENOR = "3M"  # you can set to "6M" or "1Y" for horizon-matching
RUN_TENOR_ROBUSTNESS = True  # set False if you only want baseline plots/CSVs

TENOR_TO_TYEARS = {
    "1W": 7.0 / 365.0,
    "1M": 1.0 / 12.0,
    "2M": 2.0 / 12.0,
    "3M": 3.0 / 12.0,
    "6M": 6.0 / 12.0,
    "9M": 9.0 / 12.0,
    "1Y": 1.0
}

ROMAN_MAP = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
    "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12
}


# -------------------------------------------------
# Helper: pick path from Dropbox or /mnt/data
# -------------------------------------------------
def pick_path(filename: str) -> Path:
    p = DATA_DIR / filename
    if p.exists():
        return p
    p2 = ALT_DATA_DIR / filename
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Could not find {filename} in {DATA_DIR} or {ALT_DATA_DIR}")


# -------------------------------------------------
# 1) Table 21 wide loader (preserves column positions)
# -------------------------------------------------
def excel_col_to_idx(col: str) -> int:
    col = col.upper().strip()
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def load_table21_wide(filepath: Path) -> pd.DataFrame:
    """
    Reads LB Table 21 exports:
      - 6 header rows, then Year / Month + wide numeric columns.
    Keeps wide structure so Excel column letters map to iloc positions.
    """
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    df = pd.read_csv(
        filepath,
        skiprows=6,
        na_values=["x", "-", "–", " ", "..", "…"],
        low_memory=False
    )
    df.columns.values[0] = "Year"
    df.columns.values[1] = "Month"

    # Forward-fill Year only (NO filling of rates)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").ffill()

    # Roman months -> integers
    df["Month"] = df["Month"].astype(str).str.strip().map(ROMAN_MAP)

    df = df.dropna(subset=["Year", "Month"]).copy()

    # Convert wide data columns to numeric
    for c in df.columns[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Build date without triggering fragmentation warning
    df["date"] = pd.to_datetime(
        {"year": df["Year"].astype(int), "month": df["Month"].astype(int), "day": 1}
    )

    return df.set_index("date")


def series_by_letter(df_wide: pd.DataFrame, col_letter: str) -> pd.Series:
    j = excel_col_to_idx(col_letter)
    if j < 0 or j >= df_wide.shape[1]:
        raise IndexError(
            f"Column {col_letter} -> index {j} out of bounds for df with {df_wide.shape[1]} columns."
        )
    return df_wide.iloc[:, j]


# -------------------------------------------------
# 2) Forwards: monthly F/S ratios + sanity checks
# -------------------------------------------------
def load_monthly_forward_ratios(
    forwards_xlsx: Path,
    tenors=("1W", "1M", "2M", "3M", "6M", "9M", "1Y"),
    agg="mean"  # "mean" or "eom"
) -> pd.DataFrame:
    fwd = pd.read_excel(forwards_xlsx).copy()

    # Date column
    if "Name" in fwd.columns:
        fwd = fwd.rename(columns={"Name": "date"})
    elif "Date" in fwd.columns:
        fwd = fwd.rename(columns={"Date": "date"})
    elif "date" not in fwd.columns:
        fwd = fwd.rename(columns={fwd.columns[0]: "date"})
    fwd["date"] = pd.to_datetime(fwd["date"])

    def pick_col(patterns):
        for pat in patterns:
            cols = [c for c in fwd.columns if re.search(pat, str(c))]
            if cols:
                return cols[0]
        return None

    # Spot
    spot_col = pick_col([r"\bSPOT\b", r"\bSpot\b", r"\bON\b", r"\bS\b"])
    if spot_col is None:
        raise ValueError("Could not identify spot column (tried SPOT/Spot/ON/S).")

    # Tenors
    tenor_cols = {}
    for t in tenors:
        col = pick_col([rf"\b{re.escape(t)}\b"])
        if col is not None:
            tenor_cols[t] = col
    if not tenor_cols:
        raise ValueError("No forward tenor columns found (expected 1W/1M/2M/3M/6M/9M/1Y).")

    keep = ["date", spot_col] + list(tenor_cols.values())
    daily = fwd[keep].copy()
    daily = daily.dropna(subset=[spot_col]).copy()
    daily = daily.rename(columns={spot_col: "S"})
    daily["S"] = pd.to_numeric(daily["S"], errors="coerce")

    # ---- SANITY: spot scale
    spot_med = float(np.nanmedian(daily["S"].values))
    print(f"[SANITY] Spot column chosen: '{spot_col}' | median={spot_med:.6g}")
    if 0.0 < spot_med < 0.3:
        print("  [WARN] Spot median looks too small for FX spot (might be an interest-rate 'ON' column).")
    if spot_med > 50:
        print("  [WARN] Spot median looks too large for FX spot. Check the forward file.")

    # F/S ratios
    for t, col in tenor_cols.items():
        daily[t] = pd.to_numeric(daily[col], errors="coerce") / daily["S"]

    daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp()

    if agg == "mean":
        out = daily.groupby("month")[list(tenor_cols.keys())].mean().reset_index().rename(columns={"month": "date"})
    elif agg == "eom":
        out = daily.sort_values("date").groupby("month").tail(1)[["month"] + list(tenor_cols.keys())]
        out = out.rename(columns={"month": "date"}).reset_index(drop=True)
    else:
        raise ValueError("agg must be 'mean' or 'eom'")

    return out


def sanity_fs_ratio(out: pd.DataFrame, hedge_tenor: str):
    if hedge_tenor not in out.columns:
        print(f"[WARN] hedge_tenor={hedge_tenor} not present in monthly ratios output.")
        return

    q = out[hedge_tenor].dropna().quantile([0.05, 0.50, 0.95]).to_dict()
    print(f"[SANITY] F/S({hedge_tenor}) quantiles:", {k: float(v) for k, v in q.items()})
    med = float(out[hedge_tenor].dropna().median())
    if med < 0.7 or med > 1.3:
        print("  [WARN] F/S ratio median far from 1. Wrong spot column or quote convention mismatch.")


# -------------------------------------------------
# 3) Hedged conversion
# -------------------------------------------------
def eur_to_lvl_hedged_simple(r_eur_annual_pct: pd.Series, f_over_s: pd.Series, T_years: float) -> pd.Series:
    """
    Simple return version:
      R^{LVL,hedged(EUR)}_{t,T} = (1 + r^{EUR}_t * T) * (F/S) - 1
      annualised_simple = (R/T)
    """
    r = pd.to_numeric(r_eur_annual_pct, errors="coerce") / 100.0
    fs = pd.to_numeric(f_over_s, errors="coerce")
    R = (1.0 + r * T_years) * fs - 1.0
    return (R / T_years) * 100.0


# -------------------------------------------------
# 4) Build dataset by column letter + sanity checks
# -------------------------------------------------
def build_real_vs_synth_df_by_letter(
    lvl_csv: str,
    eur_csv: str,
    forwards_xlsx: str,
    col_letter: str,
    hedge_tenor: str,
    fwd_agg: str = "mean",
    auto_invert_fs: bool = True,
    do_print_sanity: bool = True
) -> pd.DataFrame:
    if hedge_tenor not in TENOR_TO_TYEARS:
        raise ValueError(f"Unknown hedge_tenor={hedge_tenor}. Allowed: {list(TENOR_TO_TYEARS)}")

    lvl_w = load_table21_wide(pick_path(lvl_csv))
    eur_w = load_table21_wide(pick_path(eur_csv))

    s_lvl = series_by_letter(lvl_w, col_letter).rename("rate_LVL")
    s_eur = series_by_letter(eur_w, col_letter).rename("rate_EUR")

    df = pd.concat([s_lvl, s_eur], axis=1).reset_index().rename(columns={"index": "date"})

    monthly_ratio = load_monthly_forward_ratios(pick_path(forwards_xlsx), agg=fwd_agg)
    if do_print_sanity:
        sanity_fs_ratio(monthly_ratio, hedge_tenor)

    df = df.merge(monthly_ratio, on="date", how="left").sort_values("date")

    if hedge_tenor not in df.columns:
        raise ValueError(f"Forward ratio for {hedge_tenor} not found. Available: {list(monthly_ratio.columns)}")

    # Optional heuristic: invert if median far from 1 (quote mismatch)
    fs = pd.to_numeric(df[hedge_tenor], errors="coerce")
    fs_med = float(np.nanmedian(fs.values))
    if auto_invert_fs and np.isfinite(fs_med) and (fs_med < 0.7 or fs_med > 1.3):
        print(f"[HEURISTIC] F/S({hedge_tenor}) median={fs_med:.4g} far from 1 -> trying inversion (1/(F/S)).")
        df[hedge_tenor] = 1.0 / fs
        fs2 = pd.to_numeric(df[hedge_tenor], errors="coerce")
        fs2_med = float(np.nanmedian(fs2.values))
        print(f"[HEURISTIC] After inversion median={fs2_med:.4g}.")
        if np.isfinite(fs2_med) and abs(fs2_med - 1.0) > abs(fs_med - 1.0):
            print("[HEURISTIC] Inversion made median farther from 1 -> reverting.")
            df[hedge_tenor] = fs

    T_years = TENOR_TO_TYEARS[hedge_tenor]
    df["hedged_eur_to_lvl"] = eur_to_lvl_hedged_simple(df["rate_EUR"], df[hedge_tenor], T_years)
    df["delta_lvl_minus_hedged"] = df["rate_LVL"] - df["hedged_eur_to_lvl"]
    df["fwd_premium"] = (pd.to_numeric(df[hedge_tenor], errors="coerce") - 1.0) / T_years * 100.0

    # Only meaningful pre-euro adoption
    pre = df["date"] < EURO_ADOPTION
    df.loc[~pre, ["hedged_eur_to_lvl", "delta_lvl_minus_hedged", "fwd_premium"]] = np.nan

    # Ensure numeric columns
    for c in ["rate_LVL", "rate_EUR", "hedged_eur_to_lvl", "delta_lvl_minus_hedged", "fwd_premium"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- SANITY: overlap counts pre-2014
    if do_print_sanity:
        pre_df = df.loc[df["date"] < EURO_ADOPTION].copy()
        n_lvl = int(pre_df["rate_LVL"].notna().sum())
        n_eur = int(pre_df["rate_EUR"].notna().sum())
        n_fs = int(pre_df[hedge_tenor].notna().sum())
        n_all = int((pre_df["rate_LVL"].notna() & pre_df["rate_EUR"].notna() & pre_df[hedge_tenor].notna()).sum())
        print(f"[SANITY] Pre-2014 non-missing counts: LVL={n_lvl}, EUR={n_eur}, F/S={n_fs}, ALL={n_all}")

        tmp = pre_df[["rate_LVL", "hedged_eur_to_lvl", "delta_lvl_minus_hedged"]].dropna()
        if len(tmp) > 10:
            q = pre_df["delta_lvl_minus_hedged"].quantile([0.10, 0.50, 0.90]).to_dict()
            corr = tmp[["rate_LVL", "hedged_eur_to_lvl"]].corr().iloc[0, 1]
            print("[SANITY] Δ quantiles (10/50/90):", {k: float(v) for k, v in q.items()})
            print("[SANITY] Corr(real LVL, synthetic hedged):", float(corr))
        else:
            print("[SANITY] Not enough overlapping observations to compute Δ quantiles / correlation.")

    return df


# -------------------------------------------------
# 5) Plotting helpers
# -------------------------------------------------
def plot_real_vs_synth(df: pd.DataFrame, title: str, outpath: Path, hedge_tenor: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["date"], df["rate_LVL"], lw=2.5, label="Real LVL rate")
    ax.plot(
        df["date"], df["hedged_eur_to_lvl"], lw=2.5, ls="--",
        label=f"Synthetic: EUR hedged into LVL ({hedge_tenor} fwd)"
    )
    ax.axvline(EURO_ADOPTION, color="black", lw=2.5, label="Euro Adoption")
    ax.set_title(title)
    ax.set_ylabel("Interest rate (%)")
    ax.set_xlabel("Year")
    ax.grid(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_delta(df: pd.DataFrame, title: str, outpath: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["date"], df["delta_lvl_minus_hedged"], lw=2.8, label="Δ = LVL − synthetic(hedged EUR→LVL)")
    mask = df["delta_lvl_minus_hedged"].notna()
    ax.fill_between(df.loc[mask, "date"], df.loc[mask, "delta_lvl_minus_hedged"], 0.0, alpha=0.12)
    ax.axhline(0, color="black", lw=1.2)
    ax.axvline(EURO_ADOPTION, color="black", lw=2.5)
    ax.set_title(title)
    ax.set_ylabel("Spread (pp)")
    ax.set_xlabel("Year")
    ax.grid(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# -------------------------------------------------
# 6) Tenor robustness check (optional overlays + tables)
# -------------------------------------------------
def robustness_check_tenors(
    name: str,
    lvl_csv: str,
    eur_csv: str,
    forwards_xlsx: str,
    col_letter: str,
    tenors=("3M", "6M", "1Y"),
    fwd_agg="mean"
):
    dfs = {}
    for t in tenors:
        print(f"\n--- {name}: building synthetic with {t} ---")
        df = build_real_vs_synth_df_by_letter(
            lvl_csv=lvl_csv,
            eur_csv=eur_csv,
            forwards_xlsx=forwards_xlsx,
            col_letter=col_letter,
            hedge_tenor=t,
            fwd_agg=fwd_agg,
            auto_invert_fs=True,
            do_print_sanity=True
        )
        dfs[t] = df

    # align on dates (pre-2014)
    base = None
    for t, df in dfs.items():
        x = df[["date", "rate_LVL", "hedged_eur_to_lvl", "delta_lvl_minus_hedged"]].copy()
        x = x.rename(columns={
            "hedged_eur_to_lvl": f"hedged_{t}",
            "delta_lvl_minus_hedged": f"delta_{t}"
        })
        base = x if base is None else base.merge(x, on=["date", "rate_LVL"], how="outer")

    pre = base["date"] < EURO_ADOPTION
    base = base.loc[pre].sort_values("date").copy()

    req_cols = [f"hedged_{t}" for t in tenors] + [f"delta_{t}" for t in tenors]
    base_all = base.dropna(subset=req_cols + ["rate_LVL"]).copy()

    # stats table
    rows = []
    for t in tenors:
        corr = base_all[["rate_LVL", f"hedged_{t}"]].corr().iloc[0, 1]
        rows.append({
            "tenor": t,
            "corr(real, hedged)": float(corr),
            "mean(|delta|)": float(np.nanmean(np.abs(base_all[f"delta_{t}"]))),
            "median(delta)": float(np.nanmedian(base_all[f"delta_{t}"]))
        })
    stats_df = pd.DataFrame(rows)

    # pairwise diffs across hedged series
    pair_rows = []
    for i in range(len(tenors)):
        for j in range(i + 1, len(tenors)):
            t1, t2 = tenors[i], tenors[j]
            d = base_all[f"hedged_{t1}"] - base_all[f"hedged_{t2}"]
            pair_rows.append({
                "pair": f"{t1} vs {t2}",
                "mean_abs_diff(hedged)": float(np.nanmean(np.abs(d))),
                "max_abs_diff(hedged)": float(np.nanmax(np.abs(d))),
                "corr(hedged)": float(base_all[[f"hedged_{t1}", f"hedged_{t2}"]].corr().iloc[0, 1])
            })
    pair_df = pd.DataFrame(pair_rows)

    # plots: hedged overlays + delta overlays
    safe_name = name.lower().replace(" ", "_").replace("—", "-").replace("–", "-")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(base["date"], base["rate_LVL"], lw=2.8, label="Real LVL")
    for t in tenors:
        ax.plot(base["date"], base[f"hedged_{t}"], lw=2.2, ls="--", label=f"Synthetic (EUR→LVL) {t}")
    ax.axvline(EURO_ADOPTION, color="black", lw=2.2)
    ax.set_title(f"{name}: Real vs Synthetic — Tenor robustness")
    ax.set_ylabel("Interest rate (%)")
    ax.set_xlabel("Year")
    ax.grid(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{safe_name}_real_vs_synth_tenor_robust.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    for t in tenors:
        ax.plot(base["date"], base[f"delta_{t}"], lw=2.4, label=f"Δ ({t})")
    ax.axhline(0, color="black", lw=1.2)
    ax.axvline(EURO_ADOPTION, color="black", lw=2.2)
    ax.set_title(f"{name}: Wedge Δ — Tenor robustness")
    ax.set_ylabel("Spread (pp)")
    ax.set_xlabel("Year")
    ax.grid(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{safe_name}_delta_tenor_robust.png", dpi=300)
    plt.close(fig)

    # save tables
    stats_df.to_csv(OUTPUT_DIR / f"{safe_name}_tenor_stats.csv", index=False)
    pair_df.to_csv(OUTPUT_DIR / f"{safe_name}_tenor_pairwise.csv", index=False)

    print("\n[ROBUSTNESS] Saved:")
    print(" ", OUTPUT_DIR / f"{safe_name}_real_vs_synth_tenor_robust.png")
    print(" ", OUTPUT_DIR / f"{safe_name}_delta_tenor_robust.png")
    print(" ", OUTPUT_DIR / f"{safe_name}_tenor_stats.csv")
    print(" ", OUTPUT_DIR / f"{safe_name}_tenor_pairwise.csv")
    print("\n[ROBUSTNESS] Tenor stats:\n", stats_df)
    print("\n[ROBUSTNESS] Pairwise hedged comparisons:\n", pair_df)


# -------------------------------------------------
# 7) Run baseline + (optional) robustness
# -------------------------------------------------
def main():
    # Set Table21 column letters (must match your verified mapping)
    HH_DEP_COL = "D"     # HH deposits — new business, agreed maturity, up to 1Y
    NFC_LOAN_COL = "AW"  # NFC loans — outstanding, maturity up to 1Y

    # --------- Baseline: Deposits ----------
    df_dep = build_real_vs_synth_df_by_letter(
        lvl_csv="table21_rates_deposits_lats.csv",
        eur_csv="table21_rates_deposits_euro.csv",
        forwards_xlsx="latvian_forwards.xlsx",
        col_letter=HH_DEP_COL,
        hedge_tenor=BASELINE_TENOR,
        fwd_agg="mean",
        auto_invert_fs=True,
        do_print_sanity=True
    )

    out_dep_csv = OUTPUT_DIR / f"latvia_dep_hh_upto1y_real_vs_synth_{BASELINE_TENOR}.csv"
    df_dep.to_csv(out_dep_csv, index=False)

    plot_real_vs_synth(
        df_dep,
        title="Latvia: HH Deposits (Agreed maturity, up to 1Y) — Real vs Synthetic (FX-hedged)",
        outpath=OUTPUT_DIR / f"dep_hh_upto1y_real_vs_synth_{BASELINE_TENOR}.png",
        hedge_tenor=BASELINE_TENOR
    )
    plot_delta(
        df_dep,
        title="Latvia: HH Deposits (Up to 1Y) — Home premium net of FX",
        outpath=OUTPUT_DIR / f"dep_hh_upto1y_delta_{BASELINE_TENOR}.png"
    )

    # --------- Baseline: Loans ----------
    df_loan = build_real_vs_synth_df_by_letter(
        lvl_csv="table21_rates_loans_lats.csv",
        eur_csv="table21_rates_loans_euro.csv",
        forwards_xlsx="latvian_forwards.xlsx",
        col_letter=NFC_LOAN_COL,
        hedge_tenor=BASELINE_TENOR,
        fwd_agg="mean",
        auto_invert_fs=True,
        do_print_sanity=True
    )

    out_loan_csv = OUTPUT_DIR / f"latvia_loan_nfc_upto1y_real_vs_synth_{BASELINE_TENOR}.csv"
    df_loan.to_csv(out_loan_csv, index=False)

    plot_real_vs_synth(
        df_loan,
        title="Latvia: NFC Loans (Up to 1Y) — Real vs Synthetic (FX-hedged)",
        outpath=OUTPUT_DIR / f"loan_nfc_upto1y_real_vs_synth_{BASELINE_TENOR}.png",
        hedge_tenor=BASELINE_TENOR
    )
    plot_delta(
        df_loan,
        title="Latvia: NFC Loans (Up to 1Y) — Home premium net of FX",
        outpath=OUTPUT_DIR / f"loan_nfc_upto1y_delta_{BASELINE_TENOR}.png"
    )

    print("\nSaved baseline CSVs:")
    print(" ", out_dep_csv)
    print(" ", out_loan_csv)
    print("Baseline PNGs in:", OUTPUT_DIR)

    # --------- Optional: tenor robustness ----------
    if RUN_TENOR_ROBUSTNESS:
        robustness_check_tenors(
            name="HH Deposits up to 1Y",
            lvl_csv="table21_rates_deposits_lats.csv",
            eur_csv="table21_rates_deposits_euro.csv",
            forwards_xlsx="latvian_forwards.xlsx",
            col_letter=HH_DEP_COL,
            tenors=("3M", "6M", "1Y"),
            fwd_agg="mean"
        )
        robustness_check_tenors(
            name="NFC Loans up to 1Y",
            lvl_csv="table21_rates_loans_lats.csv",
            eur_csv="table21_rates_loans_euro.csv",
            forwards_xlsx="latvian_forwards.xlsx",
            col_letter=NFC_LOAN_COL,
            tenors=("3M", "6M", "1Y"),
            fwd_agg="mean"
        )


if __name__ == "__main__":
    main()
