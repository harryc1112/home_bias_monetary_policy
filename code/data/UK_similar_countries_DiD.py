from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(os.path.expanduser("~")) / "Dropbox" / "home_bias_monetary_policy"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output" / "DiD"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# INPUT FILES (combined multi-country exports)
# -------------------------------------------------
FILES = {
    "HH_domestic": "HH_domestic.csv",
    "NFC_domestic": "NFC_domestic.csv",
    "HH_foreign_EA": "Euro_Area_HH_foreign.csv",
    "NFC_foreign_EA": "Euro_Area_NFC_foreign.csv",
}

# Treated and donor pool candidates
TREATED = "GB"  # ECB uses GB in BSI keys
DONOR_CANDIDATES = ["DK", "SE", "PL", "CZ", "HU"]  # we will auto-filter based on availability


# -------------------------------------------------
# Helpers: parse combined ECB export columns
# -------------------------------------------------
ISO_PAT = re.compile(r"\(BSI\.M\.([A-Z]{2})\.", re.IGNORECASE)
ISO_ONLY_PAT = re.compile(r"^[A-Z]{2}$")


def _detect_date_col(df: pd.DataFrame) -> str:
    for c in ["DATE", "TIME PERIOD", "Time period", "TIME_PERIOD", "date"]:
        if c in df.columns:
            return c
    return df.columns[0]


def _col_to_iso(col: str) -> Optional[str]:
    s = str(col).strip()
    if ISO_ONLY_PAT.match(s.upper()):
        return s.upper()
    m = ISO_PAT.search(s)
    if m:
        return m.group(1).upper()
    return None


def read_combined_ecb_file(path: Path) -> pd.DataFrame:
    """
    Supports:
      (A) ISO as column headers
      (B) ISO embedded in headers '(BSI.M.<ISO>...)'
      (C) ISO codes in first data row (metadata row)
    """
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    date_col = _detect_date_col(df)

    # Case (C): metadata row contains ISO codes
    tmp_dates = pd.to_datetime(df[date_col], errors="coerce")
    if len(df) > 1 and pd.isna(tmp_dates.iloc[0]) and tmp_dates.notna().sum() >= 3:
        first_row = df.iloc[0]
        iso_map: Dict[str, str] = {}
        for c in df.columns:
            if c == date_col:
                continue
            v = str(first_row[c]).strip().upper()
            if ISO_ONLY_PAT.match(v):
                iso_map[c] = v
        if len(iso_map) >= 3:
            df = df.drop(index=df.index[0]).copy()
            df = df.rename(columns=iso_map)

    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Detect ISO columns
    iso_cols: Dict[str, str] = {}
    for c in df.columns:
        iso = _col_to_iso(c)
        if iso is None:
            continue
        iso_cols[iso] = c

    if not iso_cols:
        sample_cols = list(df.columns)[:15]
        raise ValueError(
            f"No ISO-coded columns detected in {path.name}. "
            f"First columns: {sample_cols}. "
            f"If ISO codes are in a metadata row, ensure it is the FIRST row."
        )

    iso_df = pd.DataFrame(index=df.index)
    for iso, colname in iso_cols.items():
        iso_df[iso] = pd.to_numeric(df[colname], errors="coerce")

    iso_df = iso_df.dropna(how="all")
    return iso_df


def load_all_inputs() -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for k, fname in FILES.items():
        p = DATA_DIR / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing {fname} in {DATA_DIR}")
        out[k] = read_combined_ecb_file(p)
    return out


# -------------------------------------------------
# Home shares
# -------------------------------------------------
def home_share(dom: pd.Series, foreign: pd.Series) -> pd.Series:
    df = pd.concat([dom.rename("dom"), foreign.rename("for")], axis=1).dropna()
    return (df["dom"] / (df["dom"] + df["for"])).rename("home_share")


def home_share_total(dom_hh: pd.Series, dom_nfc: pd.Series, for_hh: pd.Series, for_nfc: pd.Series) -> pd.Series:
    df = pd.concat(
        [dom_hh.rename("dhh"), dom_nfc.rename("dnfc"), for_hh.rename("fhh"), for_nfc.rename("fnfc")],
        axis=1
    ).dropna()
    dom = df["dhh"] + df["dnfc"]
    fore = df["fhh"] + df["fnfc"]
    return (dom / (dom + fore)).rename("home_share")


def build_country_shares(inputs: Dict[str, pd.DataFrame], country: str) -> Dict[str, pd.Series]:
    cc = country.upper()
    hh_dom = inputs["HH_domestic"][cc]
    nfc_dom = inputs["NFC_domestic"][cc]
    hh_for = inputs["HH_foreign_EA"][cc]
    nfc_for = inputs["NFC_foreign_EA"][cc]

    y_hh = home_share(hh_dom, hh_for)
    y_nfc = home_share(nfc_dom, nfc_for)
    y_tot = home_share_total(hh_dom, nfc_dom, hh_for, nfc_for)
    return {"HH": y_hh, "NFC": y_nfc, "TOT": y_tot}


# -------------------------------------------------
# Synthetic weights (simplex projection + projected gradient)
# -------------------------------------------------
def project_to_simplex(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, v.size + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        w = np.zeros_like(v)
        w[np.argmax(v)] = 1.0
        return w
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    if s <= 0:
        w = np.zeros_like(v)
        w[np.argmax(v)] = 1.0
    else:
        w = w / s
    return w


def fit_synth_weights(
    y: np.ndarray,
    X: np.ndarray,
    n_iter: int = 8000,
    lr: float = 0.05,
    tol: float = 1e-12,
) -> np.ndarray:
    w = np.ones(X.shape[1]) / X.shape[1]
    XtX = X.T @ X
    Xty = X.T @ y

    last_obj = np.inf
    for _ in range(n_iter):
        grad = 2.0 * (XtX @ w - Xty)
        w_new = project_to_simplex(w - lr * grad)
        r = y - X @ w_new
        obj = float(r @ r)
        if abs(last_obj - obj) < tol:
            w = w_new
            break
        w, last_obj = w_new, obj
    return w


# -------------------------------------------------
# HAC tests + event study for a GAP series
# -------------------------------------------------
def hac_post_test_gap(
    gap: pd.Series,
    event_date: str,
    lags: int = 12,
    add_trend: bool = False,
) -> Tuple[float, float, float, float, int]:
    d = gap.dropna().copy()
    event = pd.to_datetime(event_date)

    post = (d.index > event).astype(int)
    X = pd.DataFrame({"const": 1.0, "post": post}, index=d.index)
    if add_trend:
        X["trend"] = np.arange(len(X), dtype=float)

    res = sm.OLS(d.values, X.values).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    beta = float(res.params[1])
    se = float(res.bse[1])
    t = float(res.tvalues[1])
    p = float(res.pvalues[1])
    n = int(res.nobs)
    return beta, se, t, p, n


def event_study_design_matrix_months(
    idx: pd.DatetimeIndex,
    event_date: str,
    k_pre: int,
    k_post: int,
    omit_k: int = -1,
) -> tuple[pd.DataFrame, pd.Series]:
    event = pd.to_datetime(event_date)
    idx_p = pd.PeriodIndex(idx, freq="M")
    ev_p = pd.Period(event, freq="M")
    k_vals = (idx_p.year * 12 + idx_p.month) - (ev_p.year * 12 + ev_p.month)
    k = pd.Series(k_vals.astype(int), index=idx, name="event_time")

    X = pd.DataFrame(index=idx)
    X["const"] = 1.0
    for kk in range(-k_pre, k_post + 1):
        if kk == omit_k:
            continue
        X[f"k_{kk}"] = (k == kk).astype(int).values
    return X, k


def run_event_study_gap(
    gap: pd.Series,
    event_date: str,
    k_pre: int = 36,
    k_post: int = 36,
    omit_k: int = -1,
    hac_lags: int = 12,
) -> pd.DataFrame:
    d = gap.dropna().copy()
    X, k_ser = event_study_design_matrix_months(d.index, event_date, k_pre, k_post, omit_k=omit_k)

    in_window = (k_ser >= -k_pre) & (k_ser <= k_post)
    d = d.loc[in_window]
    X = X.loc[in_window]

    res = sm.OLS(d.values, X.values).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    rows = []
    colnames = X.columns.tolist()
    for kk in range(-k_pre, k_post + 1):
        if kk == omit_k:
            rows.append([kk, 0.0, np.nan, np.nan, np.nan])
            continue
        j = colnames.index(f"k_{kk}")
        beta = float(res.params[j])
        se = float(res.bse[j])
        t = float(res.tvalues[j])
        p = float(res.pvalues[j])
        rows.append([kk, beta, se, t, p])

    out = pd.DataFrame(rows, columns=["k", "beta", "se_hac", "t", "p"])
    out["ci_low"] = out["beta"] - 1.96 * out["se_hac"]
    out["ci_high"] = out["beta"] + 1.96 * out["se_hac"]
    out["n_obs"] = int(res.nobs)
    return out


# -------------------------------------------------
# Plot helpers
# -------------------------------------------------
def plot_series(df: pd.DataFrame, title: str, ylabel: str, outpath: Path, vline: Optional[str] = None) -> None:
    plt.figure()
    for c in df.columns:
        plt.plot(df.index, df[c], label=c)
    if vline is not None:
        plt.axvline(pd.to_datetime(vline), linestyle="--")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_event_study(est: pd.DataFrame, title: str, outpath: Path) -> None:
    est = est.sort_values("k")
    plt.figure()
    plt.plot(est["k"], est["beta"], marker="o")
    m = est["se_hac"].notna()
    plt.fill_between(est.loc[m, "k"], est.loc[m, "ci_low"], est.loc[m, "ci_high"], alpha=0.2)
    plt.axhline(0.0, linewidth=1)
    plt.axvline(0, linestyle="--")
    plt.title(title)
    plt.xlabel("Event time k (months)")
    plt.ylabel("Effect on gap relative to baseline k=-1")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    events = {
        "ref_2016_06_30": "2016-06-30",
        "exit_2020_01_31": "2020-01-31",
        "trans_end_2020_12_31": "2020-12-31",
    }

    inputs = load_all_inputs()

    # Print availability (useful debug)
    print("\n=== ISO columns available by file (intersection will be used) ===")
    for k, df in inputs.items():
        avail = sorted(df.columns.tolist())
        print(f"{FILES[k]}: {avail}")

    # Donors = intersection across all four inputs and your candidate list
    avail_all = set(inputs["HH_domestic"].columns)
    for key in ["NFC_domestic", "HH_foreign_EA", "NFC_foreign_EA"]:
        avail_all &= set(inputs[key].columns)

    donors = [cc for cc in DONOR_CANDIDATES if cc in avail_all]
    if TREATED not in avail_all:
        raise ValueError(f"Treated country {TREATED} not available in all four files. Available intersection: {sorted(avail_all)}")

    if len(donors) < 2:
        raise ValueError(f"Too few donors available after intersection: {donors}. Intersection was: {sorted(avail_all)}")

    print("\nUsing treated:", TREATED)
    print("Using donors:", donors)
    print("Dropped donors (not available):", [cc for cc in DONOR_CANDIDATES if cc not in donors])

    treated_sh = build_country_shares(inputs, TREATED)
    donor_sh = {cc: build_country_shares(inputs, cc) for cc in donors}

    pre_end = pd.to_datetime(events["ref_2016_06_30"])
    out_rows = []

    for block in ["HH", "NFC", "TOT"]:
        df = pd.DataFrame({"TREATED": treated_sh[block]})
        for cc in donors:
            df[cc] = donor_sh[cc][block]
        df = df.dropna().sort_index()

        df_pre = df.loc[df.index <= pre_end]
        y = df_pre["TREATED"].values
        X = df_pre[donors].values

        w = fit_synth_weights(y, X, n_iter=8000, lr=0.05)

        pd.DataFrame({"donor": donors, "weight": w}).to_csv(OUTPUT_DIR / f"synth_weights_{block}.csv", index=False)

        df["SYNTH"] = df[donors].values @ w
        df["GAP"] = df["TREATED"] - df["SYNTH"]
        df.to_csv(OUTPUT_DIR / f"panel_{TREATED}_vs_SYNTH_{block}.csv")

        plot_series(
            df[["TREATED", "SYNTH"]].rename(columns={"TREATED": TREATED}),
            title=f"{block}: {TREATED} vs Synthetic (donors={','.join(donors)})",
            ylabel="Home share (U6 / (U6+U2))",
            outpath=OUTPUT_DIR / f"plot_{block}_{TREATED}_vs_SYNTH.png",
            vline=events["ref_2016_06_30"],
        )

        plot_series(
            df[["GAP"]],
            title=f"{block}: Gap = {TREATED} - Synthetic",
            ylabel="Gap",
            outpath=OUTPUT_DIR / f"plot_{block}_gap_{TREATED}_minus_SYNTH.png",
            vline=events["ref_2016_06_30"],
        )

        gap = df["GAP"].rename("gap")
        for ev_name, ev_date in events.items():
            b, se, t, p, n = hac_post_test_gap(gap, ev_date, lags=12, add_trend=False)
            out_rows.append([block, ev_name, "post_only", b, se, t, p, n])

            b, se, t, p, n = hac_post_test_gap(gap, ev_date, lags=12, add_trend=True)
            out_rows.append([block, ev_name, "post_plus_trend", b, se, t, p, n])

        est = run_event_study_gap(gap, events["ref_2016_06_30"], k_pre=36, k_post=36, omit_k=-1, hac_lags=12)
        est.to_csv(OUTPUT_DIR / f"event_study_SYNTH_{block}_ref2016.csv", index=False)
        plot_event_study(
            est,
            title=f"Event study on gap ({TREATED} - Synthetic), {block}: event=2016-06",
            outpath=OUTPUT_DIR / f"plot_event_study_SYNTH_{block}_ref2016.png",
        )

    sig = pd.DataFrame(
        out_rows,
        columns=["block", "event", "spec", "beta_post", "se_hac", "t", "p", "n_obs"]
    ).sort_values(["block", "event", "spec"])
    sig.to_csv(OUTPUT_DIR / "did_significance_SYNTH_hac_tests.csv", index=False)

    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 80)
    print("\n=== Improved DiD: HAC tests on GAP = treated - synthetic ===")
    print(sig.to_string(index=False))

    print("\nDone. Outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
