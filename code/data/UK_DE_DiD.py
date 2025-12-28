from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

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
# Reading ECB Data Portal exports (CSV or Excel)
# -------------------------------------------------
def read_ecb_portal_file(path: Path) -> pd.Series:
    """
    Reads an ECB Data Portal export (CSV/XLSX) and returns a Series indexed by date.
    """
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    date_col = "DATE" if "DATE" in df.columns else df.columns[0]
    value_cols = [c for c in df.columns if c not in [date_col, "TIME PERIOD"]]
    if len(value_cols) != 1:
        value_cols = [c for c in df.columns if c != date_col]
        val_col = value_cols[-1]
    else:
        val_col = value_cols[0]

    s = df[[date_col, val_col]].copy()
    s.columns = ["date", "value"]
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s["value"] = pd.to_numeric(s["value"], errors="coerce")

    s = s.dropna(subset=["date"]).set_index("date")["value"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def find_one(stem: str) -> Path:
    """
    Looks for {stem}.csv or {stem}.xlsx or {stem}.xls inside DATA_DIR.
    """
    for ext in (".csv", ".xlsx", ".xls"):
        p = DATA_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing {stem}.(csv/xlsx/xls) in {DATA_DIR}")


def load_inputs() -> Dict[str, pd.Series]:
    """
    Expects these 8 stems:
      DE_HH_domestic, DE_HH_foreign, UK_HH_domestic, UK_HH_foreign,
      DE_NFC_domestic, DE_NFC_foreign, UK_NFC_domestic, UK_NFC_foreign
    """
    stems = [
        "DE_HH_domestic", "DE_HH_foreign",
        "UK_HH_domestic", "UK_HH_foreign",
        "DE_NFC_domestic", "DE_NFC_foreign",
        "UK_NFC_domestic", "UK_NFC_foreign",
    ]
    return {stem: read_ecb_portal_file(find_one(stem)) for stem in stems}


# -------------------------------------------------
# Shares & gaps
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


def make_panel(y_uk: pd.Series, y_de: pd.Series) -> pd.DataFrame:
    return pd.concat([y_uk.rename("UK"), y_de.rename("DE")], axis=1).dropna().sort_index()


def gap_synth(panel: pd.DataFrame) -> pd.Series:
    # With only DE as donor: synth == DE
    return (panel["UK"] - panel["DE"]).rename("gap_synth")


def gap_did(panel: pd.DataFrame, pre_end: str) -> pd.Series:
    """
    DiD-style gap series: (UK-DE) demeaned by the mean of (UK-DE) in pre period <= pre_end
    """
    pre_end = pd.to_datetime(pre_end)
    diff = panel["UK"] - panel["DE"]
    pre_mean = diff.loc[diff.index <= pre_end].mean()
    return (diff - pre_mean).rename("gap_did")


# -------------------------------------------------
# Significance tests (HAC/Newey-West)
# -------------------------------------------------
def hac_post_test(
    panel: pd.DataFrame,
    event_date: str,
    lags: int = 12,
    add_trend: bool = False,
    sample_start: Optional[str] = None,
    sample_end: Optional[str] = None,
) -> Tuple[float, float, float, float, int]:
    """
    Tests whether the mean UK-DE gap shifts after event_date:
        d_t = alpha + beta * post_t + u_t   (optionally + gamma * trend_t)
    where d_t = UK - DE and post_t = 1{t > event_date}.
    Returns: beta, se(HAC), t, p, n_obs
    """
    d = (panel["UK"] - panel["DE"]).dropna()
    event = pd.to_datetime(event_date)

    mask = pd.Series(True, index=d.index)
    if sample_start is not None:
        mask &= (d.index >= pd.to_datetime(sample_start))
    if sample_end is not None:
        mask &= (d.index <= pd.to_datetime(sample_end))
    d = d.loc[mask].dropna()

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


def run_significance_table(panels: Dict[str, pd.DataFrame], events: Dict[str, str], lags: int = 12) -> pd.DataFrame:
    rows = []
    for block, panel in panels.items():
        for ev_name, ev_date in events.items():
            b, se, t, p, n = hac_post_test(panel, ev_date, lags=lags, add_trend=False)
            rows.append([block, ev_name, "post_only", b, se, t, p, n])

            b, se, t, p, n = hac_post_test(panel, ev_date, lags=lags, add_trend=True)
            rows.append([block, ev_name, "post_plus_trend", b, se, t, p, n])

            if ev_name == "ref_2016_06_30":
                b, se, t, p, n = hac_post_test(
                    panel, ev_date, lags=lags, add_trend=False,
                    sample_start="2013-01-01", sample_end="2019-12-31"
                )
                rows.append([block, ev_name, "post_only_2013_2019", b, se, t, p, n])

    out = pd.DataFrame(
        rows,
        columns=["block", "event", "spec", "beta_post", "se_hac", "t", "p", "n_obs"]
    ).sort_values(["block", "event", "spec"])
    return out


# -------------------------------------------------
# Event study (leads/lags) — robust month indexing
# -------------------------------------------------
def event_study_design_matrix(
    idx: pd.DatetimeIndex,
    event_date: str,
    k_pre: int,
    k_post: int,
    freq: str = "M",
    omit_k: int = -1,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build lead/lag dummies for an event-study, using robust monthly month-distance:
        k = (year*12 + month) difference between idx and event_date.
    """
    event = pd.to_datetime(event_date)

    if freq.upper() != "M":
        raise ValueError("Only monthly ('M') implemented.")

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


def run_event_study(
    panel: pd.DataFrame,
    event_date: str,
    k_pre: int = 36,
    k_post: int = 36,
    omit_k: int = -1,
    hac_lags: int = 12,
    restrict_start: str | None = None,
    restrict_end: str | None = None,
) -> pd.DataFrame:
    """
    Event-study on d_t = UK - DE with lead/lag dummies.
    Baseline omitted period is omit_k (default -1).
    """
    d = (panel["UK"] - panel["DE"]).dropna()

    if restrict_start is not None:
        d = d.loc[d.index >= pd.to_datetime(restrict_start)]
    if restrict_end is not None:
        d = d.loc[d.index <= pd.to_datetime(restrict_end)]

    X, k_ser = event_study_design_matrix(d.index, event_date, k_pre, k_post, omit_k=omit_k)
    in_window = (k_ser >= -k_pre) & (k_ser <= k_post)
    d = d.loc[in_window]
    X = X.loc[in_window]
    k_ser = k_ser.loc[in_window]

    res = sm.OLS(d.values, X.values).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    rows = []
    colnames = X.columns.tolist()

    for kk in range(-k_pre, k_post + 1):
        if kk == omit_k:
            rows.append([kk, 0.0, np.nan, np.nan, np.nan])
            continue
        name = f"k_{kk}"
        j = colnames.index(name)
        beta = float(res.params[j])
        se = float(res.bse[j])
        t = float(res.tvalues[j])
        p = float(res.pvalues[j])
        rows.append([kk, beta, se, t, p])

    out = pd.DataFrame(rows, columns=["k", "beta", "se_hac", "t", "p"])
    out["ci_low"] = out["beta"] - 1.96 * out["se_hac"]
    out["ci_high"] = out["beta"] + 1.96 * out["se_hac"]
    out["n_obs"] = int(res.nobs)
    out["event_date"] = pd.to_datetime(event_date)
    out["omit_k"] = omit_k
    out["k_pre"] = k_pre
    out["k_post"] = k_post
    out["hac_lags"] = hac_lags
    return out


def plot_event_study(est: pd.DataFrame, title: str, outpath: Path, vline_k: int = 0) -> None:
    est = est.sort_values("k")

    plt.figure()
    plt.plot(est["k"], est["beta"], marker="o")
    m = est["se_hac"].notna()
    plt.fill_between(est.loc[m, "k"], est.loc[m, "ci_low"], est.loc[m, "ci_high"], alpha=0.2)

    plt.axhline(0.0, linewidth=1)
    plt.axvline(vline_k, linestyle="--")

    plt.title(title)
    plt.xlabel("Event time k (months)")
    plt.ylabel("Effect on d_t = (UK-DE) relative to baseline k=-1")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -------------------------------------------------
# Plots
# -------------------------------------------------
def plot_shares(panel: pd.DataFrame, title: str, outpath: Path) -> None:
    plt.figure()
    plt.plot(panel.index, panel["UK"], label="UK")
    plt.plot(panel.index, panel["DE"], label="DE")
    plt.title(title)
    plt.ylabel("Home share")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_gaps(df: pd.DataFrame, vline: str, title: str, outpath: Path) -> None:
    plt.figure()
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    plt.axvline(pd.to_datetime(vline), linestyle="--")
    plt.title(title)
    plt.ylabel("Gap")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -------------------------------------------------
# Run
# -------------------------------------------------
def main():
    events = {
        "ref_2016_06_30": "2016-06-30",
        "exit_2020_01_31": "2020-01-31",
        "trans_end_2020_12_31": "2020-12-31",
    }

    s = load_inputs()

    # Shares (HH, NFC, Total)
    y_hh_uk = home_share(s["UK_HH_domestic"], s["UK_HH_foreign"])
    y_hh_de = home_share(s["DE_HH_domestic"], s["DE_HH_foreign"])

    y_nfc_uk = home_share(s["UK_NFC_domestic"], s["UK_NFC_foreign"])
    y_nfc_de = home_share(s["DE_NFC_domestic"], s["DE_NFC_foreign"])

    y_tot_uk = home_share_total(
        s["UK_HH_domestic"], s["UK_NFC_domestic"],
        s["UK_HH_foreign"], s["UK_NFC_foreign"]
    )
    y_tot_de = home_share_total(
        s["DE_HH_domestic"], s["DE_NFC_domestic"],
        s["DE_HH_foreign"], s["DE_NFC_foreign"]
    )

    panels = {
        "HH": make_panel(y_hh_uk, y_hh_de),
        "NFC": make_panel(y_nfc_uk, y_nfc_de),
        "TOT": make_panel(y_tot_uk, y_tot_de),
    }

    # Build output tables
    sheets = {}
    for block, panel in panels.items():
        out = pd.DataFrame(index=panel.index)
        out[f"y_UK_{block}"] = panel["UK"]
        out[f"y_DE_{block}"] = panel["DE"]
        out[f"gap_synth_{block}"] = gap_synth(panel)
        for ev, pre_end in events.items():
            out[f"gap_did_{block}_{ev}"] = gap_did(panel, pre_end)
        sheets[block] = out

    wide = pd.concat(sheets.values(), axis=1)

    # Save main CSV (always)
    csv_path = OUTPUT_DIR / "home_bias_ea_shares_and_gaps.csv"
    wide.to_csv(csv_path)

    # Save XLSX only if openpyxl is available
    xlsx_path = OUTPUT_DIR / "home_bias_ea_shares_and_gaps.xlsx"
    try:
        import openpyxl  # noqa: F401
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name)
            wide.to_excel(writer, sheet_name="ALL")
        wrote_xlsx = True
    except ModuleNotFoundError:
        wrote_xlsx = False
        for name, df in sheets.items():
            df.to_csv(OUTPUT_DIR / f"home_bias_{name}.csv")

    # Plots (shares)
    plot_shares(panels["HH"], "EA-only home share (Households)", OUTPUT_DIR / "plot_home_share_HH.png")
    plot_shares(panels["NFC"], "EA-only home share (NFCs)", OUTPUT_DIR / "plot_home_share_NFC.png")
    plot_shares(panels["TOT"], "EA-only home share (HH+NFC)", OUTPUT_DIR / "plot_home_share_TOT.png")

    # Plot DiD gaps for the referendum cutoff
    ref = events["ref_2016_06_30"]
    gaps_ref = pd.DataFrame(
        {
            "HH": sheets["HH"]["gap_did_HH_ref_2016_06_30"],
            "NFC": sheets["NFC"]["gap_did_NFC_ref_2016_06_30"],
            "TOT": sheets["TOT"]["gap_did_TOT_ref_2016_06_30"],
        }
    ).dropna()

    plot_gaps(
        gaps_ref,
        vline=ref,
        title="DiD gaps: (UK-DE) demeaned by pre-period mean (<= 2016-06-30)",
        outpath=OUTPUT_DIR / "plot_did_gaps_ref2016.png",
    )

    # Significance table
    sig = run_significance_table(panels, events, lags=12)
    sig_path = OUTPUT_DIR / "did_significance_hac_tests.csv"
    sig.to_csv(sig_path, index=False)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 50)
    print("\n=== HAC significance tests (beta_post = post-period shift in UK-DE gap) ===")
    print(sig.to_string(index=False))

    # Event-study around referendum
    EVENT_DATE = events["ref_2016_06_30"]
    K_PRE = 36
    K_POST = 36
    HAC_LAGS = 12
    OMIT_K = -1

    for block, panel in panels.items():
        est = run_event_study(
            panel,
            event_date=EVENT_DATE,
            k_pre=K_PRE,
            k_post=K_POST,
            omit_k=OMIT_K,
            hac_lags=HAC_LAGS,
        )
        est_path = OUTPUT_DIR / f"event_study_{block}_ref2016.csv"
        est.to_csv(est_path, index=False)

        plot_path = OUTPUT_DIR / f"plot_event_study_{block}_ref2016.png"
        plot_event_study(
            est,
            title=f"Event study (UK-DE), {block}: event = 2016-06",
            outpath=plot_path,
            vline_k=0,
        )

    print("\nDone.")
    print(f"Inputs read from: {DATA_DIR}")
    print(f"Outputs written to: {OUTPUT_DIR}")
    print(f"  - {csv_path.name}")
    if wrote_xlsx:
        print(f"  - {xlsx_path.name}")
    else:
        print("  - (xlsx skipped: openpyxl not installed) wrote per-block CSVs home_bias_HH.csv / home_bias_NFC.csv / home_bias_TOT.csv")
    print(f"  - {sig_path.name}")
    print("Plots:")
    print("  - plot_home_share_HH.png")
    print("  - plot_home_share_NFC.png")
    print("  - plot_home_share_TOT.png")
    print("  - plot_did_gaps_ref2016.png")
    print("Event-study outputs:")
    print("  - event_study_<HH|NFC|TOT>_ref2016.csv")
    print("  - plot_event_study_<HH|NFC|TOT>_ref2016.png")


if __name__ == "__main__":
    main()
