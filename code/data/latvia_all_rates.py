# =========================================================
# Latvia — FX-hedged EUR->LVL benchmark + "home premium net of FX"
# (Plot style MATCHES your merged Latvia plotting script)
# =========================================================

from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# -------------------------------------------------
# 0) Paths & constants  (match your main script)
# -------------------------------------------------
BASE_DIR = Path(os.path.expanduser("~")) / "Dropbox" / "home_bias_monetary_policy"
DATA_DIR = BASE_DIR / "data"
ALT_DATA_DIR = Path("/mnt/data")

OUTPUT_DIR = BASE_DIR / "output" / "motivating_facts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PEG_DATE    = pd.Timestamp("2005-01-01")
EURO_ADOPTION = pd.Timestamp("2014-01-01")
CRISIS_DATE = pd.Timestamp("2018-02-01")

# Match your global plotting style
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

# -------------------------------------------------
# Helper: pick path from Dropbox or /mnt/data  (match your script pattern)
# -------------------------------------------------
def pick_path(filename: str) -> Path:
    p = DATA_DIR / filename
    if p.exists():
        return p
    p2 = ALT_DATA_DIR / filename
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Could not find {filename} in {DATA_DIR} or {ALT_DATA_DIR}")

# -----------------------------
# 1) Table 21 parser (header ffill)
# -----------------------------
ROMAN_MAP = {"I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6,"VII":7,"VIII":8,"IX":9,"X":10,"XI":11,"XII":12}

def _make_unique(names):
    counts = {}
    out = []
    for n in names:
        if n not in counts:
            counts[n] = 0
            out.append(n)
        else:
            counts[n] += 1
            out.append(f"{n}__{counts[n]}")
    return out

def parse_table21_ffill_headers(path: Path, currency: str, kind: str) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, dtype=str)

    # find first year row (e.g. "2004")
    start = None
    for i in range(len(raw)):
        v = raw.iloc[i, 0]
        if isinstance(v, str) and re.fullmatch(r"\d{4}", v.strip()):
            start = i
            break
    if start is None:
        raise ValueError(f"No year row found in {path}")

    header_rows = raw.iloc[:start, :].copy()
    header_rows = header_rows.replace({np.nan: None})

    # forward-fill each header row across columns (from col 2 onward)
    for i in range(header_rows.shape[0]):
        filled = list(header_rows.iloc[i, :].values)
        last = None
        for j in range(2, len(filled)):
            cell = filled[j]
            s = None if cell is None else str(cell).strip()
            if s is None or s == "" or s.lower() == "nan":
                filled[j] = last
            else:
                last = s
                filled[j] = s
        header_rows.iloc[i, :] = filled

    # build column names
    colnames = []
    for j in range(raw.shape[1]):
        if j == 0:
            colnames.append("year")
            continue
        if j == 1:
            colnames.append("month")
            continue
        parts = []
        for i in range(start):
            cell = header_rows.iat[i, j]
            if cell is None:
                continue
            s = str(cell).strip()
            if s == "" or s.lower() == "nan":
                continue
            s = re.sub(r"\s+", " ", s.replace("\n", " ")).strip()
            parts.append(s)
        seen = set()
        uniq = []
        for p in parts:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        name = " | ".join(uniq) if uniq else f"col{j}"
        colnames.append(name)

    colnames = _make_unique(colnames)

    # data block
    data = raw.iloc[start:, :].copy()
    data.columns = colnames

    data["year"] = data["year"].ffill()
    mons = data["month"].astype(str).str.strip()
    month_num = mons.map(ROMAN_MAP)

    mask = month_num.notna() & data["year"].notna()
    data = data.loc[mask].copy()
    data["year"] = data["year"].astype(int)
    data["month_num"] = month_num.loc[mask].astype(int)
    data["date"] = pd.to_datetime(dict(year=data["year"], month=data["month_num"], day=1))

    series_cols = [c for c in data.columns if c not in ["year", "month", "month_num", "date"]]
    for c in series_cols:
        data[c] = pd.to_numeric(
            data[c].astype(str)
                 .str.replace(",", ".", regex=False)
                 .str.replace("–", "", regex=False),
            errors="coerce"
        )

    long = data.melt(id_vars=["date"], value_vars=series_cols, var_name="series", value_name="rate")
    long["currency"] = currency
    long["kind"] = kind
    long = long.dropna(subset=["rate"])
    return long

# -----------------------------
# 2) Forwards: build monthly F/S ratios
# -----------------------------
def load_monthly_forward_ratios(
    forwards_xlsx: Path,
    spot_col_regex=r"\bON\b",         # your sheet uses "ON" for spot
    tenors=("1W","1M","2M","3M","6M","9M","1Y"),
    agg="mean",                       # "mean" or "eom"
) -> pd.DataFrame:
    fwd = pd.read_excel(forwards_xlsx).copy()
    if "Name" in fwd.columns:
        fwd = fwd.rename(columns={"Name": "date"})
    fwd["date"] = pd.to_datetime(fwd["date"])

    def pick_col(pattern):
        cols = [c for c in fwd.columns if re.search(pattern, str(c))]
        return cols[0] if cols else None

    spot_col = pick_col(spot_col_regex)
    if spot_col is None:
        raise ValueError("Could not identify spot column (expected something matching regex like '\\bON\\b').")

    tenor_cols = {}
    for t in tenors:
        col = pick_col(rf"\b{re.escape(t)}\b")
        if col is not None:
            tenor_cols[t] = col

    if len(tenor_cols) == 0:
        raise ValueError("No forward tenor columns found (expected 1W/1M/2M/3M/6M/9M/1Y).")

    keep = ["date", spot_col] + list(tenor_cols.values())
    daily = fwd[keep].dropna(subset=[spot_col]).copy()
    daily = daily.rename(columns={spot_col: "S"})

    for t, col in tenor_cols.items():
        daily[t] = pd.to_numeric(daily[col], errors="coerce") / pd.to_numeric(daily["S"], errors="coerce")

    daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp()

    if agg == "mean":
        out = daily.groupby("month")[list(tenor_cols.keys())].mean().reset_index().rename(columns={"month": "date"})
    elif agg == "eom":
        out = daily.sort_values("date").groupby("month").tail(1)[["month"] + list(tenor_cols.keys())]
        out = out.rename(columns={"month": "date"}).reset_index(drop=True)
    else:
        raise ValueError("agg must be 'mean' or 'eom'")

    return out

# -----------------------------
# 3) Hedged conversion function
# -----------------------------
def eur_to_lvl_hedged_simple(r_eur_annual_pct: pd.Series, f_over_s: pd.Series, T_years: float) -> pd.Series:
    """
    R^{LVL,hedged(EUR)}_{t,T} = (1 + r^{EUR}_{t} * T) * (F/S) - 1
    r^{LVL,hedged(EUR)}_{t,T} ≈ R/T  (annualised simple)
    """
    r = pd.to_numeric(r_eur_annual_pct, errors="coerce") / 100.0
    fs = pd.to_numeric(f_over_s, errors="coerce")
    R = (1.0 + r * T_years) * fs - 1.0
    return (R / T_years) * 100.0

# -----------------------------
# 4) Load inputs
# -----------------------------
FORWARDS_XLSX = pick_path("latvian_forwards.xlsx")

DEP_LVL_CSV = pick_path("table21_rates_deposits_lats.csv")
DEP_EUR_CSV = pick_path("table21_rates_deposits_euro.csv")
DEP_USD_CSV = pick_path("table21_rates_deposits_dollar.csv")

deps_lvl = parse_table21_ffill_headers(DEP_LVL_CSV, "LVL", "deposits")
deps_eur = parse_table21_ffill_headers(DEP_EUR_CSV, "EUR", "deposits")
deps_usd = parse_table21_ffill_headers(DEP_USD_CSV, "USD", "deposits")

monthly_ratio = load_monthly_forward_ratios(FORWARDS_XLSX, agg="mean")

# -----------------------------
# 5) Select series: HH deposits, agreed maturity, up to 1Y
# -----------------------------
series_candidates = pd.Series(deps_eur["series"].unique())

hh_upto1_series = series_candidates[
    series_candidates.str.contains("Deposits from households", na=False) &
    series_candidates.str.contains("With agreed maturity", na=False) &
    series_candidates.str.contains("Up to 1 year", na=False)
]

if len(hh_upto1_series) != 1:
    raise ValueError(
        f"Expected exactly 1 HH up-to-1Y series, got {len(hh_upto1_series)}:\n{hh_upto1_series.tolist()}"
    )

HH_UPTO1 = hh_upto1_series.iloc[0]

def _wide_one(long_df: pd.DataFrame, series_name: str, cc: str) -> pd.DataFrame:
    out = long_df.loc[long_df["series"] == series_name, ["date", "rate"]].copy()
    return out.rename(columns={"rate": f"rate_{cc}"})

df = (
    _wide_one(deps_lvl, HH_UPTO1, "LVL")
    .merge(_wide_one(deps_eur, HH_UPTO1, "EUR"), on="date", how="outer")
    .merge(_wide_one(deps_usd, HH_UPTO1, "USD"), on="date", how="outer")
    .merge(monthly_ratio, on="date", how="left")
    .sort_values("date")
)

# -----------------------------
# 6) Compute hedged series + Δ and save CSV
# -----------------------------
TENOR = "3M"
T_YEARS = 0.25

if TENOR not in df.columns:
    raise ValueError(f"Monthly forward ratio for tenor {TENOR} not found. Available: {list(monthly_ratio.columns)}")

df[f"eur_hedged_to_lvl_{TENOR}"] = eur_to_lvl_hedged_simple(df["rate_EUR"], df[TENOR], T_YEARS)
df[f"delta_lvl_minus_hedged_{TENOR}"] = df["rate_LVL"] - df[f"eur_hedged_to_lvl_{TENOR}"]
df[f"fwd_premium_{TENOR}"] = (pd.to_numeric(df[TENOR], errors="coerce") - 1.0) / T_YEARS * 100.0

out_csv = OUTPUT_DIR / f"latvia_hh_deposits_upto1y_hedged_{TENOR}.csv"
df.to_csv(out_csv, index=False)
print(f"Saved monthly series: {out_csv}")

# -------------------------------------------------
# 7) Plots (STYLE MATCHES your merged script)
# -------------------------------------------------
# Colors consistent with your "overall" plots
hh_color = "#ff7f0e"        # same as your HH line / Euro in HH stacks
lats_color = "#4d5d6d"      # same as your Lats colour in stacks
other_fx_color = "#d3d3d3"  # same as your Other FX colour (light gray)
usd_color = "#1f77b4"       # use matplotlib default blue (close to your NFC color); OK for USD dashed
hedge_color = "#2ca02c"     # green for hedged benchmark (distinct)

# Ensure numeric for plotting
for c in ["rate_LVL", "rate_EUR", "rate_USD", f"eur_hedged_to_lvl_{TENOR}", f"delta_lvl_minus_hedged_{TENOR}"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# --- Plot A: Levels (LVL/EUR/USD + hedged EUR->LVL) ---
plt.figure(figsize=(12, 7))  # match your 12x7 default
plt.plot(df["date"], df["rate_LVL"], color=lats_color, lw=2.5, label="Lats (LVL)")
plt.plot(df["date"], df["rate_EUR"], color=hh_color, lw=2.5, label="Euro (EUR)")
plt.plot(df["date"], df["rate_USD"], color=usd_color, lw=2.0, ls="--", alpha=0.8, label="USD (Foreign)")
plt.plot(df["date"], df[f"eur_hedged_to_lvl_{TENOR}"], color=hedge_color, lw=2.5,
         label=f"EUR hedged into LVL ({TENOR} fwd)")

# Vertical markers (same style as your big script)
plt.axvline(PEG_DATE,   color="grey",  ls="-.", lw=1.5, label="Lats-EUR Peg")
plt.axvline(EURO_ADOPTION, color="black", ls="--", lw=2.0, label="Euro Adoption")
plt.axvline(CRISIS_DATE, color="black", ls=":",  lw=1.5, label="Banking Crisis")

plt.title("Latvia: Interest Rates — HH Deposits (Agreed maturity, up to 1Y)")
plt.ylabel("Interest Rate (%)")
plt.ylim(bottom=min(0.0, np.nanmin(df[["rate_LVL","rate_EUR","rate_USD",f"eur_hedged_to_lvl_{TENOR}"]].to_numpy()) - 1.0))
plt.grid(True)                         # your script uses plt.grid(True)
plt.legend(loc="lower right")          # your default location
plt.tight_layout()

out_levels = OUTPUT_DIR / f"fig6_hh_deposits_upto1y_levels_hedged_{TENOR}.png"
plt.savefig(out_levels, dpi=300)
plt.show()

# --- Plot B: Δ wedge = LVL - hedged(EUR->LVL) ---
plt.figure(figsize=(12, 7))  # match your 12x7 default
plt.plot(df["date"], df[f"delta_lvl_minus_hedged_{TENOR}"],
         color=hedge_color, lw=2.5, label=f"Δ = LVL − hedged(EUR→LVL) ({TENOR})")

# Shade to zero (same vibe as before, but keep light)
mask = df[f"delta_lvl_minus_hedged_{TENOR}"].notna()
x = df.loc[mask, "date"].values
y = df.loc[mask, f"delta_lvl_minus_hedged_{TENOR}"].values.astype(float)
plt.fill_between(x, y, 0.0, alpha=0.12, color=hedge_color)

plt.axhline(0, color="black", lw=1.0)
plt.axvline(PEG_DATE,   color="grey",  ls="-.", lw=1.5, label="Lats-EUR Peg")
plt.axvline(EURO_ADOPTION, color="black", ls="--", lw=2.0, label="Euro Adoption")
plt.axvline(CRISIS_DATE, color="black", ls=":",  lw=1.5, label="Banking Crisis")

plt.title(f"Latvia: Home Premium Net of FX — HH Deposits (Up to 1Y), Hedge Tenor {TENOR}")
plt.ylabel("Spread (pp)")
plt.ylim(
    np.nanmin(df[f"delta_lvl_minus_hedged_{TENOR}"]) - 1.0,
    np.nanmax(df[f"delta_lvl_minus_hedged_{TENOR}"]) + 1.0
)
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()

out_delta = OUTPUT_DIR / f"fig7_hh_deposits_upto1y_delta_hedged_{TENOR}.png"
plt.savefig(out_delta, dpi=300)
plt.show()

# --- (Optional) Plot C: forward premium (annualised) ---
plt.figure(figsize=(12, 7))
plt.plot(df["date"], df[f"fwd_premium_{TENOR}"], lw=2.5, label=f"Forward premium (annualised simple) {TENOR}")
plt.axhline(0, color="black", lw=1.0)
plt.axvline(PEG_DATE,   color="grey",  ls="-.", lw=1.5, label="Lats-EUR Peg")
plt.axvline(EURO_ADOPTION, color="black", ls="--", lw=2.0, label="Euro Adoption")
plt.axvline(CRISIS_DATE, color="black", ls=":",  lw=1.5, label="Banking Crisis")
plt.title(f"Latvia: LVL/EUR Forward Premium — {TENOR}")
plt.ylabel("pp (annualised)")
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()

out_fp = OUTPUT_DIR / f"fig8_forward_premium_{TENOR}.png"
plt.savefig(out_fp, dpi=300)
plt.show()

print(f"Saved plots:\n- {out_levels}\n- {out_delta}\n- {out_fp}")
