import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os

# =================================================
# 0) Paths & constants
# =================================================
BASE_DIR = Path(os.path.expanduser("~")) / "Dropbox" / "home_bias_monetary_policy"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output" / "motivating_facts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ADOPT_DATE = pd.Timestamp("2014-01-01")

plt.rcParams.update({
    "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "font.size": 11,
    "grid.alpha": 0.2
})

# =================================================
# 1) Load + clean (wide structure preserved)
# =================================================
def load_lb_interest_rates(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        print(f"Warning: {filepath} not found.")
        return pd.DataFrame()

    df = pd.read_csv(
        filepath,
        skiprows=6,
        na_values=["x", "-", "–", " ", "..", "…"],
        low_memory=False
    )

    df.columns.values[0] = "Year"
    df.columns.values[1] = "Month"

    # forward-fill Year only (rates are NOT filled)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").ffill()

    roman_map = {
        "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
        "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12
    }
    df["Month"] = df["Month"].astype(str).str.strip().map(roman_map)

    df = df.dropna(subset=["Year", "Month"])

    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(Day=1))
    return df.set_index("Date")

# =================================================
# 2) Excel-column helpers
# =================================================
def excel_col_to_idx(col: str) -> int:
    col = col.upper().strip()
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1

def pick_excel_col(df: pd.DataFrame, col_letter: str) -> pd.Series:
    j = excel_col_to_idx(col_letter)
    if j < 0 or j >= df.shape[1]:
        raise IndexError(f"Column {col_letter} -> index {j} out of bounds.")
    return df.iloc[:, j]

# =================================================
# 3) Plot helper (single series, LVL/EUR/USD)
# =================================================
def plot_three_currency_single(df_lvl, df_eur, df_usd, col_letter, title, filename):
    s_lvl = pick_excel_col(df_lvl, col_letter)
    s_eur = pick_excel_col(df_eur, col_letter)
    s_usd = pick_excel_col(df_usd, col_letter)

    all_dates = s_lvl.index.union(s_eur.index).union(s_usd.index).sort_values()
    s_lvl = s_lvl.reindex(all_dates)
    s_eur = s_eur.reindex(all_dates)
    s_usd = s_usd.reindex(all_dates)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    ax.plot(s_lvl.dropna(), label="Lats (Home < 2014)", color="navy", lw=2)
    ax.plot(s_eur.dropna(), label="Euro (Home > 2014)", color="orange", lw=2)
    ax.plot(s_usd.dropna(), label="USD (Foreign)", color="gray", linestyle="--", alpha=0.6)

    ax.axvline(ADOPT_DATE, color="black", lw=1.5, label="Euro Adoption")

    ax.set_title(title)
    ax.set_ylabel("Interest rate (%)")
    ax.set_xlabel("Year")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

# =================================================
# 4) Run: HH Deposits + NFC Loans (Table 21, currency)
# =================================================

# -----------------------------
# A) Households Deposits
# -----------------------------
d_lvl = load_lb_interest_rates(DATA_DIR / "table21_rates_deposits_lats.csv")
d_eur = load_lb_interest_rates(DATA_DIR / "table21_rates_deposits_euro.csv")
d_usd = load_lb_interest_rates(DATA_DIR / "table21_rates_deposits_dollar.csv")

# NEW business (C–G), skip notice over 3 months (H)
HH_NEW_DEPOSITS = [
    ("C", "Latvia: HH Deposits — New business (Overnight)", "hh_dep_new_overnight.png"),
    ("D", "Latvia: HH Deposits — New business (Maturity up to 1 year)", "hh_dep_new_upto1y.png"),
    ("E", "Latvia: HH Deposits — New business (Maturity 1–2 years)", "hh_dep_new_1to2y.png"),
    ("F", "Latvia: HH Deposits — New business (Maturity over 2 years)", "hh_dep_new_over2y.png"),
    ("G", "Latvia: HH Deposits — New business (Redeemable up to 3 months)", "hh_dep_new_notice_upto3m.png"),
]
for col, title, fname in HH_NEW_DEPOSITS:
    plot_three_currency_single(d_lvl, d_eur, d_usd, col, title, fname)

# OUTSTANDING amounts (Q–T), skip notice over 3 months (U)
HH_OUT_DEPOSITS = [
    ("Q", "Latvia: HH Deposits — Outstanding amounts (Overnight)", "hh_dep_out_overnight.png"),
    ("R", "Latvia: HH Deposits — Outstanding amounts (Maturity up to 2 years)", "hh_dep_out_upto2y.png"),
    ("S", "Latvia: HH Deposits — Outstanding amounts (Maturity over 2 years)", "hh_dep_out_over2y.png"),
    ("T", "Latvia: HH Deposits — Outstanding amounts (Redeemable up to 3 months)", "hh_dep_out_notice_upto3m.png"),
]
for col, title, fname in HH_OUT_DEPOSITS:
    plot_three_currency_single(d_lvl, d_eur, d_usd, col, title, fname)

# -----------------------------
# B) NFC Loans (same plotting style)
# -----------------------------
l_lvl = load_lb_interest_rates(DATA_DIR / "table21_rates_loans_lats.csv")
l_eur = load_lb_interest_rates(DATA_DIR / "table21_rates_loans_euro.csv")
l_usd = load_lb_interest_rates(DATA_DIR / "table21_rates_loans_dollar.csv")

# 1) NEW business
NFC_NEW_LOANS = [
    ("Y",  "Latvia: NFC Loans — New business (Bank overdraft)", "nfc_loans_new_overdraft.png"),
    ("AB", "Latvia: NFC Loans — New business (Other loans ≤0.25m, floating/up to 1y)", "nfc_loans_new_le025_floating_upto1y.png"),
    ("AD", "Latvia: NFC Loans — New business (Other loans ≤0.25m, fixation over 1y)", "nfc_loans_new_le025_over1y.png"),
    ("AF", "Latvia: NFC Loans — New business (Other loans 0.25–1m, floating/up to 1y)", "nfc_loans_new_025_1m_floating_upto1y.png"),
    ("AH", "Latvia: NFC Loans — New busines s (Other loans 0.25–1m, fixation over 1y)", "nfc_loans_new_025_1m_over1y.png"),
    ("AJ", "Latvia: NFC Loans — New business (Other loans >1m, floating/up to 1y)", "nfc_loans_new_gt1m_floating_upto1y.png"),
    ("AL", "Latvia: NFC Loans — New business (Other loans >1m, fixation over 1y)", "nfc_loans_new_gt1m_over1y.png"),
]
for col, title, fname in NFC_NEW_LOANS:
    plot_three_currency_single(l_lvl, l_eur, l_usd, col, title, fname)

# 2) OUTSTANDING amounts
NFC_OUT_LOANS = [
    ("AW", "Latvia: NFC Loans — Outstanding amounts (Maturity up to 1 year)", "nfc_loans_out_upto1y.png"),
    ("AX", "Latvia: NFC Loans — Outstanding amounts (Maturity 1–5 years)", "nfc_loans_out_1to5y.png"),
    ("AY", "Latvia: NFC Loans — Outstanding amounts (Maturity over 5 years)", "nfc_loans_out_over5y.png"),
]
for col, title, fname in NFC_OUT_LOANS:
    plot_three_currency_single(l_lvl, l_eur, l_usd, col, title, fname)

print("\nDone. All plots saved to:", OUTPUT_DIR)
