from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------
# Paths (your structure)
# -------------------------------------------------
BASE_DIR = Path(os.path.expanduser("~")) / "Dropbox" / "home_bias_monetary_policy"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output" / "motivating_facts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Files
# -------------------------------------------------
HR_EXCEL_NAME = "HR_deposit_sec.xlsx"  # Croatia Excel (place in DATA_DIR)
LT_CSV_NAME = "lithuania_bank_balance_sheet_agg-2025-11-27-15272.csv"  # Lithuania CSV (place in DATA_DIR)

# -------------------------------------------------
# Dates / FX
# -------------------------------------------------
CRO_EURO_ADOPTION_DATE = pd.Timestamp("2023-01-01")
HRK_PER_EUR = 7.53450  # fixed conversion rate at euro adoption
LT_EURO_ADOPTION_DATE = pd.Timestamp("2015-01-01")  # Lithuania adopted the euro

# Lithuania: treat "domestic currency" as LTL pre-2015, EUR post-2015
LT_DOMESTIC_PRE = "Litas"
LT_DOMESTIC_POST = "Euro"

LT_CURRENCY_SET = ["Total", "Litas", "Euro", "Other currencies"]


# =================================================
# Croatia helpers (Excel)
# =================================================
def _sheet_unit(df: pd.DataFrame) -> str | None:
    """Detect whether the sheet is in thousand HRK or thousand EUR (based on header text)."""
    for i in range(min(20, len(df))):
        for v in df.iloc[i].values:
            if isinstance(v, str) and "in thousand" in v.lower():
                txt = v.lower()
                if "hrk" in txt:
                    return "HRK"
                if "eur" in txt:
                    return "EUR"
    return None


def _detect_value_columns(df: pd.DataFrame) -> dict:
    """
    Detect which columns correspond to:
      Total, Euro, Foreign currencies, Kuna indexed to foreign currency
    """
    header_row = None
    for i in range(min(25, len(df))):
        row = df.iloc[i].astype(str)
        if any(x.strip().lower() == "total" for x in row.values if x != "nan"):
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not find the header row containing 'Total'.")

    headers = df.iloc[header_row].astype(str).str.strip()

    col_total = col_foreign = col_indexed = col_euro = None
    for col, h in headers.items():
        hl = h.lower()
        if hl == "total":
            col_total = col
        elif "foreign currencies" in hl:
            col_foreign = col
        elif "indexed" in hl and "foreign" in hl:
            col_indexed = col
        elif hl == "euro":
            col_euro = col

    return {
        "header_row": header_row,
        "total": col_total,
        "foreign": col_foreign,
        "indexed": col_indexed,
        "euro": col_euro,
    }


def _get_row_values(df: pd.DataFrame, cols: dict, label_regex: str) -> dict | None:
    """Find the row where the deposit type label lives and extract numeric values from detected columns."""
    label_col = df.columns[1]  # labels in 2nd column
    mask = df[label_col].astype(str).str.contains(label_regex, case=False, regex=True, na=False)
    if not mask.any():
        return None

    r = df.loc[mask].iloc[0]
    out: dict[str, float] = {}
    for k in ["total", "foreign", "indexed", "euro"]:
        c = cols.get(k)
        out[k] = pd.to_numeric(r[c], errors="coerce") if c is not None else np.nan
    return out


def _build_tidy_from_hr_excel(xlsx_path: Path) -> pd.DataFrame:
    """Croatia: returns tidy df: country, date, year, type, unit, total, foreign, indexed, euro."""
    xl = pd.ExcelFile(xlsx_path)

    type_labels = {
        "transaction": r"Total transaction accounts deposits",
        "savings": r"Total savings deposits",
        "time": r"Total time deposits",
        "total_deposits": r"TOTAL DEPOSITS",
    }

    rows: list[dict] = []
    for sheet in xl.sheet_names:
        if not str(sheet).isdigit():
            continue

        year = int(sheet)
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        cols = _detect_value_columns(df)
        unit = _sheet_unit(df)

        date = pd.Timestamp(year=year, month=12, day=31)

        for typ, regex in type_labels.items():
            vals = _get_row_values(df, cols, regex)
            if vals is None:
                continue
            rows.append(
                {
                    "country": "HR",
                    "date": date,
                    "year": year,
                    "type": typ,
                    "unit": unit,
                    "total": vals["total"],
                    "foreign": vals["foreign"],
                    "indexed": vals["indexed"],
                    "euro": vals["euro"],
                }
            )

    out = pd.DataFrame(rows).sort_values(["type", "date"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("Croatia: No data extracted. Check sheet names and labels in the Excel file.")
    return out


def _add_eur_components_hr(tidy: pd.DataFrame) -> pd.DataFrame:
    """
    Croatia: convert to EUR and create stacked components:
      - domestic_eur: HRK-only domestic component pre-2023, Euro component post-2023
      - foreign_fx_eur
      - indexed_fx_eur
    """
    d = tidy.copy()

    for col in ["total", "foreign", "indexed", "euro"]:
        d[col + "_eur"] = np.where(d["unit"] == "HRK", d[col] / HRK_PER_EUR, d[col])

    domestic_pre = (d["total"] - d["foreign"] - d["indexed"]).clip(lower=0) / HRK_PER_EUR
    domestic_post = d["euro"]
    d["domestic_eur"] = np.where(d["unit"] == "HRK", domestic_pre, domestic_post)

    d["foreign_fx_eur"] = d["foreign_eur"].fillna(0.0)
    d["indexed_fx_eur"] = d["indexed_eur"].fillna(0.0)

    d["share_domestic"] = d["domestic_eur"] / d["total_eur"]
    return d


# =================================================
# Lithuania helpers (CSV)
# =================================================
def _parse_period_to_timestamp(period: str) -> pd.Timestamp:
    p = str(period).strip()
    if len(p) == 7 and p[4] == "-":
        return pd.Timestamp(p + "-01") + pd.offsets.MonthEnd(0)
    return pd.to_datetime(p, errors="coerce")


def _read_lt_csv_robust(csv_path: Path) -> pd.DataFrame:
    """
    Robust CSV reader:
      - avoids C-engine tokenization failures
      - skips malformed lines
      - handles encoding glitches
    """
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        df = pd.read_csv(f, engine="python", on_bad_lines="skip")

    for col in [
        "reporting country",
        "institution",
        "instrument",
        "maturity",
        "stocks",
        "geography",
        "sector",
        "currency",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def _build_lt_currency_panel(
    csv_path: Path,
    *,
    instrument: str,
    sector: str,
    type_name: str,
    institution: str = "Other monetary financial institutions",
    geography: str = "Residents",
    maturity: str = "Total",
    currency_set: list[str] = LT_CURRENCY_SET,
) -> pd.DataFrame:
    """
    Generic: Lithuania series by currency for a given (instrument, sector).
    Returns a panel with columns: country, type, unit, date, currency, value_eur
    """
    df = _read_lt_csv_robust(csv_path)

    needed = {
        "date",
        "value",
        "power",
        "reporting country",
        "institution",
        "instrument",
        "maturity",
        "stocks",
        "geography",
        "sector",
        "currency",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"LT CSV missing columns: {sorted(missing)}")

    base = df[
        (df["reporting country"] == "Lithuania")
        & (df["institution"] == institution)
        & (df["geography"] == geography)
        & (df["sector"] == sector)
        & (df["stocks"].str.contains("Outstanding", case=False, na=False))
        & (df["maturity"] == maturity)
        & (df["instrument"] == instrument)
        & (df["currency"].isin(currency_set))
    ].copy()

    if base.empty:
        raise ValueError(
            f"LT: No rows after filtering for instrument='{instrument}', sector='{sector}'. "
            f"Try relaxing filters or check labels in the CSV."
        )

    base["date"] = base["date"].apply(_parse_period_to_timestamp)
    base["value"] = pd.to_numeric(base["value"], errors="coerce")
    base["power"] = pd.to_numeric(base["power"], errors="coerce")
    base["value_eur"] = base["value"] * base["power"]

    panel = (
        base.groupby(["date", "currency"], as_index=False)["value_eur"]
        .sum()
        .sort_values(["date", "currency"])
        .reset_index(drop=True)
    )
    panel["type"] = type_name
    panel["country"] = "LT"
    panel["unit"] = "EUR"
    return panel


def _lt_components_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Turn LT panel (date x currency) into components:
      total_eur, domestic_eur, foreign_fx_eur, euro_foreign_pre_eur, share_domestic
    """
    pv = (
        panel.pivot_table(index=["date"], columns="currency", values="value_eur", aggfunc="sum")
        .reset_index()
        .sort_values("date")
    )

    for c in LT_CURRENCY_SET:
        if c not in pv.columns:
            pv[c] = 0.0

    total_from_parts = pv["Litas"] + pv["Euro"] + pv["Other currencies"]
    pv["total_eur"] = np.where(pv["Total"].fillna(0) > 0, pv["Total"], total_from_parts)

    pv["domestic_eur"] = np.where(pv["date"] < LT_EURO_ADOPTION_DATE, pv[LT_DOMESTIC_PRE], pv[LT_DOMESTIC_POST])
    pv["euro_foreign_pre_eur"] = np.where(pv["date"] < LT_EURO_ADOPTION_DATE, pv["Euro"], 0.0)
    pv["foreign_fx_eur"] = pv["Other currencies"].fillna(0.0)

    pv["share_domestic"] = pv["domestic_eur"] / pv["total_eur"]

    pv["country"] = "LT"
    pv["unit"] = "EUR"
    return pv


def _lt_year_end(df: pd.DataFrame) -> pd.DataFrame:
    """Convert monthly LT series to year-end (last obs in each year) to match HR year-end."""
    tmp = df.copy()
    tmp["year"] = tmp["date"].dt.year
    out = tmp.sort_values("date").groupby("year", as_index=False).tail(1).drop(columns=["year"])
    return out


# =================================================
# Plotting
# =================================================
def _plot_stacked_hr(d: pd.DataFrame, typ: str, outpath: Path) -> None:
    sub = d[d["type"] == typ].sort_values("date")
    if sub.empty:
        return

    x = sub["date"]
    y_dom = sub["domestic_eur"].fillna(0).values
    y_fx = sub["foreign_fx_eur"].fillna(0).values
    y_idx = sub["indexed_fx_eur"].fillna(0).values

    plt.figure()
    plt.stackplot(
        x, y_dom, y_fx, y_idx,
        labels=["Domestic currency (HRK pre / EUR post)", "Foreign currencies", "Indexed to FX (HRK only)"],
    )
    plt.axvline(CRO_EURO_ADOPTION_DATE, linestyle="--")
    plt.title(f"HR: {typ.replace('_',' ').title()} (EUR)")
    plt.ylabel("EUR (thousands)")
    plt.xlabel("Year-end")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _plot_share(d: pd.DataFrame, title: str, adoption_date: pd.Timestamp, outpath: Path) -> None:
    if d.empty:
        return
    plt.figure()
    plt.plot(d["date"], d["share_domestic"], marker="o")
    plt.axvline(adoption_date, linestyle="--")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.ylabel("Share")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _plot_stacked_lt(lt: pd.DataFrame, title: str, outpath: Path) -> None:
    sub = lt.sort_values("date")
    if sub.empty:
        return

    x = sub["date"]
    y_dom = sub["domestic_eur"].fillna(0).values
    y_other = sub["foreign_fx_eur"].fillna(0).values
    y_euro_pre = sub["euro_foreign_pre_eur"].fillna(0).values

    plt.figure()
    plt.stackplot(
        x, y_dom, y_other, y_euro_pre,
        labels=["Domestic currency (LTL pre / EUR post)", "Other currencies", "Euro (foreign pre-2015)"],
    )
    plt.axvline(LT_EURO_ADOPTION_DATE, linestyle="--")
    plt.title(title)
    plt.ylabel("EUR")
    plt.xlabel("Date")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _plot_joint_share(hr: pd.DataFrame, lt_dep_y: pd.DataFrame, lt_loan_y: pd.DataFrame, outpath: Path) -> None:
    plt.figure()

    hr_sub = hr[hr["type"] == "total_deposits"].sort_values("date")
    if not hr_sub.empty:
        plt.plot(hr_sub["date"], hr_sub["share_domestic"], marker="o", label="HR: deposits (year-end)")
        plt.axvline(CRO_EURO_ADOPTION_DATE, linestyle="--")

    if not lt_dep_y.empty:
        plt.plot(lt_dep_y["date"], lt_dep_y["share_domestic"], marker="o", label="LT: HH deposits (year-end)")
        plt.axvline(LT_EURO_ADOPTION_DATE, linestyle="--")

    if not lt_loan_y.empty:
        plt.plot(lt_loan_y["date"], lt_loan_y["share_domestic"], marker="o", label="LT: NFC loans (year-end)")

    plt.ylim(0, 1.05)
    plt.title("Domestic currency share: Croatia vs Lithuania (deposits + NFC loans)")
    plt.ylabel("Share")
    plt.xlabel("Date")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# =================================================
# Run
# =================================================
def main() -> None:
    # ---- Croatia deposits (Excel)
    hr_xlsx = DATA_DIR / HR_EXCEL_NAME
    if not hr_xlsx.exists():
        raise FileNotFoundError(f"Missing {HR_EXCEL_NAME} in {DATA_DIR}")

    hr_tidy = _build_tidy_from_hr_excel(hr_xlsx)
    hr = _add_eur_components_hr(hr_tidy)

    out_hr_csv = OUTPUT_DIR / "croatia_deposits_by_type_currency_tidy.csv"
    hr[
        [
            "country", "date", "year", "type", "unit",
            "total_eur", "domestic_eur", "foreign_fx_eur", "indexed_fx_eur",
            "share_domestic",
        ]
    ].sort_values(["type", "date"]).to_csv(out_hr_csv, index=False)

    for typ in sorted(hr["type"].unique()):
        _plot_stacked_hr(hr, typ, OUTPUT_DIR / f"HR_{typ}_stacked.png")

    _plot_share(
        hr[hr["type"] == "total_deposits"].sort_values("date"),
        "HR: share domestic currency (total deposits)",
        CRO_EURO_ADOPTION_DATE,
        OUTPUT_DIR / "HR_total_deposits_share_domestic.png",
    )

    # ---- Lithuania (CSV): (1) HH deposits by currency, (2) NFC loans by currency
    lt_csv = DATA_DIR / LT_CSV_NAME
    if not lt_csv.exists():
        raise FileNotFoundError(f"Missing {LT_CSV_NAME} in {DATA_DIR}")

    lt_dep_panel = _build_lt_currency_panel(
        lt_csv,
        instrument="Deposits",
        sector="Households and NPIs serving households",
        type_name="hh_deposits_total",
    )
    lt_dep = _lt_components_from_panel(lt_dep_panel)
    lt_dep["type"] = "hh_deposits_total"

    lt_loan_panel = _build_lt_currency_panel(
        lt_csv,
        instrument="Loans",
        sector="Non-financial corporations",
        type_name="nfc_loans_total",
    )
    lt_loan = _lt_components_from_panel(lt_loan_panel)
    lt_loan["type"] = "nfc_loans_total"

    out_lt_dep_csv = OUTPUT_DIR / "lithuania_hh_deposits_total_by_currency_tidy.csv"
    lt_dep.sort_values("date").to_csv(out_lt_dep_csv, index=False)

    out_lt_loan_csv = OUTPUT_DIR / "lithuania_nfc_loans_total_by_currency_tidy.csv"
    lt_loan.sort_values("date").to_csv(out_lt_loan_csv, index=False)

    _plot_stacked_lt(lt_dep, "LT: Total household deposits by currency (EUR)", OUTPUT_DIR / "LT_hh_deposits_total_stacked.png")
    _plot_share(
        lt_dep.sort_values("date"),
        "LT: share domestic currency (total household deposits)",
        LT_EURO_ADOPTION_DATE,
        OUTPUT_DIR / "LT_hh_deposits_total_share_domestic.png",
    )

    _plot_stacked_lt(lt_loan, "LT: Total NFC loans by currency (EUR)", OUTPUT_DIR / "LT_nfc_loans_total_stacked.png")
    _plot_share(
        lt_loan.sort_values("date"),
        "LT: share domestic currency (total NFC loans)",
        LT_EURO_ADOPTION_DATE,
        OUTPUT_DIR / "LT_nfc_loans_total_share_domestic.png",
    )

    lt_dep_y = _lt_year_end(lt_dep)
    lt_loan_y = _lt_year_end(lt_loan)

    out_joint = OUTPUT_DIR / "HR_LT_joint_share_domestic_deposits_and_loans.png"
    _plot_joint_share(hr, lt_dep_y, lt_loan_y, out_joint)

    combined = pd.concat(
        [
            hr[["country", "date", "type", "total_eur", "domestic_eur", "foreign_fx_eur", "share_domestic"]],
            lt_dep[["country", "date", "type", "total_eur", "domestic_eur", "foreign_fx_eur", "share_domestic"]],
            lt_loan[["country", "date", "type", "total_eur", "domestic_eur", "foreign_fx_eur", "share_domestic"]],
        ],
        ignore_index=True,
    )
    combined_out = OUTPUT_DIR / "croatia_lithuania_tidy_for_plots_deposits_and_loans.csv"
    combined.sort_values(["country", "type", "date"]).to_csv(combined_out, index=False)

    print("Done.")
    print(f"Output folder: {OUTPUT_DIR}")
    print("Key outputs:")
    print(f"  - {out_hr_csv.name}")
    print(f"  - {out_lt_dep_csv.name}")
    print(f"  - {out_lt_loan_csv.name}")
    print(f"  - {combined_out.name}")
    print("Plots:")
    print("  - HR_<transaction|savings|time|total_deposits>_stacked.png")
    print("  - HR_total_deposits_share_domestic.png")
    print("  - LT_hh_deposits_total_stacked.png")
    print("  - LT_hh_deposits_total_share_domestic.png")
    print("  - LT_nfc_loans_total_stacked.png")
    print("  - LT_nfc_loans_total_share_domestic.png")
    print("  - HR_LT_joint_share_domestic_deposits_and_loans.png")


if __name__ == "__main__":
    main()
