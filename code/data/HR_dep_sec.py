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
# Settings
# -------------------------------------------------
EXCEL_NAME = "HR_deposit_sec.xlsx"  # put this file into DATA_DIR
EURO_ADOPTION_DATE = pd.Timestamp("2023-01-01")
HRK_PER_EUR = 7.53450  # fixed conversion rate at euro adoption


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def _sheet_unit(df: pd.DataFrame) -> str | None:
    """Detects whether the sheet is in thousand HRK or thousand EUR (based on header text)."""
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
    """
    Find the row where the deposit type label lives (typically in column 1),
    and extract the numeric values from the detected value columns.
    """
    label_col = df.columns[1]  # these sheets typically store labels in the 2nd column
    mask = df[label_col].astype(str).str.contains(label_regex, case=False, regex=True, na=False)
    if not mask.any():
        return None

    r = df.loc[mask].iloc[0]
    out = {}
    for k in ["total", "foreign", "indexed", "euro"]:
        c = cols.get(k)
        if c is not None:
            out[k] = pd.to_numeric(r[c], errors="coerce")
        else:
            out[k] = np.nan
    return out


def _build_tidy_from_excel(xlsx_path: Path) -> pd.DataFrame:
    """
    Returns a tidy dataframe:date, year, type, unit, total, foreign, indexed, euro
    """
    xl = pd.ExcelFile(xlsx_path)

    # What we want from each sheet/year
    type_labels = {
        "transaction": r"Total transaction accounts deposits",
        "savings": r"Total savings deposits",
        "time": r"Total time deposits",
        "total_deposits": r"TOTAL DEPOSITS",
    }

    rows = []
    for sheet in xl.sheet_names:
        # only process sheets that look like years
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
        raise ValueError("No data extracted. Check sheet names and labels in the Excel file.")
    return out


def _add_eur_components(tidy: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all values to EUR and create 3 stacked components:
      - home_domestic_eur: HRK-only domestic component pre-2023, Euro component post-2023
      - foreign_fx_eur
      - indexed_fx_eur  (pre-2023 only)
    """
    d = tidy.copy()

    # Convert each raw column into EUR
    for col in ["total", "foreign", "indexed", "euro"]:
        d[col + "_eur"] = np.where(d["unit"] == "HRK", d[col] / HRK_PER_EUR, d[col])

    # Home/domestic component:
    # - If unit is HRK: infer home currency as total - foreign - indexed
    # - If unit is EUR (post-2023): use the "Euro" column as home currency
    home_pre = (d["total"] - d["foreign"] - d["indexed"]).clip(lower=0) / HRK_PER_EUR
    home_post = d["euro"]  # already in EUR-thousands when unit==EUR
    d["home_domestic_eur"] = np.where(d["unit"] == "HRK", home_pre, home_post)

    d["foreign_fx_eur"] = d["foreign_eur"].fillna(0)
    d["indexed_fx_eur"] = d["indexed_eur"].fillna(0)

    # Shares (useful for the “shift into euros” plot)
    d["share_domestic_legal_tender"] = d["home_domestic_eur"] / d["total_eur"]
    return d


def _plot_stacked_by_type(d: pd.DataFrame, typ: str, outpath: Path) -> None:
    sub = d[d["type"] == typ].sort_values("date")
    x = sub["date"]

    y_home = sub["home_domestic_eur"].fillna(0).values
    y_fx = sub["foreign_fx_eur"].fillna(0).values
    y_idx = sub["indexed_fx_eur"].fillna(0).values

    plt.figure()
    plt.stackplot(
        x,
        y_home,
        y_fx,
        y_idx,
        labels=[
            "Home currency (HRK pre / EUR post)",
            "Foreign currencies",
            "Indexed to FX (HRK only)",
        ],
    )
    plt.axvline(EURO_ADOPTION_DATE, linestyle="--")
    plt.title(f"Croatia deposits – {typ.replace('_', ' ').title()} (converted to EUR)")
    plt.ylabel("EUR (thousands)")
    plt.xlabel("Year-end")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _plot_domestic_share_total(d: pd.DataFrame, outpath: Path) -> None:
    sub = d[d["type"] == "total_deposits"].sort_values("date").copy()

    plt.figure()
    plt.plot(sub["date"], sub["share_domestic_legal_tender"], marker="o")
    plt.axvline(EURO_ADOPTION_DATE, linestyle="--")
    plt.ylim(0, 1.05)
    plt.title("Croatia – share of domestic deposits (HRK pre / EUR post)")
    plt.ylabel("Share")
    plt.xlabel("Year-end")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -------------------------------------------------
# Run
# -------------------------------------------------
def main() -> None:
    xlsx_path = DATA_DIR / EXCEL_NAME
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Missing {EXCEL_NAME} in {DATA_DIR}")

    tidy = _build_tidy_from_excel(xlsx_path)
    d = _add_eur_components(tidy)

    # Save tidy data
    out_csv = OUTPUT_DIR / "croatia_deposits_by_type_currency_tidy.csv"
    d[
        [
            "date",
            "year",
            "type",
            "unit",
            "total_eur",
            "home_domestic_eur",
            "foreign_fx_eur",
            "indexed_fx_eur",
            "share_domestic_legal_tender",
        ]
    ].sort_values(["type", "date"]).to_csv(out_csv, index=False)

    # Stacked plots by type
    for typ in sorted(d["type"].unique()):
        out_png = OUTPUT_DIR / f"HR_deposits_{typ}_stacked.png"
        _plot_stacked_by_type(d, typ, out_png)

    # “Shift into euros” share plot (total deposits)
    out_share = OUTPUT_DIR / "HR_deposits_total_share_domestic_legal_tender.png"
    _plot_domestic_share_total(d, out_share)

    print("Done.")
    print(f"Read:  {xlsx_path}")
    print(f"Wrote: {out_csv}")
    print("Plots:")
    print("  - HR_deposits_<transaction|savings|time|total_deposits>_stacked.png")
    print("  - HR_deposits_total_share_domestic_legal_tender.png")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
