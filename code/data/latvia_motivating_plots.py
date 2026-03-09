import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(os.path.expanduser("~")) / "Dropbox" / "home_bias_monetary_policy"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output" / "motivating_facts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILE_14_HH_DEP = DATA_DIR / "table_14_HH.csv"
FILE_16_NFC_LOAN = DATA_DIR / "table_16_loans.csv"
FILE_20_HH_DEP = DATA_DIR / "table_20_resident_dep.csv"
FILE_20_NFC_LOAN = DATA_DIR / "table_20_resident_loans.csv"
FILE_HH_LOANS_MORT = DATA_DIR / "table_16_loans_HH.xlsx"

FILE_2010_2014 = DATA_DIR / "latvia_aggregate_bank_balance_sheets_2010_2014.csv"
FILE_2014_2025 = DATA_DIR / "latvia_aggregate_bank_balance_sheets_2014_2025.csv"

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def clean_val(val):
    if pd.isna(val) or str(val).strip() in ["nan", "", "None", "0"]:
        return 0.0
    clean = str(val).replace('"', '').replace(',', '')
    try:
        return float(clean)
    except Exception:
        return 0.0

def robust_date_parser(d):
    d = str(d).strip()
    if d in ["nan", "", "None"] or "filter" in d.lower():
        return pd.NaT
    res = pd.to_datetime(d, format='%m/%d/%y', errors='coerce')
    if pd.isna(res):
        res = pd.to_datetime(d, errors='coerce')
    return res

def clean_roman_label(lbl):
    """Strip trailing digits, asterisks, etc. from Roman numeral labels.
    Handles cases like 'V1' -> 'V', 'IX*' -> 'IX'."""
    import re
    return re.sub(r'[\d\*]+$', '', lbl.strip())

def safe_num(df, row, col):
    """Read a numeric cell, returning 0.0 for NaN (e.g. Non-profit FX cols)."""
    v = pd.to_numeric(df.iloc[row, col], errors='coerce')
    return 0.0 if pd.isna(v) else v

def parse_legacy_table(file_path, start_row, cols_map):
    df_raw = pd.read_csv(file_path, skiprows=start_row, header=None)
    roman = {
        "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
        "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12
    }
    data, curr_year = [], None

    for _, row in df_raw.iterrows():
        lbl = str(row[0]).strip()
        if lbl.isdigit() and len(lbl) == 4:
            curr_year = int(lbl)
            continue
        lbl_clean = clean_roman_label(lbl)
        if lbl_clean in roman and curr_year:
            entry = {'date': pd.Timestamp(curr_year, roman[lbl_clean], 1)}
            for idx, name in cols_map.items():
                entry[name] = clean_val(row[idx])
            data.append(entry)

    return pd.DataFrame(data)

# -------------------------------------------------
# Legacy lats amounts
# -------------------------------------------------
def get_lats_levels_legacy():
    hh = parse_legacy_table(FILE_14_HH_DEP, 10, {8: 'total', 9: 'amt_lats'})
    hh = hh[['date', 'total', 'amt_lats']].copy()
    hh['category'] = 'HH_Deposits'

    df16 = parse_legacy_table(
        FILE_16_NFC_LOAN, 9,
        {10: 'pub_t', 11: 'pub_l', 15: 'priv_t', 16: 'priv_l'}
    )
    df16['total'] = df16['pub_t'] + df16['priv_t']
    df16['amt_lats'] = df16['pub_l'] + df16['priv_l']
    nfc = df16[['date', 'total', 'amt_lats']].copy()
    nfc['category'] = 'NFC_Loans'

    return pd.concat([hh, nfc], ignore_index=True)[['date', 'category', 'total', 'amt_lats']]

# -------------------------------------------------
# Legacy extension for home-currency plot
# -------------------------------------------------
def get_home_currency_legacy():
    # HH deposits
    hh = parse_legacy_table(FILE_14_HH_DEP, 10, {8: 'total', 9: 'amt_lats'})
    hh['category'] = 'HH_Deposits'

    df20h = parse_legacy_table(FILE_20_HH_DEP, 12, {8: 'fx_pct', 9: 'eur_pct'})
    hh = pd.merge(hh, df20h, on='date', how='left')
    hh['res_eur'] = hh['total'] * (hh['eur_pct'] / 100.0)
    hh['res_fx'] = hh['total'] - hh['amt_lats'] - hh['res_eur']
    hh.loc[hh['res_fx'].abs() < 1e-8, 'res_fx'] = 0.0
    hh['res_total'] = hh['total']
    hh['home_currency_share'] = hh['amt_lats'] / hh['total']
    hh = hh[['date', 'category', 'res_eur', 'res_fx', 'res_total', 'amt_lats', 'home_currency_share']]

    # NFC loans
    df16 = parse_legacy_table(
        FILE_16_NFC_LOAN, 9,
        {10: 'pub_t', 11: 'pub_l', 15: 'priv_t', 16: 'priv_l'}
    )
    df16['total'] = df16['pub_t'] + df16['priv_t']
    df16['amt_lats'] = df16['pub_l'] + df16['priv_l']

    df20n = parse_legacy_table(FILE_20_NFC_LOAN, 11, {3: 'fx_pct', 4: 'eur_pct'})
    nfc = pd.merge(df16[['date', 'total', 'amt_lats']], df20n, on='date', how='left')
    nfc['category'] = 'NFC_Loans'
    nfc['res_eur'] = nfc['total'] * (nfc['eur_pct'] / 100.0)
    nfc['res_fx'] = nfc['total'] - nfc['amt_lats'] - nfc['res_eur']
    nfc.loc[nfc['res_fx'].abs() < 1e-8, 'res_fx'] = 0.0
    nfc['res_total'] = nfc['total']
    nfc['home_currency_share'] = nfc['amt_lats'] / nfc['total']
    nfc = nfc[['date', 'category', 'res_eur', 'res_fx', 'res_total', 'amt_lats', 'home_currency_share']]

    return pd.concat([hh, nfc], ignore_index=True)

# -------------------------------------------------
# Mortgage from HH loans legacy file (2003-07 to 2013-12)
#
# table_16_loans_HH.xlsx columns:
#   Col 13 (N) = total HH + Non-profit loans
#   Col 14 (O) = HH + Non-profit loans in lats
#   Col 15 (P) = mortgage share
#
# Calculation:
#   mort_total = mort_share * total
#   mort_lats  = mort_share * lats
#   home_currency_share = mort_lats / mort_total
# -------------------------------------------------
def get_hh_mortgage_share_legacy():
    if not FILE_HH_LOANS_MORT.exists():
        return pd.DataFrame()

    df_raw = pd.read_excel(FILE_HH_LOANS_MORT, header=None)

    roman = {
        "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
        "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12
    }
    data, curr_year = [], None

    for _, row in df_raw.iterrows():
        lbl = str(row[0]).strip()
        if lbl.isdigit() and len(lbl) == 4:
            curr_year = int(lbl)
            continue
        lbl_clean = clean_roman_label(lbl)
        if lbl_clean in roman and curr_year:
            total_hh = clean_val(row[13])
            lats_hh = clean_val(row[14])
            mort_share = clean_val(row[15])

            mort_total = mort_share * total_hh
            mort_lats = mort_share * lats_hh
            hcs = mort_lats / mort_total if mort_total > 0 else np.nan

            data.append({
                'date': pd.Timestamp(curr_year, roman[lbl_clean], 1),
                'category': 'HH_Mortgage',
                'hh_loans_total_legacy': total_hh,
                'hh_loans_lats_legacy': lats_hh,
                'mortgage_share': mort_share,
                'mort_total': mort_total,
                'mort_home': mort_lats,
                'home_currency_share': hcs
            })

    out = pd.DataFrame(data)
    out['date'] = pd.to_datetime(out['date'])
    return out

# -------------------------------------------------
# 2010-2014 file: residents
# -------------------------------------------------
def get_series_from_2010_2014():
    if not FILE_2010_2014.exists():
        return pd.DataFrame()

    df_raw = pd.read_csv(FILE_2010_2014, header=None)
    labels = df_raw.iloc[:, :5].copy()
    labels[0] = labels[0].ffill()
    labels[1] = labels[1].ffill()

    lats_legacy = get_lats_levels_legacy()
    euro_adopt = pd.Timestamp('2014-01-01')

    hh_dep_mask = (
        labels[0].astype(str).str.contains('LIABILITIES', na=False, case=False)
        & (labels[4].astype(str).str.strip() == 'Households')
    )
    nfc_pub_mask = (
        labels[0].astype(str).str.contains('ASSETS', na=False, case=False)
        & (labels[4].astype(str).str.strip() == 'Public non-financial corporations')
    )
    nfc_priv_mask = (
        labels[0].astype(str).str.contains('ASSETS', na=False, case=False)
        & (labels[4].astype(str).str.strip() == 'Private non-financial corporations')
    )

    results = []

    for start_col in range(5, df_raw.shape[1], 6):
        date_obj = robust_date_parser(df_raw.iloc[0, start_col])
        if pd.isna(date_obj):
            continue

        # HH deposits
        hh_rows = df_raw[hh_dep_mask]
        if not hh_rows.empty:
            row = hh_rows.iloc[0]

            euro_col = pd.to_numeric(row.iloc[start_col], errors='coerce')
            foreign_col = pd.to_numeric(row.iloc[start_col + 1], errors='coerce')
            total_col = pd.to_numeric(row.iloc[start_col + 2], errors='coerce')

            if date_obj < euro_adopt:
                lats_match = lats_legacy[
                    (lats_legacy['date'] == date_obj) &
                    (lats_legacy['category'] == 'HH_Deposits')
                ]
                amt_lats = lats_match['amt_lats'].iloc[0] if not lats_match.empty else np.nan
                res_eur = total_col - foreign_col - amt_lats
                home_currency_share = amt_lats / total_col
            else:
                amt_lats = 0.0
                res_eur = euro_col
                home_currency_share = res_eur / total_col

            results.append({
                'date': date_obj,
                'category': 'HH_Deposits',
                'res_eur': res_eur,
                'res_total': total_col,
                'amt_lats': amt_lats,
                'resident_euro_share': res_eur / total_col,
                'home_currency_share': home_currency_share,
                'source': '2010_2014'
            })

        # NFC loans
        pub_rows = df_raw[nfc_pub_mask]
        priv_rows = df_raw[nfc_priv_mask]

        if (not pub_rows.empty) and (not priv_rows.empty):
            pub = pub_rows.iloc[0]
            priv = priv_rows.iloc[0]

            euro_col = (
                pd.to_numeric(pub.iloc[start_col], errors='coerce') +
                pd.to_numeric(priv.iloc[start_col], errors='coerce')
            )
            foreign_col = (
                pd.to_numeric(pub.iloc[start_col + 1], errors='coerce') +
                pd.to_numeric(priv.iloc[start_col + 1], errors='coerce')
            )
            total_col = (
                pd.to_numeric(pub.iloc[start_col + 2], errors='coerce') +
                pd.to_numeric(priv.iloc[start_col + 2], errors='coerce')
            )

            if date_obj < euro_adopt:
                lats_match = lats_legacy[
                    (lats_legacy['date'] == date_obj) &
                    (lats_legacy['category'] == 'NFC_Loans')
                ]
                amt_lats = lats_match['amt_lats'].iloc[0] if not lats_match.empty else np.nan
                res_eur = total_col - foreign_col - amt_lats
                home_currency_share = amt_lats / total_col
            else:
                amt_lats = 0.0
                res_eur = euro_col
                home_currency_share = res_eur / total_col

            if pd.notna(res_eur) and abs(res_eur) < 1:
                res_eur = max(res_eur, 0.0)

            results.append({
                'date': date_obj,
                'category': 'NFC_Loans',
                'res_eur': res_eur,
                'res_total': total_col,
                'amt_lats': amt_lats,
                'resident_euro_share': res_eur / total_col,
                'home_currency_share': home_currency_share,
                'source': '2010_2014'
            })

    return pd.DataFrame(results)

# -------------------------------------------------
# 2014-2025 file: residents
# -------------------------------------------------
def get_series_from_2014_2025():
    if not FILE_2014_2025.exists():
        return pd.DataFrame()

    df_raw = pd.read_csv(FILE_2014_2025, header=None)
    df_raw[0] = df_raw[0].ffill()

    def matching_rows(side, breakdowns):
        mask_side = df_raw[0].astype(str).str.contains(side, na=False, case=False)
        mask_break = df_raw.iloc[:, 1:5].apply(
            lambda x: x.astype(str).str.strip().isin(breakdowns).any(), axis=1
        )
        return df_raw.index[mask_side & mask_break].tolist()

    hh_rows = matching_rows('LIABILITIES', ['Households'])
    nfc_rows = matching_rows('ASSETS', [
        'Public non-financial corporations',
        'Private non-financial corporations'
    ])
    if len(nfc_rows) == 0:
        nfc_rows = matching_rows('ASSETS', ['Private non-financial corporations'])

    results = []

    for start_col in range(5, df_raw.shape[1], 10):
        date_cell = df_raw.iloc[0, start_col] if pd.notna(df_raw.iloc[0, start_col]) else df_raw.iloc[1, start_col]
        date_obj = robust_date_parser(date_cell)
        if pd.isna(date_obj):
            continue

        if len(hh_rows) > 0:
            row_idx = hh_rows[0]
            res_eur = pd.to_numeric(df_raw.iloc[row_idx, start_col], errors='coerce')
            res_fx = pd.to_numeric(df_raw.iloc[row_idx, start_col + 1], errors='coerce')
            res_total = res_eur + res_fx

            results.append({
                'date': date_obj,
                'category': 'HH_Deposits',
                'res_eur': res_eur,
                'res_total': res_total,
                'amt_lats': 0.0,
                'resident_euro_share': res_eur / res_total,
                'home_currency_share': res_eur / res_total,
                'source': '2014_2025'
            })

        if len(nfc_rows) > 0:
            res_eur = pd.to_numeric(df_raw.iloc[nfc_rows, start_col], errors='coerce').sum()
            res_fx = pd.to_numeric(df_raw.iloc[nfc_rows, start_col + 1], errors='coerce').sum()
            res_total = res_eur + res_fx

            results.append({
                'date': date_obj,
                'category': 'NFC_Loans',
                'res_eur': res_eur,
                'res_total': res_total,
                'amt_lats': 0.0,
                'resident_euro_share': res_eur / res_total,
                'home_currency_share': res_eur / res_total,
                'source': '2014_2025'
            })

    return pd.DataFrame(results)

# -------------------------------------------------
# HH Mortgage from aggregate 2010-2014 file
# Sum rows 18 (Households) + 19 (Non-profit)
# 6-col blocks: [res_eur, res_fx, res_total, ...]
# Apply last mortgage share from legacy xlsx
# Only used from 2014-01 onward (pre-2014 comes from legacy xlsx)
# -------------------------------------------------
def get_hh_mort_from_2010_2014():
    if not FILE_2010_2014.exists():
        return pd.DataFrame()

    df_raw = pd.read_csv(FILE_2010_2014, header=None)
    mort_legacy = get_hh_mortgage_share_legacy()
    last_mort_share = mort_legacy['mortgage_share'].iloc[-1] if not mort_legacy.empty else np.nan

    results = []
    for start_col in range(5, df_raw.shape[1], 6):
        date_obj = robust_date_parser(df_raw.iloc[0, start_col])
        if pd.isna(date_obj) or date_obj < pd.Timestamp('2014-01-01'):
            continue

        # Sum Households (row 18) + Non-profit (row 19)
        euro_sum = safe_num(df_raw, 18, start_col) + safe_num(df_raw, 19, start_col)
        total_sum = safe_num(df_raw, 18, start_col + 2) + safe_num(df_raw, 19, start_col + 2)

        mort_total = last_mort_share * total_sum
        mort_euro = last_mort_share * euro_sum
        hcs = mort_euro / mort_total if mort_total > 0 else np.nan

        results.append({
            'date': date_obj,
            'category': 'HH_Mortgage',
            'mortgage_share': last_mort_share,
            'mort_total': mort_total,
            'mort_home': mort_euro,
            'home_currency_share': hcs,
            'source': '2010_2014'
        })

    return pd.DataFrame(results)

# -------------------------------------------------
# HH Mortgage from aggregate 2014-2025 file
# Sum rows 23 (Households) + 24 (Non-profit)
# 10-col blocks: [Latvia_EUR, Latvia_FX, Latvia_total, ...]
# Apply last mortgage share from legacy xlsx
# NaN in Non-profit FX treated as 0
# -------------------------------------------------
def get_hh_mort_from_2014_2025():
    if not FILE_2014_2025.exists():
        return pd.DataFrame()

    df_raw = pd.read_csv(FILE_2014_2025, header=None)
    mort_legacy = get_hh_mortgage_share_legacy()
    last_mort_share = mort_legacy['mortgage_share'].iloc[-1] if not mort_legacy.empty else np.nan

    results = []
    for start_col in range(5, df_raw.shape[1], 10):
        date_cell = df_raw.iloc[0, start_col] if pd.notna(df_raw.iloc[0, start_col]) else df_raw.iloc[1, start_col]
        date_obj = robust_date_parser(date_cell)
        if pd.isna(date_obj):
            continue

        # Sum Households (row 23) + Non-profit (row 24), NaN -> 0
        euro_sum = safe_num(df_raw, 23, start_col) + safe_num(df_raw, 24, start_col)
        fx_sum = safe_num(df_raw, 23, start_col + 1) + safe_num(df_raw, 24, start_col + 1)
        total_sum = euro_sum + fx_sum

        mort_total = last_mort_share * total_sum
        mort_euro = last_mort_share * euro_sum
        hcs = mort_euro / mort_total if mort_total > 0 else np.nan

        results.append({
            'date': date_obj,
            'category': 'HH_Mortgage',
            'mortgage_share': last_mort_share,
            'mort_total': mort_total,
            'mort_home': mort_euro,
            'home_currency_share': hcs,
            'source': '2014_2025'
        })

    return pd.DataFrame(results)

# -------------------------------------------------
# Non-residents from 2010-2014 file
# -------------------------------------------------
def get_nonres_from_2010_2014():
    if not FILE_2010_2014.exists():
        return pd.DataFrame()

    df_raw = pd.read_csv(FILE_2010_2014, header=None)
    labels = df_raw.iloc[:, :5].copy()
    labels[0] = labels[0].ffill()
    labels[1] = labels[1].ffill()

    hh_mask = (
        labels[0].astype(str).str.contains('LIABILITIES', na=False, case=False)
        & (labels[4].astype(str).str.strip() == 'Households')
    )
    nfc_pub_mask = (
        labels[0].astype(str).str.contains('ASSETS', na=False, case=False)
        & (labels[4].astype(str).str.strip() == 'Public non-financial corporations')
    )
    nfc_priv_mask = (
        labels[0].astype(str).str.contains('ASSETS', na=False, case=False)
        & (labels[4].astype(str).str.strip() == 'Private non-financial corporations')
    )

    results = []

    for start_col in range(5, df_raw.shape[1], 6):
        date_obj = robust_date_parser(df_raw.iloc[0, start_col])
        if pd.isna(date_obj):
            continue

        hh_rows = df_raw[hh_mask]
        if not hh_rows.empty:
            row = hh_rows.iloc[0]
            nonres_eur = pd.to_numeric(row.iloc[start_col + 3], errors='coerce')
            nonres_fx = pd.to_numeric(row.iloc[start_col + 4], errors='coerce')
            nonres_total = pd.to_numeric(row.iloc[start_col + 5], errors='coerce')

            results.append({
                'date': date_obj,
                'category': 'HH_Deposits',
                'nonres_eur_total': nonres_eur,
                'nonres_fx_total': nonres_fx,
                'nonres_total': nonres_total,
                'nonresident_euro_share': nonres_eur / nonres_total,
                'source': '2010_2014'
            })

        pub_rows = df_raw[nfc_pub_mask]
        priv_rows = df_raw[nfc_priv_mask]
        if (not pub_rows.empty) and (not priv_rows.empty):
            pub = pub_rows.iloc[0]
            priv = priv_rows.iloc[0]

            nonres_eur = (
                pd.to_numeric(pub.iloc[start_col + 3], errors='coerce') +
                pd.to_numeric(priv.iloc[start_col + 3], errors='coerce')
            )
            nonres_fx = (
                pd.to_numeric(pub.iloc[start_col + 4], errors='coerce') +
                pd.to_numeric(priv.iloc[start_col + 4], errors='coerce')
            )
            nonres_total = (
                pd.to_numeric(pub.iloc[start_col + 5], errors='coerce') +
                pd.to_numeric(priv.iloc[start_col + 5], errors='coerce')
            )

            results.append({
                'date': date_obj,
                'category': 'NFC_Loans',
                'nonres_eur_total': nonres_eur,
                'nonres_fx_total': nonres_fx,
                'nonres_total': nonres_total,
                'nonresident_euro_share': nonres_eur / nonres_total,
                'source': '2010_2014'
            })

    return pd.DataFrame(results)

# -------------------------------------------------
# Non-residents from 2014-2025 file
# -------------------------------------------------
def get_nonres_from_2014_2025():
    if not FILE_2014_2025.exists():
        return pd.DataFrame()

    df_raw = pd.read_csv(FILE_2014_2025, header=None)
    df_raw[0] = df_raw[0].ffill()

    def matching_rows(side, breakdowns):
        mask_side = df_raw[0].astype(str).str.contains(side, na=False, case=False)
        mask_break = df_raw.iloc[:, 1:5].apply(
            lambda x: x.astype(str).str.strip().isin(breakdowns).any(), axis=1
        )
        return df_raw.index[mask_side & mask_break].tolist()

    hh_rows = matching_rows('LIABILITIES', ['Households'])
    nfc_rows = matching_rows('ASSETS', [
        'Public non-financial corporations',
        'Private non-financial corporations'
    ])
    if len(nfc_rows) == 0:
        nfc_rows = matching_rows('ASSETS', ['Private non-financial corporations'])

    results = []

    for start_col in range(5, df_raw.shape[1], 10):
        date_cell = df_raw.iloc[0, start_col] if pd.notna(df_raw.iloc[0, start_col]) else df_raw.iloc[1, start_col]
        date_obj = robust_date_parser(date_cell)
        if pd.isna(date_obj):
            continue

        if len(hh_rows) > 0:
            row_idx = hh_rows[0]
            nonres_eur_total = (
                pd.to_numeric(df_raw.iloc[row_idx, start_col + 3], errors='coerce') +
                pd.to_numeric(df_raw.iloc[row_idx, start_col + 6], errors='coerce')
            )
            nonres_fx_total = (
                pd.to_numeric(df_raw.iloc[row_idx, start_col + 4], errors='coerce') +
                pd.to_numeric(df_raw.iloc[row_idx, start_col + 7], errors='coerce')
            )
            nonres_total = nonres_eur_total + nonres_fx_total

            results.append({
                'date': date_obj,
                'category': 'HH_Deposits',
                'nonres_eur_total': nonres_eur_total,
                'nonres_fx_total': nonres_fx_total,
                'nonres_total': nonres_total,
                'nonresident_euro_share': nonres_eur_total / nonres_total,
                'source': '2014_2025'
            })

        if len(nfc_rows) > 0:
            nonres_eur_total = (
                pd.to_numeric(df_raw.iloc[nfc_rows, start_col + 3], errors='coerce').sum() +
                pd.to_numeric(df_raw.iloc[nfc_rows, start_col + 6], errors='coerce').sum()
            )
            nonres_fx_total = (
                pd.to_numeric(df_raw.iloc[nfc_rows, start_col + 4], errors='coerce').sum() +
                pd.to_numeric(df_raw.iloc[nfc_rows, start_col + 7], errors='coerce').sum()
            )
            nonres_total = nonres_eur_total + nonres_fx_total

            results.append({
                'date': date_obj,
                'category': 'NFC_Loans',
                'nonres_eur_total': nonres_eur_total,
                'nonres_fx_total': nonres_fx_total,
                'nonres_total': nonres_total,
                'nonresident_euro_share': nonres_eur_total / nonres_total,
                'source': '2014_2025'
            })

    return pd.DataFrame(results)

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    print("Building resident and non-resident series...")

    euro_adopt = pd.Timestamp('2014-01-01')
    euro_peg = pd.Timestamp('2005-01-01')
    crisis = pd.Timestamp('2018-02-01')
    splice_date = pd.Timestamp('2014-12-01')
    end_nonres = pd.Timestamp('2024-12-01')

    df_old = get_series_from_2010_2014()
    df_new = get_series_from_2014_2025()

    # Resident euro share: 2010-2025
    df_res = pd.concat([
        df_old[df_old['date'] < splice_date],
        df_new[df_new['date'] >= splice_date]
    ], ignore_index=True)

    df_res['date'] = pd.to_datetime(df_res['date'], errors='coerce')
    for col in ['res_eur', 'res_total', 'amt_lats', 'resident_euro_share', 'home_currency_share']:
        df_res[col] = pd.to_numeric(df_res[col], errors='coerce')
    df_res = df_res.dropna(subset=['date', 'resident_euro_share', 'home_currency_share'])
    df_res = df_res.sort_values(['category', 'date']).reset_index(drop=True)

    # Home-currency legacy extension: HH deposits + NFC loans
    df_legacy_home = get_home_currency_legacy()
    df_legacy_home['date'] = pd.to_datetime(df_legacy_home['date'], errors='coerce')

    # ---- HH Mortgage ----
    # Legacy xlsx (2003-07 to 2013-12): self-consistent source
    # mort_total = mort_share * total, mort_lats = mort_share * lats
    mort_legacy_direct = get_hh_mortgage_share_legacy()
    if not mort_legacy_direct.empty:
        mort_legacy_direct['date'] = pd.to_datetime(mort_legacy_direct['date'])
        mort_legacy_part = mort_legacy_direct[['date', 'category', 'home_currency_share']].copy()
    else:
        mort_legacy_part = pd.DataFrame(columns=['date', 'category', 'home_currency_share'])

    # Aggregate files from 2014-01 onward:
    # Sum Households + Non-profit, apply last mortgage share to euro and total
    df_hh_mort_agg1 = get_hh_mort_from_2010_2014()   # 2014-01 to 2014-11
    df_hh_mort_agg2 = get_hh_mort_from_2014_2025()   # 2014-12 onward

    df_hh_mort_agg = pd.concat([
        df_hh_mort_agg1[df_hh_mort_agg1['date'] < splice_date] if not df_hh_mort_agg1.empty else pd.DataFrame(),
        df_hh_mort_agg2[df_hh_mort_agg2['date'] >= splice_date] if not df_hh_mort_agg2.empty else pd.DataFrame()
    ], ignore_index=True)

    if not df_hh_mort_agg.empty:
        df_hh_mort_agg['date'] = pd.to_datetime(df_hh_mort_agg['date'], errors='coerce')
        df_hh_mort_agg['home_currency_share'] = pd.to_numeric(df_hh_mort_agg['home_currency_share'], errors='coerce')
        df_hh_mort_agg = df_hh_mort_agg.dropna(subset=['date', 'home_currency_share'])

    # Combine: legacy xlsx through 2013-12, aggregate from 2014-01
    df_home = pd.concat([
        # HH Deposits + NFC Loans pre-2010 from legacy CSVs
        df_legacy_home[df_legacy_home['date'] < pd.Timestamp('2010-01-01')][['date', 'category', 'home_currency_share']],
        # HH Mortgage 2003-2013 from legacy xlsx (self-consistent)
        mort_legacy_part,
        # HH Deposits + NFC Loans 2010-2025 from aggregate files
        df_res[['date', 'category', 'home_currency_share']],
        # HH Mortgage 2014-2025 from aggregate files (HH + Non-profit summed)
        df_hh_mort_agg[['date', 'category', 'home_currency_share']] if not df_hh_mort_agg.empty else pd.DataFrame(columns=['date', 'category', 'home_currency_share'])
    ], ignore_index=True)

    df_home = df_home.dropna(subset=['date', 'home_currency_share'])
    df_home = df_home.sort_values(['category', 'date']).reset_index(drop=True)

    # Non-residents: 2010-2024
    df_nonres_old = get_nonres_from_2010_2014()
    df_nonres_new = get_nonres_from_2014_2025()

    df_nonres = pd.concat([
        df_nonres_old[df_nonres_old['date'] < splice_date],
        df_nonres_new[(df_nonres_new['date'] >= splice_date) & (df_nonres_new['date'] <= end_nonres)]
    ], ignore_index=True)

    df_nonres['date'] = pd.to_datetime(df_nonres['date'], errors='coerce')
    for col in ['nonres_eur_total', 'nonres_fx_total', 'nonres_total', 'nonresident_euro_share']:
        df_nonres[col] = pd.to_numeric(df_nonres[col], errors='coerce')
    df_nonres = df_nonres.dropna(subset=['date', 'nonresident_euro_share'])
    df_nonres = df_nonres.sort_values(['category', 'date']).reset_index(drop=True)

    colors = {
        'HH_Deposits': '#ff7f0e',
        'NFC_Loans': '#1f77b4',
        'HH_Mortgage': '#2ca02c'
    }
    labels = {
        'HH_Deposits': 'HH Deposits',
        'NFC_Loans': 'NFC Loans',
        'HH_Mortgage': 'HH Mortgage'
    }

    # Figure 1: Resident Euro Share
    plt.figure(figsize=(11, 6))
    for cat in ['HH_Deposits', 'NFC_Loans']:
        sub = df_res[df_res['category'] == cat].sort_values('date')
        plt.plot(sub['date'], sub['resident_euro_share'], label=labels[cat], color=colors[cat], lw=2)

    plt.axvline(euro_adopt, color='black', ls='--', lw=2, label='Euro Adoption (2014)')
    plt.axvline(crisis, color='0.25', ls=':', alpha=0.8, label='Banking Crisis')

    plt.title("Latvia: Euro Share of residents, 2010-2024")
    plt.ylabel("Euro Share")
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_resident_euro_share_2010_2025.png", dpi=300)
    plt.show()

    # Figure 2: Home Currency Share
    plt.figure(figsize=(11, 6))
    for cat in ['HH_Deposits', 'NFC_Loans', 'HH_Mortgage']:
        sub = df_home[df_home['category'] == cat].sort_values('date')
        if not sub.empty:
            plt.plot(sub['date'], sub['home_currency_share'], color=colors[cat], lw=2.5, label=labels[cat])

    plt.axvline(euro_peg, color='dimgray', ls='-.', alpha=0.8, label='Lats-EUR Peg (2005)')
    plt.axvline(euro_adopt, color='black', ls='--', lw=2, label='Euro Adoption (2014)')
    plt.axvline(crisis, color='0.25', ls=':', alpha=0.8, label='Banking Crisis')

    plt.title('Latvia: Home Currency Share')
    plt.ylabel('Home Currency Share')
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_home_currency_share_extended.png", dpi=300)
    plt.show()

      # Figure 3: Home Currency Share
    plt.figure(figsize=(11, 6))
    for cat in ['HH_Deposits', 'NFC_Loans']:
        sub = df_home[df_home['category'] == cat].sort_values('date')
        if not sub.empty:
            plt.plot(sub['date'], sub['home_currency_share'], color=colors[cat], lw=2.5, label=labels[cat])

    plt.axvline(euro_peg, color='dimgray', ls='-.', alpha=0.8, label='Lats-EUR Peg (2005)')
    plt.axvline(euro_adopt, color='black', ls='--', lw=2, label='Euro Adoption (2014)')
    plt.axvline(crisis, color='0.25', ls=':', alpha=0.8, label='Banking Crisis')

    plt.title('Latvia: Home Currency Share')
    plt.ylabel('Home Currency Share')
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_home_currency_share.png", dpi=300)
    plt.show()

    # Figure 4: Non-Residents Euro Share of Total
    plt.figure(figsize=(11, 6))
    for cat in ['HH_Deposits', 'NFC_Loans']:
        sub = df_nonres[df_nonres['category'] == cat].sort_values('date')
        plt.plot(sub['date'], sub['nonresident_euro_share'], label=labels[cat], color=colors[cat], lw=2)
    
    plt.axvline(euro_adopt, color='black', ls='--', lw=2, label='Euro Adoption (2014)')
    plt.axvline(crisis, color='0.25', ls=':', alpha=0.8, label='Banking Crisis')

    plt.title("Latvia: Euro Share of non-residents, 2010-2024")
    plt.ylabel("Euro Share")
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_nonresident_euro_share_2010_2024.png", dpi=300)
    plt.show()

    df_res.to_csv(OUTPUT_DIR / "latvia_resident_series_2010_2025.csv", index=False)
    df_home.to_csv(OUTPUT_DIR / "latvia_home_currency_series_extended.csv", index=False)
    df_nonres.to_csv(OUTPUT_DIR / "latvia_nonresident_series_2010_2024.csv", index=False)
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()