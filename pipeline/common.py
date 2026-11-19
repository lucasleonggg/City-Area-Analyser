"""Shared loading + feature helpers for the EquiCity 2.0 pipeline.

Everything in pipeline/ that touches transaction-level data goes through
`load_transactions()` here, so every stage sees the same rows, the same
derived columns and the same category spellings.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PROCESSED_DIR = os.path.join(ROOT, "processed_data")
ANALYTICS_DIR = os.path.join(PROCESSED_DIR, "analytics")
DATA_PATH = os.path.join(PROCESSED_DIR, "hdb-all-cleaned.csv")

# URA planning regions. HDB towns map onto them cleanly; Lim Chu Kang is in
# URA's North region but has no current HDB population and only a handful of
# historical transactions.
REGION_OF_TOWN = {
    "Bishan": "Central", "Bukit Merah": "Central", "Bukit Timah": "Central",
    "Central Area": "Central", "Geylang": "Central", "Kallang/Whampoa": "Central",
    "Marine Parade": "Central", "Queenstown": "Central", "Toa Payoh": "Central",
    "Bedok": "East", "Pasir Ris": "East", "Tampines": "East",
    "Sembawang": "North", "Woodlands": "North", "Yishun": "North",
    "Lim Chu Kang": "North",
    "Ang Mo Kio": "North-East", "Hougang": "North-East", "Punggol": "North-East",
    "Sengkang": "North-East", "Serangoon": "North-East",
    "Bukit Batok": "West", "Bukit Panjang": "West", "Choa Chu Kang": "West",
    "Clementi": "West", "Jurong East": "West", "Jurong West": "West",
}

FLAT_TYPE_ROOMS = {
    "1-Room": 1, "2-Room": 2, "3-Room": 3, "4-Room": 4,
    "5-Room": 5, "Executive": 6, "Multi-Generation": 7,
}


def _storey_mid(s: str) -> float:
    a, b = s.split(" TO ")
    return (int(a) + int(b)) / 2


def load_transactions(min_year: int | None = None) -> pd.DataFrame:
    """All cleaned resale transactions with the derived columns every stage
    needs. Memory-lean: categoricals for the string columns."""
    usecols = [
        "month", "town", "flat_type", "block", "street_name", "storey_range",
        "floor_area_sqm", "flat_model", "lease_commence_date",
        "remaining_lease", "resale_price",
    ]
    df = pd.read_csv(
        DATA_PATH, usecols=usecols,
        dtype={
            "month": "string", "town": "category", "flat_type": "category",
            "block": "string", "street_name": "category",
            "storey_range": "category", "flat_model": "category",
            "floor_area_sqm": "float32", "lease_commence_date": "int16",
            "remaining_lease": "float32", "resale_price": "float32",
        },
    )
    # Canonical row order. merge.py + clean.py regenerate the same 982,011
    # rows as the committed file but in a different order, and some steps
    # (HGB bin sampling, bootstrap draws) depend on row order. Sorting here
    # makes every result independent of how the file happens to be ordered.
    df = df.sort_values(["month", "town", "street_name", "block", "flat_type", "storey_range",
                         "floor_area_sqm", "flat_model", "lease_commence_date", "remaining_lease",
                         "resale_price"], kind="mergesort",
                        # sort on VALUES, not category codes: category order can differ between files
                        key=lambda s: s.astype(str) if isinstance(s.dtype, pd.CategoricalDtype) else s
                        ).reset_index(drop=True)
    df["year"] = df["month"].str[:4].astype("int16")
    if min_year is not None:
        df = df[df["year"] >= min_year].copy()
    mon = df["month"].str[5:7].astype("int16")
    # Months since Jan 1990: an integer clock used for splits and the index.
    df["month_idx"] = ((df["year"] - 1990) * 12 + (mon - 1)).astype("int16")
    df["t"] = (df["year"] + (mon - 1) / 12).astype("float32")
    storey_map = {s: _storey_mid(s) for s in df["storey_range"].cat.categories}
    df["storey_mid"] = df["storey_range"].map(storey_map).astype("float32")
    df["region"] = df["town"].astype(str).map(REGION_OF_TOWN).astype("category")
    df["log_price"] = np.log(df["resale_price"]).astype("float32")
    df["block_key"] = (df["street_name"].astype(str) + " | " + df["block"].astype(str))
    df = df.drop(columns=["storey_range"]).reset_index(drop=True)
    return df


def month_label(month_idx: int) -> str:
    y, m = divmod(int(month_idx), 12)
    return f"{1990 + y}-{m + 1:02d}"
