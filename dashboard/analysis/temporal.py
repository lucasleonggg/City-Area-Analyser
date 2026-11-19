"""Temporal analysis helpers (EquiCity 2.0, stage 4).

Everything here works on CONSTANT-QUALITY prices from the hedonic backbone,
so "Punggol grew fastest" means the price of the same reference flat rose
fastest there -- not that Punggol happened to sell newer flats later on.
"""

import numpy as np
import pandas as pd

from . import store


def national_monthly() -> pd.DataFrame:
    return store.table("hedonic_national_month.csv")


def national_quarterly_vs_official() -> pd.DataFrame:
    """EquiCity index vs HDB's official Resale Price Index (both 1Q2009=100)."""
    n = national_monthly().copy()
    y = n["month"].str[:4].astype(int)
    q = (n["month"].str[5:7].astype(int) - 1) // 3 + 1
    n["quarter"] = y.astype(str) + "-Q" + q.astype(str)
    g = n.groupby("quarter")["log_index"]
    ours = g.mean()[g.size() == 3]  # complete quarters only
    base = ours.loc["2009-Q1"]
    ours = (np.exp(ours - base) * 100).rename("equicity").reset_index()
    off = store.official_rpi().rename(columns={"index": "official_rpi"})
    out = ours.merge(off, on="quarter", how="left")
    out["year"] = out["quarter"].str[:4].astype(int)
    return out


def town_prices(flat_type="4-Room", min_n_town=0) -> pd.DataFrame:
    t = store.table("hedonic_town_year_type.csv")
    t = t[(t["flat_type"] == flat_type) & (t["n_town"] >= min_n_town)]
    return t


def growth_table(start: int, end: int, flat_type="4-Room", min_n_town=50) -> pd.DataFrame:
    """Constant-quality growth by town between two years, with a robust
    z-score against the cross-town distribution to flag unusual growth."""
    t = town_prices(flat_type)
    a = t[t["year"] == start].set_index("town")
    b = t[t["year"] == end].set_index("town")
    towns = a.index.intersection(b.index)
    towns = [x for x in towns if a.loc[x, "n_town"] >= min_n_town and b.loc[x, "n_town"] >= min_n_town]
    g = np.log(b.loc[towns, "price_ref"] / a.loc[towns, "price_ref"])
    yrs = max(end - start, 1)
    med = float(np.median(g))
    mad = float(np.median(np.abs(g - med))) * 1.4826
    out = pd.DataFrame({
        "town": towns, "region": b.loc[towns, "region"].values,
        f"price_{start}": a.loc[towns, "price_ref"].values,
        f"price_{end}": b.loc[towns, "price_ref"].values,
        "total_growth_pct": (np.exp(g.values) - 1) * 100,
        "annualised_pct": (np.exp(g.values / yrs) - 1) * 100,
        "relative_to_median_pp": (np.exp(g.values - med) - 1) * 100,
        "robust_z": (g.values - med) / mad if mad > 0 else 0.0,
    })
    out["flag"] = np.where(out["robust_z"] >= 2, "Unusually fast",
                           np.where(out["robust_z"] <= -2, "Unusually slow", ""))
    return out.sort_values("total_growth_pct", ascending=False).reset_index(drop=True)


def rolling_growth_flags(window=3, flat_type="4-Room", min_n_town=50) -> pd.DataFrame:
    """For every end year, which towns' trailing `window`-year growth was an
    outlier (|robust z| >= 2)? Long format for a heatmap."""
    t = town_prices(flat_type)
    years = sorted(t["year"].unique())
    rows = []
    for end in years:
        start = end - window
        if start < years[0]:
            continue
        g = growth_table(start, end, flat_type, min_n_town)
        g["end_year"] = end
        rows.append(g[["town", "end_year", "annualised_pct", "robust_z", "flag"]])
    return pd.concat(rows, ignore_index=True)


def lease_curves() -> pd.DataFrame:
    return store.table("lease_curve.csv")


def lease_band_trends() -> pd.DataFrame:
    return store.table("lease_band_year.csv")
