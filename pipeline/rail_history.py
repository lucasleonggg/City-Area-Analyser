"""Time-correct rail accessibility (EquiCity 3.0).

v2 measured every flat's distance to TODAY's rail network, so a flat sold in
2010 was credited with a Downtown Line station that opened in 2015. Here each
transaction is matched to the nearest MRT/LRT station that was actually OPEN
in the month of sale.

Station table: raw_data/rail_stations_opening_dates.csv
  coordinates  LTA "MRT Station Exit" GeoJSON (Aug 2025), exits averaged per
               station; five stations labelled only by code in that file are
               mapped to their names (CC30 Keppel, CC31 Cantonment, CC32 Prince
               Edward Road, DT4 Hume, NE18 Punggol Coast).
  opened       Wikipedia "List of Singapore MRT stations" (Opened column, or
               the station article's infobox where the list cell is empty).
               LRT dates were hand-checked because the Wikipedia LRT table
               shares one date across a whole loop (it lists Sam Kee as 2005;
               it opened 29 Feb 2016). Interchanges take the EARLIEST opening
               of any rail service at that location.
  closed       Ten Mile Junction LRT closed 13 Jan 2019; it is not in the 2025
               LTA file, so it has no coordinates and is omitted entirely --
               a small understatement of access near Bukit Panjang 1999-2018.

The distance is straight-line to the station's exit centroid, not walking
distance or journey time.
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIONS = os.path.join(ROOT, "raw_data", "rail_stations_opening_dates.csv")


def load_stations() -> pd.DataFrame:
    s = pd.read_csv(STATIONS)
    s["open_month_idx"] = ((s["opened"].str[:4].astype(int) - 1990) * 12
                           + s["opened"].str[5:7].astype(int) - 1)
    s["close_month_idx"] = np.where(
        s["closed"].notna(),
        (s["closed"].fillna("2999-01").str[:4].astype(int) - 1990) * 12
        + s["closed"].fillna("2999-01").str[5:7].astype(int) - 1, 10 ** 6)
    return s


def _haversine_matrix(lat_a, lng_a, lat_b, lng_b):
    la, lb = np.radians(lat_a)[:, None], np.radians(lat_b)[None, :]
    dl = np.radians(lng_b)[None, :] - np.radians(lng_a)[:, None]
    a = np.sin((lb - la) / 2) ** 2 + np.cos(la) * np.cos(lb) * np.sin(dl / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def open_mask(stations: pd.DataFrame, month_idx: int) -> np.ndarray:
    """Station counted as open for a sale in month m if it opened in or
    before month m and had not closed."""
    return ((stations["open_month_idx"].to_numpy() <= month_idx)
            & (stations["close_month_idx"].to_numpy() > month_idx))


def nearest_open_distance(lat, lng, month_idx, stations: pd.DataFrame | None = None) -> np.ndarray:
    """Distance (km) from each point to the nearest station open in that
    point's month. Vectorised over network 'epochs' (distinct opening
    months), so 1M transactions take a few seconds."""
    stations = load_stations() if stations is None else stations
    lat, lng = np.asarray(lat, float), np.asarray(lng, float)
    month_idx = np.asarray(month_idx, int)
    out = np.full(len(lat), np.nan)
    ok = ~np.isnan(lat)
    if not ok.any():
        return out
    # unique points first (blocks repeat across many sales)
    pts = pd.DataFrame({"lat": lat[ok], "lng": lng[ok]})
    uniq, inv = np.unique(pts.to_numpy(), axis=0, return_inverse=True)
    inv = inv.ravel()
    D = _haversine_matrix(uniq[:, 0], uniq[:, 1], stations["lat"].to_numpy(), stations["lng"].to_numpy())
    events = np.unique(np.r_[stations["open_month_idx"].to_numpy(), stations["close_month_idx"].to_numpy()])
    epoch = np.searchsorted(events, month_idx[ok], side="right") - 1
    res = np.full(ok.sum(), np.nan)
    for e in np.unique(epoch):
        if e < 0:  # before the first station ever opened: no access (NaN)
            continue
        rows = epoch == e
        mask = open_mask(stations, events[e])
        if not mask.any():
            continue
        dmin = D[:, mask].min(axis=1)
        res[rows] = dmin[inv[rows]]
    out[ok] = res
    return out


def block_year_access(blocks: pd.DataFrame, years=range(1990, 2027)) -> pd.DataFrame:
    """Nearest open station per block at mid-year (June), every year."""
    s = load_stations()
    rows = []
    for y in years:
        m = (y - 1990) * 12 + 5
        d = nearest_open_distance(blocks["lat"], blocks["lng"], np.full(len(blocks), m), s)
        rows.append(pd.DataFrame({"block_key": blocks["block_key"].to_numpy(), "town": blocks["town"].to_numpy(),
                                  "year": y, "dist_km": d}))
    return pd.concat(rows, ignore_index=True)
