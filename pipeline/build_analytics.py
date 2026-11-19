"""Build the analytical tables the EquiCity 2.0 dashboard reads.

    python pipeline/build_analytics.py

Writes to processed_data/analytics/ (all small CSVs, so the deployed
dashboard never touches the 132 MB transaction file):

  hedonic_town_year_type.csv  constant-quality price of a reference flat,
                              by town x flat type x year (1990-2026)
  hedonic_national_month.csv  national constant-quality price index, monthly,
                              rebased 2009-Q1 = 100 (same base as HDB's RPI)
  hedonic_region_year.csv     constant-quality 4-room price by URA region
  lease_curve.csv             price of a flat with L years left relative to
                              90 years left, re-estimated every year
  hedonic_fit_summary.csv     per-year fit diagnostics and key coefficients
  observed_town_type_year.csv raw medians (what actually transacted)
  lease_band_year.csv         median $/sqm by remaining-lease band and year
  recent_sales.csv            last 24 months of transactions (predictor UI)
  rail_access_town_year.csv   time-correct distance to the nearest OPEN rail
                              station, by town and year (block-weighted)
  rail_event_study.csv        descriptive event study: within-town price
                              residual of blocks that first came within 1 km
                              of an open station, relative to same-town
                              blocks that did not, years -6..+6
  rail_events.csv             the treated blocks and their event years
  block_premium.csv           constant-quality location premium per block,
                              with block-level Gi* hot/cold spots (needs
                              raw_data/hdb_block_coordinates.csv)

Runtime: ~1 minute, ~2 GB RAM.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ANALYTICS_DIR, REGION_OF_TOWN, load_transactions  # noqa: E402
from hedonic import (REF_AREA, REF_LEASE, REF_STOREY, fit_all_years,  # noqa: E402
                     fixed_town_weights, modal_models,
                     national_monthly_log_index, weighted_town_effect)

LEASE_GRID = np.arange(40, 100, 1.0)
COORDS = os.path.join(os.path.dirname(ANALYTICS_DIR), "..", "raw_data", "hdb_block_coordinates.csv")
BLOCK_YEARS = 3          # pool the last 3 calendar years of sales per block
BLOCK_MIN_SALES = 5
GI_BAND_KM = 1.0


EVENT_KM = 1.0
EVENT_WINDOW = 6


def rail_analysis(df, fits):
    """Time-correct rail access by town-year, and a descriptive event study
    of blocks that gained a station within 1 km."""
    from rail_history import block_year_access
    c = pd.read_csv(COORDS)
    c = c[c["coord_ok"]].copy()
    c["block_key"] = c["street_name"].astype(str) + " | " + c["block"].astype(str)
    acc = block_year_access(c[["block_key", "town", "lat", "lng"]])
    built = df.groupby("block_key")["lease_commence_date"].min()
    # only blocks that existed in that year count towards town access
    ex = acc[acc["year"] >= acc["block_key"].map(built).fillna(9999)]
    town = (ex.groupby(["town", "year"])
               .agg(mean_dist_km=("dist_km", "mean"), median_dist_km=("dist_km", "median"),
                    share_within_500m=("dist_km", lambda d: (d <= 0.5).mean() * 100),
                    share_within_1km=("dist_km", lambda d: (d <= 1.0).mean() * 100),
                    n_blocks=("dist_km", "size")).reset_index())
    nat = (ex.groupby("year").agg(mean_dist_km=("dist_km", "mean"), median_dist_km=("dist_km", "median"),
                                   share_within_500m=("dist_km", lambda d: (d <= 0.5).mean() * 100),
                                   share_within_1km=("dist_km", lambda d: (d <= 1.0).mean() * 100),
                                   n_blocks=("dist_km", "size")).reset_index().assign(town="All HDB blocks"))
    pd.concat([town, nat]).to_csv(os.path.join(ANALYTICS_DIR, "rail_access_town_year.csv"), index=False)

    # block-year within-town residuals from each year's hedonic fit
    parts = []
    for y, g in df.groupby("year"):
        r = g["log_price"].to_numpy(float) - fits[int(y)].predict_log(g)
        parts.append(pd.DataFrame({"block_key": g["block_key"].to_numpy(), "year": int(y), "resid": r}))
    br = (pd.concat(parts).groupby(["block_key", "year"])["resid"].agg(["mean", "size"])
            .rename(columns={"mean": "resid", "size": "n"}).reset_index())

    piv = acc.pivot(index="block_key", columns="year", values="dist_km")
    years = piv.columns.to_numpy()
    within = piv.to_numpy() <= EVENT_KM
    first = np.where(within.any(axis=1), years[np.argmax(within, axis=1)], 0)
    ev = pd.DataFrame({"block_key": piv.index, "event_year": first})
    ev = ev[(ev["event_year"] > years.min())].merge(c[["block_key", "town"]], on="block_key")
    ev = ev[ev["event_year"].between(1996, 2022)]  # need years on both sides
    # Only blocks that already existed (and were resellable) well before the
    # station opened, so the pre-period is the same flats, not an empty field.
    ev = ev[ev["block_key"].map(built) <= ev["event_year"] - 8]
    ev.to_csv(os.path.join(ANALYTICS_DIR, "rail_events.csv"), index=False)

    br = br.merge(c[["block_key", "town"]], on="block_key")
    br = br.merge(acc[["block_key", "year", "dist_km"]], on=["block_key", "year"])
    evmap = dict(zip(ev["block_key"], ev["event_year"]))
    all_first = dict(zip(piv.index, np.where(within.any(axis=1), first, 9999)))
    br["first_within"] = br["block_key"].map(all_first)
    # controls: block-years more than 1 km from an open station and at least
    # EVENT_WINDOW years away from ever getting one
    ctrl = br[(br["dist_km"] > EVENT_KM) & ((br["first_within"] - br["year"]) > EVENT_WINDOW)]
    cm = (ctrl.assign(w=ctrl["resid"] * ctrl["n"]).groupby(["town", "year"])
              .agg(wsum=("w", "sum"), n=("n", "sum")))
    cm = (cm["wsum"] / cm["n"]).rename("ctrl").reset_index()
    tr = br[br["block_key"].isin(evmap)].copy()
    tr["k"] = tr["year"] - tr["block_key"].map(evmap)
    tr = tr[tr["k"].between(-EVENT_WINDOW, EVENT_WINDOW)].merge(cm, on=["town", "year"])
    tr["rel"] = tr["resid"] - tr["ctrl"]
    blocks = tr["block_key"].unique()
    rng = np.random.default_rng(0)

    def path(sub):
        p = sub.groupby("k")["rel"].mean()
        return p - p.get(-1, 0.0)

    base = path(tr)
    by_block = {b: g for b, g in tr.groupby("block_key")}
    boots = []
    for _ in range(300):
        pick = rng.choice(blocks, len(blocks), replace=True)
        boots.append(path(pd.concat([by_block[b] for b in pick])).reindex(base.index))
    B = np.vstack(boots)
    es = pd.DataFrame({"k": base.index, "effect_log": base.values,
                       "effect_pct": (np.exp(base.values) - 1) * 100,
                       "lo95_pct": (np.exp(np.nanpercentile(B, 2.5, axis=0)) - 1) * 100,
                       "hi95_pct": (np.exp(np.nanpercentile(B, 97.5, axis=0)) - 1) * 100,
                       "n_blocks": tr.groupby("k")["block_key"].nunique().reindex(base.index).values})
    es.to_csv(os.path.join(ANALYTICS_DIR, "rail_event_study.csv"), index=False)
    print(f"rail: {len(ev)} blocks first within {EVENT_KM} km of a station 1996-2022; "
          f"event study k=+3 {es.set_index('k').loc[3, 'effect_pct']:.2f}% "
          f"[{es.set_index('k').loc[3, 'lo95_pct']:.2f}, {es.set_index('k').loc[3, 'hi95_pct']:.2f}]")


def block_premium(df, fits, weights):
    """Within-town residual of each block's sales from that year's hedonic
    fit, plus its town's location premium, pooled over recent years. Then
    block-level Gi* with a fixed distance band."""
    from scipy import sparse
    from scipy.spatial import cKDTree
    from scipy.stats import norm

    last_year = int(df["year"].max())
    recent = df[df["year"] > last_year - BLOCK_YEARS].copy()
    parts = []
    for y, g in recent.groupby("year"):
        f = fits[int(y)]
        resid = g["log_price"].to_numpy(float) - f.predict_log(g)
        nat = weighted_town_effect(f, weights)
        town_rel = g["town"].astype(str).map(lambda t: f.town_effect(t) - nat).to_numpy()
        parts.append(pd.DataFrame({"block_key": g["block_key"].to_numpy(), "town": g["town"].astype(str).to_numpy(),
                                   "resid": resid, "town_rel": town_rel}))
    r = pd.concat(parts)
    b = r.groupby("block_key").agg(town=("town", "first"), n=("resid", "size"),
                                   within_town=("resid", "mean"), town_rel=("town_rel", "mean"))
    b = b[b["n"] >= BLOCK_MIN_SALES].copy()
    b["log_premium"] = b["within_town"] + b["town_rel"]
    b["premium_pct"] = (np.exp(b["log_premium"]) - 1) * 100
    b["within_town_pct"] = (np.exp(b["within_town"]) - 1) * 100
    c = pd.read_csv(COORDS)
    c = c[c["coord_ok"]]
    c["block_key"] = c["street_name"].astype(str) + " | " + c["block"].astype(str)
    b = b.join(c.set_index("block_key")[["lat", "lng", "postal"]], how="inner").reset_index()

    # local Gi* (self-inclusive binary distance band), analytic z
    xy = np.column_stack([np.radians(b["lat"]) * 6371, np.radians(b["lng"]) * 6371 * np.cos(np.radians(1.35))])
    tree = cKDTree(xy)
    pairs = tree.query_pairs(GI_BAND_KM, output_type="ndarray")
    n = len(b)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)])
    W = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    x = b["log_premium"].to_numpy(float)
    xb, s = x.mean(), x.std(ddof=0)
    sw = np.asarray(W.sum(axis=1)).ravel()
    num = W @ x - xb * sw
    den = s * np.sqrt((n * sw - sw ** 2) / (n - 1))
    z = np.where(den > 0, num / den, 0.0)
    p = 2 * (1 - norm.cdf(np.abs(z)))
    order = np.argsort(p)
    ranked = np.minimum.accumulate((p[order] * n / (np.arange(n) + 1))[::-1])[::-1]
    fdr = np.empty(n); fdr[order] = np.minimum(ranked, 1)
    b["gi_z"], b["p_fdr"], b["neighbours_1km"] = z.round(3), fdr, (sw - 1).astype(int)
    b["hotspot"] = np.where(fdr >= 0.05, "Not significant", np.where(z > 0, "Hot spot", "Cold spot"))

    # global Moran's I, same band without self, permutation p (199)
    W0 = W - sparse.identity(n, format="csr")
    zc = x - xb
    S0 = W0.sum()
    I = n / S0 * (zc @ (W0 @ zc)) / (zc @ zc)
    rng = np.random.default_rng(42)
    perm = np.array([n / S0 * (v @ (W0 @ v)) / (v @ v) for v in (rng.permutation(zc) for _ in range(199))])
    moran = {"I": float(I), "p_sim": float((np.sum(perm >= I) + 1) / 200), "n_blocks": int(n),
             "band_km": GI_BAND_KM, "years": f"{last_year - BLOCK_YEARS + 1}-{last_year}"}
    return b, moran


def main():
    t0 = time.time()
    os.makedirs(ANALYTICS_DIR, exist_ok=True)
    df = load_transactions()
    print(f"loaded {len(df):,} rows in {time.time() - t0:.1f}s")

    fits = fit_all_years(df)
    ref_models = modal_models(df)
    weights = fixed_town_weights(df)
    print(f"fitted {len(fits)} yearly hedonic models in {time.time() - t0:.1f}s")
    print("reference flat models:", ref_models)

    # --- 1. constant-quality town x type x year --------------------------
    rows = []
    for y, f in fits.items():
        mm = f.mean_month_effect()
        for town in f.towns:
            te = f.town_effect(town)
            for ft in REF_AREA:
                ap = f.attr_part(ft, ref_models.get(ft, "Model A"))
                if np.isnan(ap):
                    continue
                lp = f.b("const") + mm + te + ap
                rows.append({
                    "year": y, "town": town, "region": REGION_OF_TOWN.get(town),
                    "flat_type": ft, "log_price_ref": round(lp, 5),
                    "price_ref": round(float(np.exp(lp))),
                    "town_effect": round(te, 5), "town_se": round(f.town_se(town), 5),
                    "n_town": f.n_by_town.get(town, 0),
                    "n_cell": f.n_by_town_type.get((town, ft), 0),
                })
    tt = pd.DataFrame(rows)
    # Location premium: town effect relative to the fixed-weight national
    # average town effect, as a percentage.
    nat_te = {y: weighted_town_effect(f, weights) for y, f in fits.items()}
    tt["location_premium_pct"] = (
        (np.exp(tt["town_effect"] - tt["year"].map(nat_te)) - 1) * 100).round(2)
    tt.to_csv(os.path.join(ANALYTICS_DIR, "hedonic_town_year_type.csv"), index=False)

    # --- 2. national monthly index, 2009-Q1 = 100 -------------------------
    li = national_monthly_log_index(fits, weights, ref_models)
    base = li.loc[[(2009 - 1990) * 12 + k for k in range(3)]].mean()
    nat = pd.DataFrame({
        "month_idx": li.index.astype(int),
        "month": [f"{1990 + m // 12}-{m % 12 + 1:02d}" for m in li.index],
        "log_index": li.values.round(5),
        "price_ref_4room": np.exp(li.values).round(0),
        "index_2009q1_100": (np.exp(li.values - base) * 100).round(2),
    })
    nat.to_csv(os.path.join(ANALYTICS_DIR, "hedonic_national_month.csv"), index=False)

    # --- 3. region x year (4-room reference) ------------------------------
    rrows = []
    for y, f in fits.items():
        mm = f.mean_month_effect()
        ap = f.attr_part("4-Room", ref_models.get("4-Room", "Model A"))
        for region in sorted(set(REGION_OF_TOWN.values())):
            towns = [t for t, r in REGION_OF_TOWN.items() if r == region]
            if not any(t in f.towns for t in towns):
                continue
            te = weighted_town_effect(f, weights, towns)
            rrows.append({"year": y, "region": region,
                          "price_ref_4room": round(float(np.exp(f.b("const") + mm + te + ap)))})
    pd.DataFrame(rrows).to_csv(os.path.join(ANALYTICS_DIR, "hedonic_region_year.csv"), index=False)

    # --- 4. lease curve ---------------------------------------------------
    lrows = []
    for y, f in fits.items():
        sub = df.loc[df["year"] == y, "remaining_lease"]
        lo, hi = float(sub.quantile(0.01)), float(sub.quantile(0.99))
        for L in LEASE_GRID:
            lrows.append({"year": y, "remaining_lease": L,
                          "rel_price_vs_90": round(f.lease_relative(L, 90.0), 4),
                          "in_support": bool(lo <= L <= hi)})
    pd.DataFrame(lrows).to_csv(os.path.join(ANALYTICS_DIR, "lease_curve.csv"), index=False)

    # --- 5. fit summary ---------------------------------------------------
    srows = []
    for y, f in fits.items():
        srows.append({
            "year": y, "n": f.n, "r2": round(f.r2, 4),
            "resid_sd_log": round(float(np.sqrt(f.sigma2)), 4),
            "area_elasticity": round(f.b("log_area"), 4),
            "lease_60_vs_90": round(f.lease_relative(60, 90), 4),
            "lease_80_vs_90": round(f.lease_relative(80, 90), 4),
            "storey_14_vs_5_pct": round(
                (np.exp(f.b("storey") * 9 + f.b("storey2") * (14 ** 2 - 5 ** 2) / 100) - 1) * 100, 2),
            "n_towns": len(f.towns),
        })
    pd.DataFrame(srows).to_csv(os.path.join(ANALYTICS_DIR, "hedonic_fit_summary.csv"), index=False)

    # --- 6. observed medians ---------------------------------------------
    obs = (df.groupby(["year", "town", "flat_type"], observed=True)
             .agg(median_price=("resale_price", "median"),
                  n=("resale_price", "size"),
                  median_area=("floor_area_sqm", "median"),
                  median_lease=("remaining_lease", "median"))
             .reset_index())
    obs["region"] = obs["town"].astype(str).map(REGION_OF_TOWN)
    obs[["town", "flat_type"]] = obs[["town", "flat_type"]].astype(str)
    obs = obs.sort_values(["year", "town", "flat_type"]).reset_index(drop=True)  # file-order independent
    obs.to_csv(os.path.join(ANALYTICS_DIR, "observed_town_type_year.csv"), index=False)

    # --- 7. lease bands ----------------------------------------------------
    lb = df[["year", "remaining_lease", "resale_price", "floor_area_sqm"]].copy()
    lb["band"] = pd.cut(lb["remaining_lease"], [0, 50, 60, 70, 80, 90, 100],
                        labels=["<50", "50-60", "60-70", "70-80", "80-90", "90+"])
    lb["psm"] = lb["resale_price"] / lb["floor_area_sqm"]
    (lb.groupby(["year", "band"], observed=True)
       .agg(median_psm=("psm", "median"), n=("psm", "size")).reset_index()
       .to_csv(os.path.join(ANALYTICS_DIR, "lease_band_year.csv"), index=False))

    # --- 8. recent sales (predictor comparables) ---------------------------
    last = int(df["month_idx"].max())
    rs = df[df["month_idx"] > last - 24][["month", "town", "street_name", "block", "flat_type",
                                          "flat_model", "floor_area_sqm", "storey_mid",
                                          "remaining_lease", "lease_commence_date", "resale_price"]]
    rs.to_csv(os.path.join(ANALYTICS_DIR, "recent_sales.csv"), index=False)

    # --- 9. block-level location premium + hot spots -----------------------
    moran = None
    if os.path.exists(COORDS):
        rail_analysis(df, fits)
        bp, moran = block_premium(df, fits, weights)
        bp.to_csv(os.path.join(ANALYTICS_DIR, "block_premium.csv"), index=False)
        print(f"block premium: {len(bp)} blocks; Moran's I {moran['I']:.3f} (p={moran['p_sim']:.3f})")

    meta = {
        "block_morans_i": moran,
        "built_utc": pd.Timestamp.now("UTC").isoformat(),
        "n_transactions": int(len(df)),
        "first_month": str(df["month"].min()), "last_month": str(df["month"].max()),
        "reference_flat": {"remaining_lease": REF_LEASE, "storey": REF_STOREY,
                           "floor_area_by_type": REF_AREA, "flat_model_by_type": ref_models},
        "town_weights_years": "2015-2025 transaction shares (fixed)",
    }
    with open(os.path.join(ANALYTICS_DIR, "analytics_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"done in {time.time() - t0:.1f}s -> {ANALYTICS_DIR}")


if __name__ == "__main__":
    main()
