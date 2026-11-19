"""Precompute global explanations for the served price model (stage 5).

    python pipeline/build_explanations.py      (after audit_price_model.py)

Writes to processed_data/analytics/:
  global_shapley.csv     mean |contribution| and mean signed contribution of
                         each driver over 300 recent transactions
  shapley_sample.csv     the per-flat contributions behind it (for the
                         beeswarm-style chart)
  pdp_<feature>.csv      partial dependence: how the model's price for a
                         typical recent flat moves as ONE feature varies,
                         others held at their observed values (averaged)
  model_town_premium.csv the model's price for the same reference 4-room
                         flat placed in each town -- an independent,
                         non-linear cross-check of the hedonic location
                         premium
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "dashboard"))
from analysis import explain as E  # noqa: E402
from analysis.store import ANALYTICS  # noqa: E402

N_FLATS = 300


def main():
    b = E.bundle()
    bg = b["background"]
    sample = bg.sample(min(N_FLATS, len(bg)), random_state=11).reset_index(drop=True)
    li = E.market_log_index(b["last_month_idx"], b)
    rows = []
    for i in range(len(sample)):
        x = sample.iloc[[i]].copy()
        x.attrs["month_idx"] = b["last_month_idx"]
        r = E.group_shapley(x, b=b, max_background=60)
        c = r["contributions"].assign(flat=i, predicted_price=r["predicted_price"])
        rows.append(c)
        if i % 50 == 0:
            print(f"  {i}/{len(sample)}", flush=True)
    sh = pd.concat(rows, ignore_index=True)
    sh.to_csv(os.path.join(ANALYTICS, "shapley_sample.csv"), index=False)
    g = (sh.groupby("driver")
           .agg(mean_abs_pct=("pct_effect", lambda v: np.mean(np.abs(v))),
                mean_abs_dollars=("dollars", lambda v: np.mean(np.abs(v))),
                mean_pct=("pct_effect", "mean"))
           .sort_values("mean_abs_dollars", ascending=False).reset_index())
    g.to_csv(os.path.join(ANALYTICS, "global_shapley.csv"), index=False)
    print(g.to_string())

    # partial dependence
    base = bg.sample(min(400, len(bg)), random_state=5).reset_index(drop=True)
    grids = {
        "remaining_lease": np.arange(45, 96, 2.5),
        "floor_area_sqm": np.arange(40, 160, 5.0),
        "storey_mid": np.arange(2, 41, 3.0),
    }
    if b["use_geo"]:
        grids["dist_mrt_km"] = np.arange(0.1, 2.05, 0.1)
        grids["dist_cbd_km"] = np.arange(2, 22, 1.0)
    for feat, grid in grids.items():
        out = []
        for v in grid:
            X = base.copy()
            X[feat] = np.float32(v)
            if feat == "remaining_lease":  # keep lease cohort consistent with sale date
                X["lease_commence_date"] = (X["t"] - 99 + v).astype("float32")
            out.append({feat: float(v), "price": float(np.exp(b["point"].predict(X)).mean() * np.exp(li))})
        pdp = pd.DataFrame(out)
        pdp["relative_to_mid"] = pdp["price"] / pdp["price"].iloc[len(pdp) // 2]
        pdp.to_csv(os.path.join(ANALYTICS, f"pdp_{feat}.csv"), index=False)

    # model-based town premium for a reference 4-room flat
    rows = []
    for town in b["categories"]["town"]:
        if town not in b["encoder"]["town"].index:
            continue
        x = E.make_row(town, "4-Room", "Model A", 93, 8, 1990 + b["last_month_idx"] / 12 - 29, b=b)
        rows.append({"town": town, "price": E.predict(x, b)["price"]})
    tp = pd.DataFrame(rows)
    tp["premium_vs_median_pct"] = (tp["price"] / tp["price"].median() - 1) * 100
    tp.to_csv(os.path.join(ANALYTICS, "model_town_premium.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
