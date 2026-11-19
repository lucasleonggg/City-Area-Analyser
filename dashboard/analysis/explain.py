"""Prediction + explanation for the served price model (stages 1 and 5).

The explanation is computed against THE SAME MODEL that produced the
prediction. v1 predicted with a 38-feature forest and explained a separate
6-feature forest, so its Shapley bars did not describe the number shown.

EXACT GROUP SHAPLEY
-------------------
Features are grouped into seven interpretable drivers (GROUPS). For a flat x
and a background sample B of recent transactions, the value of a coalition
S of groups is the model's mean prediction with x's values for S and the
background's values for everything else (interventional Shapley). With 7
groups there are 2^7 = 128 coalitions, so the Shapley values are computed
EXACTLY rather than sampled, and they sum to prediction minus baseline.

The model works in log price, so contributions are naturally percentages.
Dollar amounts use the logarithmic-mean (LMDI) conversion:
    $_i = phi_i x (P - B) / (ln P - ln B)
which is exact (the dollars add up to P - B), order-independent, and never
blows up the way proportional allocation does when effects of opposite sign
nearly cancel.

WHAT THESE NUMBERS ARE NOT
--------------------------
A Shapley value describes how THIS MODEL uses a feature, given the data it
was trained on. It is not a causal effect. "Location contributed +$80k"
means the model prices flats in this town higher than average, holding the
other drivers at background values -- not that moving the same flat would
raise its value by $80k, and not that the town causes the premium (it may
proxy for school access, amenity, flat condition or anything else
correlated with location). Features are also correlated with each other
(lease and lease cohort, town and micro-location), and interventional
Shapley evaluates the model on some feature combinations that rarely occur.
"""

import itertools
import math

import numpy as np
import pandas as pd

from .store import model_bundle

# Location is ONE driver on purpose: town, street/block encodings and
# coordinates all carry overlapping location information, and splitting them
# would divide the same effect between groups arbitrarily.
GROUPS = {
    "Location (town, street, block)": ["town", "te_street", "te_block", "lat", "lng",
                                       "dist_mrt_km", "dist_cbd_km"],
    "Floor area": ["floor_area_sqm"],
    "Flat type": ["flat_type"],
    "Flat model": ["flat_model"],
    "Floor level": ["storey_mid"],
    "Remaining lease": ["remaining_lease", "lease_commence_date"],
    "Sale date (era)": ["t"],
}

CAUSATION_NOTE = (
    "These are model attributions, not causal effects. They show how the model "
    "weighs each driver for this flat compared with a typical recent sale; they "
    "do not show what would happen to the price if you changed one feature."
)


def bundle():
    b = model_bundle()
    if b is None:
        raise RuntimeError("price_model_v2.joblib not found -- run pipeline/audit_price_model.py")
    return b


def feature_columns(b=None):
    b = b or bundle()
    return list(b["background"].columns)


def month_label(m):
    return f"{1990 + int(m) // 12}-{int(m) % 12 + 1:02d}"


def _haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def location_for(town, street=None, block=None, b=None):
    """Encodings and coordinates for a town / street / block, falling back
    up the hierarchy when the finer level is unknown."""
    b = b or bundle()
    enc = b["encoder"]
    te_town = float(enc["town"].get(town, enc["global"]))
    te_street = float(enc["street"].get(street, te_town)) if street else te_town
    key = f"{street} | {block}" if (street and block) else None
    te_block = float(enc["block"].get(key, te_street)) if key else te_street
    out = {"te_street": te_street, "te_block": te_block}
    if b["use_geo"]:
        bl = b["blocks"]
        if key is not None and key in set(bl["block_key"]):
            r = bl[bl["block_key"] == key].iloc[0]
            lat, lng = r.get("lat"), r.get("lng")
        else:
            sub = bl[(bl["street_name"] == street)] if street else bl[bl["town"] == town]
            if sub.empty or sub["lat"].isna().all():
                sub = bl[bl["town"] == town]
            lat, lng = sub["lat"].mean(), sub["lng"].mean()
        st = b["stations"]
        out.update(lat=float(lat), lng=float(lng),
                   dist_cbd_km=float(_haversine(lat, lng, 1.2837, 103.8510)),
                   dist_mrt_km=float(np.min(_haversine(lat, lng, st[:, 0], st[:, 1]))))
    return out


def make_row(town, flat_type, flat_model, floor_area_sqm, storey_mid,
             lease_commence_date, street=None, block=None, month_idx=None, b=None) -> pd.DataFrame:
    b = b or bundle()
    month_idx = b["last_month_idx"] if month_idx is None else int(month_idx)
    t = 1990 + month_idx / 12
    row = {"town": town, "flat_type": flat_type, "flat_model": flat_model,
           "floor_area_sqm": float(floor_area_sqm), "storey_mid": float(storey_mid),
           "remaining_lease": float(max(0.0, 99 - (t - float(lease_commence_date)))),
           "lease_commence_date": float(lease_commence_date), "t": float(t)}
    row.update(location_for(town, street, block, b))
    X = pd.DataFrame([row])
    cats = b["categories"]
    for c in ["town", "flat_type", "flat_model"]:
        X[c] = pd.Categorical(X[c].astype(str), categories=cats[c])
    X = X[feature_columns(b)]
    for c in X.columns:
        if c not in cats:
            X[c] = X[c].astype("float32")
    X.attrs["month_idx"] = month_idx
    return X


def market_log_index(month_idx, b=None):
    b = b or bundle()
    li = b["log_index"]
    m = min(int(month_idx), int(li.index.max()))
    return float(li.loc[m])


def predict(X: pd.DataFrame, b=None) -> dict:
    b = b or bundle()
    li = market_log_index(X.attrs.get("month_idx", b["last_month_idx"]), b)
    rel = float(b["point"].predict(X)[0])
    lo = float(b["q_lo"].predict(X)[0]) - b["cqr_margin"]
    hi = float(b["q_hi"].predict(X)[0]) + b["cqr_margin"]
    lo, hi = min(lo, rel), max(hi, rel)
    return {"price": math.exp(rel + li), "lo": math.exp(lo + li), "hi": math.exp(hi + li),
            "coverage": 1 - b["alpha"], "log_rel": rel, "log_index": li}


def _coalition_batch(x: pd.DataFrame, bg: pd.DataFrame, groups: dict):
    """All 2^k coalitions stacked: block s holds the background with x's
    values substituted for the groups switched on in subset s."""
    names = list(groups)
    k = len(names)
    subsets = list(itertools.product([0, 1], repeat=k))
    nb, ns = len(bg), len(subsets)
    col_group = {c: gi for gi, g in enumerate(names) for c in groups[g]}
    on = np.array(subsets, dtype=bool)  # (ns, k)
    data = {}
    for c in bg.columns:
        is_cat = isinstance(bg[c].dtype, pd.CategoricalDtype)
        base = bg[c].cat.codes.to_numpy() if is_cat else bg[c].to_numpy()
        arr = np.tile(base, ns)
        if c in col_group:
            xv = x[c].cat.codes.iloc[0] if is_cat else x[c].iloc[0]
            mask = np.repeat(on[:, col_group[c]], nb)
            arr = arr.copy()
            arr[mask] = xv
        data[c] = (pd.Categorical.from_codes(arr, categories=bg[c].cat.categories)
                   if is_cat else arr)
    return names, subsets, pd.DataFrame(data)[list(bg.columns)]


def group_shapley(x: pd.DataFrame, background: pd.DataFrame | None = None, b=None,
                  max_background: int = 100) -> dict:
    """Exact Shapley over GROUPS. Returns log-point contributions, %
    effects and LMDI dollars, plus the baseline."""
    b = b or bundle()
    cols = set(feature_columns(b))
    groups = {g: [c for c in cs if c in cols] for g, cs in GROUPS.items()}
    groups = {g: cs for g, cs in groups.items() if cs}
    bg = (b["background"] if background is None else background)
    if len(bg) > max_background:
        bg = bg.sample(max_background, random_state=3)
    bg = bg.reset_index(drop=True).copy()
    names, subsets, big = _coalition_batch(x, bg, groups)
    pred = b["point"].predict(big[feature_columns(b)]).reshape(len(subsets), len(bg)).mean(axis=1)
    v = {s: p for s, p in zip(subsets, pred)}
    k = len(names)
    phi = np.zeros(k)
    for i in range(k):
        for s in subsets:
            if s[i] == 1:
                continue
            s_with = tuple(1 if j == i else s[j] for j in range(k))
            size = sum(s)
            w = math.factorial(size) * math.factorial(k - size - 1) / math.factorial(k)
            phi[i] += w * (v[s_with] - v[s])
    li = market_log_index(x.attrs.get("month_idx", b["last_month_idx"]), b)
    base_log, full_log = v[tuple([0] * k)], v[tuple([1] * k)]
    P, B = math.exp(full_log + li), math.exp(base_log + li)
    L = (P - B) / (full_log - base_log) if abs(full_log - base_log) > 1e-12 else P
    rows = [{"driver": n, "log_points": float(p), "pct_effect": (math.exp(p) - 1) * 100,
             "dollars": float(p * L)} for n, p in zip(names, phi)]
    return {"baseline_price": B, "predicted_price": P,
            "contributions": pd.DataFrame(rows).sort_values("dollars", key=np.abs, ascending=False),
            "additivity_error_log": float(phi.sum() - (full_log - base_log))}


def compare(xa: pd.DataFrame, xb: pd.DataFrame, b=None) -> dict:
    """Why do two flats get different predictions? Shapley with flat B as the
    single-row baseline: contributions sum exactly to log(P_A / P_B)."""
    out = group_shapley(xa, background=xb, b=b, max_background=1)
    out["price_a"], out["price_b"] = out.pop("predicted_price"), out.pop("baseline_price")
    return out
