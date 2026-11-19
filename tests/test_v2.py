"""Tests for the EquiCity 2.0 modules. Each one pins down an invariant that is
easy to break silently: a formula, an inverse, an exact decomposition, or a
no-leakage guarantee.

    python tests/run_tests.py          (or: python -m pytest tests/ -v)
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dashboard"))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))

from analysis import affordability as A  # noqa: E402
from analysis import spatial as S  # noqa: E402
from analysis import store  # noqa: E402


# --------------------------- affordability -----------------------------------

def test_annuity_matches_hand_calculation():
    # $300k over 25 years at 2.6%: r = 0.026/12, n = 300
    r, n = 0.026 / 12, 300
    expected = 300_000 * r / (1 - (1 + r) ** -n)
    assert float(A.annuity_payment(300_000, 0.026, 25)) == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(1361, abs=2)


def test_zero_rate_annuity_is_straight_line():
    assert float(A.annuity_payment(120_000, 0.0, 10)) == pytest.approx(1000.0)


def test_income_at_reproduces_table_points():
    t = A.income_table()
    row = t[(t["year"] == 2025) & (t["percentile"] == 50)]["income"].iloc[0]
    assert A.income_at(2025, 50)[0] == pytest.approx(row)
    assert row == pytest.approx(10591)  # SingStat 2025 household median, excl. employer CPF


def test_household_series_replaces_member_times_3_09():
    # The v1 approximation overstated low incomes by about a quarter.
    p10_household = A.income_at(2025, 10)[0]
    p10_member_x_size = 1133 * 3.09
    assert p10_member_x_size / p10_household > 1.2


def test_percentile_inverse_round_trip():
    for p in [12, 25, 50, 73, 88]:
        inc, _ = A.income_at(2024, p)
        back, flag = A.percentile_for_income(2024, inc)
        assert flag == ""
        assert back == pytest.approx(p, abs=1e-6)


def test_percentile_flags_outside_published_range():
    assert A.percentile_for_income(2025, 100)[1] == "<"
    assert A.percentile_for_income(2025, 10_000_000)[1] == ">"


def test_index_is_100_exactly_at_the_cap():
    fin = A.Financing()
    inc, _ = A.income_at(2025, 50)
    # price whose payment is exactly 30% of P50 income
    pay = inc * fin.msr_cap
    r, n = fin.rate / 12, fin.years * 12
    principal = pay * (1 - (1 + r) ** -n) / r
    price = principal / fin.ltv
    ev = A.evaluate(price, 2025, 50, fin)
    assert ev["msr_pct"] == pytest.approx(30.0)
    assert ev["affordability_index"] == pytest.approx(100.0)
    assert ev["required_percentile"] == pytest.approx(50.0, abs=1e-6)


def test_frame_matches_scalar_evaluate():
    df = pd.DataFrame({"year": [2010, 2020, 2025], "price": [350_000, 450_000, 620_000]})
    f = A.affordability_frame(df, 40)
    for _, r in f.iterrows():
        ev = A.evaluate(r["price"], int(r["year"]), 40)
        assert r["msr_pct"] == pytest.approx(ev["msr_pct"])
        assert r["required_percentile"] == pytest.approx(ev["required_percentile"])


def test_historical_ltv_steps():
    assert [A.historical_ltv(y) for y in (2019, 2022, 2023, 2025)] == [0.90, 0.85, 0.80, 0.75]


def test_median_from_brackets_interpolates():
    counts = {"Below_1_000": 10, "1_000_1_999": 10, "2_000_2_999": 20}
    # 20 of 40 below 2000 -> median sits exactly at the 2000 boundary
    assert A.median_from_brackets(counts) == pytest.approx(2000.0, abs=1.0)


# ------------------------------- spatial -------------------------------------

def _grid(n=6):
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    return np.column_stack([1.3 + ys.ravel() * 0.01, 103.8 + xs.ravel() * 0.01])


def test_morans_i_detects_a_gradient_and_not_noise():
    coords = _grid()
    W = S.knn_weights(coords, k=4)
    gradient = coords[:, 1] * 100
    assert S.morans_i(gradient, W, n_perm=499)["p_sim"] < 0.01
    noise = np.random.default_rng(0).normal(size=len(coords))
    assert S.morans_i(noise, W, n_perm=499)["p_sim"] > 0.05


def test_gi_star_finds_a_planted_hot_spot():
    coords = _grid()
    x = np.zeros(len(coords))
    x[[0, 1, 6, 7]] = 10.0  # one corner
    W = S.knn_weights(coords, k=4, include_self=True)
    gi = S.getis_ord_gi(x, W, n_perm=999)
    assert gi.loc[0, "gi_z"] > 2
    assert gi["gi_z"].idxmax() in {0, 1, 6, 7}


def test_theil_decomposition_is_exact():
    rng = np.random.default_rng(1)
    x = rng.uniform(1, 5, 40)
    g = np.repeat(list("ABCD"), 10)
    d = S.theil_decomposition(x, g)
    assert d["between"] + d["within"] == pytest.approx(d["theil"], abs=1e-12)


def test_gini_bounds():
    assert S.gini(np.ones(10)) == pytest.approx(0.0, abs=1e-12)
    assert 0.8 < S.gini(np.r_[np.zeros(99) + 1e-9, 1.0]) <= 1.0


# ------------------------------- hedonic -------------------------------------

def test_hedonic_recovers_planted_coefficients():
    from hedonic import fit_year
    rng = np.random.default_rng(3)
    n = 6000
    towns = rng.choice(["Ang Mo Kio", "Bedok", "Bishan"], n)
    lease = rng.uniform(50, 95, n)
    area = rng.uniform(60, 120, n)
    storey = rng.choice([2.0, 5.0, 8.0, 11.0], n)
    month = rng.integers(400, 412, n)
    lp = (12.0 + 0.8 * np.log(area) + 0.01 * lease
          + np.where(towns == "Bishan", 0.20, np.where(towns == "Bedok", -0.05, 0.0))
          + 0.004 * storey + rng.normal(0, 0.02, n))
    df = pd.DataFrame({"year": 2023, "month_idx": month, "town": towns, "flat_type": "4-Room",
                       "flat_model": "Model A", "remaining_lease": lease, "floor_area_sqm": area,
                       "storey_mid": storey, "log_price": lp})
    f = fit_year(df)
    assert f.b("log_area") == pytest.approx(0.8, abs=0.02)
    assert f.town_effect("Bishan") == pytest.approx(0.20, abs=0.005)
    assert f.town_effect("Bedok") == pytest.approx(-0.05, abs=0.005)
    assert f.town_effect("Ang Mo Kio") == 0.0  # reference town


# ------------------------------ price model ----------------------------------

def test_cqr_margin_achieves_coverage():
    from price_model import cqr_margin
    rng = np.random.default_rng(4)
    y = rng.normal(0, 1, 4000)
    lo, hi = np.full_like(y, -0.5), np.full_like(y, 0.5)  # deliberately too narrow
    q = cqr_margin(lo[:2000], hi[:2000], y[:2000], alpha=0.2)
    cover = np.mean((y[2000:] >= lo[2000:] - q) & (y[2000:] <= hi[2000:] + q))
    assert 0.77 < cover < 0.83


def test_target_encoding_oof_does_not_see_own_row():
    from price_model import TargetEncoder
    df = pd.DataFrame({
        "town": ["T"] * 6, "street_name": ["S"] * 6,
        "block_key": ["S | 1"] * 5 + ["S | 2"], "floor_area_sqm": [100.0] * 6})
    y = np.log(100.0) + np.array([0, 0, 0, 0, 0, 5.0])  # block 2's only sale is extreme
    st_, bl = TargetEncoder().fit_transform_oof(df, y, n_folds=6, seed=0)
    # block 2 has one sale: out of fold it must not carry its own +5
    assert bl[5] < 1.0


def test_index_carry_forward_uses_no_future_months():
    from price_model import index_for
    li = pd.Series(np.arange(10, dtype=float), index=range(10))
    df = pd.DataFrame({"month_idx": [3, 6, 9]})
    assert list(index_for(df, li, carry_from=5)) == [3.0, 5.0, 5.0]


# ----------------------------- served model ----------------------------------

def _bundle_or_skip():
    if store.model_bundle() is None:
        return None
    return store.model_bundle()


def test_group_shapley_is_additive():
    if _bundle_or_skip() is None:
        return
    from analysis import explain as E
    x = E.make_row("Tampines", "4-Room", "Model A", 92, 8, 1988, street="TAMPINES ST 81")
    r = E.group_shapley(x, max_background=30)
    assert abs(r["additivity_error_log"]) < 1e-9
    assert r["contributions"]["dollars"].sum() == pytest.approx(
        r["predicted_price"] - r["baseline_price"], rel=1e-6)


def test_pairwise_compare_sums_to_price_gap():
    if _bundle_or_skip() is None:
        return
    from analysis import explain as E
    xa = E.make_row("Bishan", "4-Room", "Model A", 95, 11, 1990)
    xb = E.make_row("Woodlands", "4-Room", "Model A", 95, 11, 1990)
    r = E.compare(xa, xb)
    assert r["contributions"]["dollars"].sum() == pytest.approx(r["price_a"] - r["price_b"], rel=1e-6)
    # identical except location: every non-location driver contributes nothing
    non_loc = r["contributions"][~r["contributions"]["driver"].str.startswith("Location")]
    assert non_loc["dollars"].abs().max() < 1e-6


def test_prediction_interval_brackets_point():
    if _bundle_or_skip() is None:
        return
    from analysis import explain as E
    p = E.predict(E.make_row("Punggol", "5-Room", "Premium Apartment", 112, 11, 2012))
    assert p["lo"] <= p["price"] <= p["hi"]
