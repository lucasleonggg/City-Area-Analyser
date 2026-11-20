"""Tests for the EquiCity 3.0 modules."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dashboard"))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))

from analysis import policy as P  # noqa: E402
from analysis import scenario as SC  # noqa: E402


# ----------------------------- scenario engine -------------------------------

def test_buyers_stamp_duty_bands():
    assert SC.buyers_stamp_duty(500_000) == pytest.approx(1_800 + 3_600 + 140_000 * 0.03)
    assert SC.buyers_stamp_duty(1_000_000) == pytest.approx(24_600)  # 1.8k + 3.6k + 19.2k


def test_annuity_principal_inverts_payment():
    pay = SC.annuity_payment(400_000, 0.03, 25)
    assert SC.annuity_principal(pay, 0.03, 25) == pytest.approx(400_000)


def test_max_price_is_the_feasibility_boundary():
    h = SC.Household(income=7_000, cash=40_000, cpf_oa=60_000, grants=30_000)
    m = SC.max_price(h, SC.Loan())
    assert SC.evaluate(m["price"], h, SC.Loan())["feasible"]
    assert not SC.evaluate(m["price"] * 1.01, h, SC.Loan())["feasible"]


def test_loan_respects_msr_ltv_and_grants():
    h = SC.Household(income=5_000, cash=1e9, cpf_oa=0, grants=0)
    e = SC.evaluate(800_000, h, SC.Loan())
    assert e["loan"] <= 0.75 * 800_000 + 1e-6
    # instalment at the ASSESSMENT rate must fit within 30% of income
    assert SC.annuity_payment(e["loan"], 0.03, 25) <= 0.30 * 5_000 + 1e-6


def test_bank_loan_needs_five_percent_cash():
    h = SC.Household(income=20_000, cash=0, cpf_oa=1e9, other_upfront=0)
    bank = SC.LOAN_PRESETS["Bank loan (3.0%, assessed at 4%, 30 yrs)"]
    assert not SC.evaluate(500_000, h, bank)["feasible"]
    assert SC.evaluate(500_000, h, SC.Loan())["feasible"]  # HDB loan: CPF can cover it all


def test_tdsr_counts_other_debts_for_bank_loans():
    bank = SC.LOAN_PRESETS["Bank loan (3.0%, assessed at 4%, 30 yrs)"]
    a = SC.max_loan_by_income(SC.Household(income=10_000, other_debt=0), bank)
    b = SC.max_loan_by_income(SC.Household(income=10_000, other_debt=3_500), bank)
    assert b < a


# ------------------------------- policy lab ----------------------------------

def test_income_only_mode_grants_do_not_change_borrowing_need():
    base = P.max_price_at(2025, 40, SC.Loan(), grants=0)
    with_grant = P.max_price_at(2025, 40, SC.Loan(), grants=50_000)
    assert with_grant == pytest.approx(base)


def test_savings_mode_grants_help():
    a = P.max_price_at(2025, 40, SC.Loan(), grants=0, assets_multiple=0.5)
    b = P.max_price_at(2025, 40, SC.Loan(), grants=50_000, assets_multiple=0.5)
    assert b > a


def test_higher_rate_never_helps():
    lo = P.max_price_at(2025, 50, SC.Loan(rate=0.026, assess_rate=0.03))
    hi = P.max_price_at(2025, 50, SC.Loan(rate=0.05, assess_rate=0.05))
    assert hi < lo


def test_min_percentile_is_monotone_in_price():
    p1, _ = P.min_percentile(400_000, 2025, SC.Loan())
    p2, _ = P.min_percentile(700_000, 2025, SC.Loan())
    assert p2 >= p1


# ------------------------------ rail history ---------------------------------

def test_station_counts_only_after_it_opens():
    from rail_history import load_stations, nearest_open_distance
    s = load_stations()
    stn = s[s["station"] == "STEVENS"]  # Downtown Line, opened 27 Dec 2015
    lat, lng = float(stn["lat"].iloc[0]), float(stn["lng"].iloc[0])
    opened = int(stn["open_month_idx"].iloc[0])
    before = nearest_open_distance([lat], [lng], [opened - 1], s)[0]
    after = nearest_open_distance([lat], [lng], [opened], s)[0]
    assert after < 0.05 and before > 0.5


def test_opening_dates_are_plausible():
    from rail_history import load_stations
    s = load_stations()
    assert s["opened"].min() >= "1987-11-07"  # first MRT segment
    assert s["opened"].max() <= "2026-12-31"
    assert len(s) >= 180


# ----------------------------- target encoding -------------------------------

def test_temporal_encoding_never_uses_same_or_later_months():
    from price_model import TargetEncoder
    df = pd.DataFrame({"town": ["T"] * 4, "street_name": ["S"] * 4, "block_key": ["S | 1"] * 4,
                       "floor_area_sqm": [100.0] * 4, "month_idx": [10, 11, 11, 12]})
    y = np.log(100.0) + np.array([0.0, 1.0, 1.0, 9.0])  # a huge later sale
    st, bl = TargetEncoder().fit_transform_temporal(df, y)
    # month-10 sale has no history; month-12 sale sees months 10-11 only (not its own +9)
    assert bl[3] < 2.0
    # the two month-11 sales see only month 10, so they are encoded identically
    assert bl[1] == pytest.approx(bl[2])


# ------------------------------- bundle io -----------------------------------

def test_bundle_round_trip_keeps_values():
    from analysis.bundle_io import _frame_to_plain, _plain_to_frame
    df = pd.DataFrame({"town": pd.Categorical(["A", "B", "A"], categories=["A", "B", "C"]),
                       "x": np.array([1.5, 2.5, 3.5], dtype="float32"), "s": ["p", "q", None]})
    back = _plain_to_frame(_frame_to_plain(df))
    assert list(back["town"].cat.categories) == ["A", "B", "C"]
    assert back["town"].tolist() == ["A", "B", "A"]
    assert np.allclose(back["x"], df["x"])


# ----------------------------- reproducibility -------------------------------

def test_transactions_load_in_canonical_order(tmp_path=None):
    """Two files with the same rows in different orders (and with category
    values first seen in different orders) must load identically."""
    import tempfile
    import common
    rows = pd.DataFrame({
        "month": ["2020-01", "2020-01", "2019-05", "2021-03"],
        "town": ["Bishan", "Ang Mo Kio", "Yishun", "Bedok"],
        "flat_type": ["4-Room", "3-Room", "5-Room", "4-Room"],
        "block": ["1", "2", "3", "4"], "street_name": ["S1", "S2", "S3", "S4"],
        "storey_range": ["01 TO 03", "04 TO 06", "07 TO 09", "10 TO 12"],
        "floor_area_sqm": [90.0, 67.0, 110.0, 92.0], "flat_model": ["Model A"] * 4,
        "lease_commence_date": [1990, 1980, 2000, 1985], "remaining_lease": [69.0, 59.0, 79.0, 63.0],
        "resale_price": [500000.0, 350000.0, 520000.0, 480000.0]})
    d = tempfile.mkdtemp()
    out = []
    for i, order in enumerate([[0, 1, 2, 3], [3, 2, 1, 0]]):
        p = os.path.join(d, f"f{i}.csv")
        rows.iloc[order].to_csv(p, index=False)
        old = common.DATA_PATH
        common.DATA_PATH = p
        try:
            out.append(common.load_transactions()[["month", "town", "resale_price"]].astype(str))
        finally:
            common.DATA_PATH = old
    assert out[0].equals(out[1])


# ----------------------- trend tests with serial correlation (3.2) -----------

def _ar1_trend(n=32, phi=0.7, slope=0.0, seed=0):
    rng = np.random.default_rng(seed)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + rng.normal()
    return slope * np.arange(n) + e


# 150 series: persistence from -0.5 to 0.9, falling / flat / rising trends,
# 10 seeds each. pymannkendall is a REQUIRED validation dependency
# (requirements-validation.txt): if it is missing this test fails, it does not skip.
_GRID = [(phi, slope, seed) for phi in (-0.5, 0.0, 0.3, 0.6, 0.9)
         for slope in (-0.05, 0.0, 0.05) for seed in range(10)]


@pytest.mark.parametrize("phi,slope,seed", _GRID)
def test_trend_tests_match_reference_implementation(phi, slope, seed):
    import pymannkendall as mk
    import spatial_robustness as SR
    y = _ar1_trend(phi=phi, slope=slope, seed=seed)
    with np.errstate(invalid="ignore"):
        pairs = [
            (SR.mann_kendall(y)[1], mk.original_test(y).p),
            (SR.mann_kendall_hamed_rao(y)[1], mk.hamed_rao_modification_test(y).p),
            (SR.mann_kendall_hamed_rao(y, max_lag=3)[1], mk.hamed_rao_modification_test(y, lag=3).p),
            (SR.mann_kendall_tfpw(y)[1], mk.trend_free_pre_whitening_modification_test(y).p),
            (SR.mann_kendall_prewhitened(y)[1], mk.pre_whitening_modification_test(y).p),
        ]
    for ours, ref in pairs:
        if np.isnan(ref):
            assert np.isnan(ours)
        else:
            assert ours == pytest.approx(ref, abs=1e-9)


def test_reference_package_is_the_pinned_version():
    import pymannkendall as mk
    assert mk.__version__ == "1.4.3"


def test_hamed_rao_is_less_confident_on_autocorrelated_noise():
    import spatial_robustness as SR
    y = _ar1_trend(phi=0.9, slope=0.0, seed=3)
    assert SR.mann_kendall_hamed_rao(y)[1] >= SR.mann_kendall(y)[1]


def test_moran_trend_summary_reports_adjusted_tests():
    path = os.path.join(ROOT, "processed_data", "analytics", "robust_moran_summary.csv")
    if not os.path.exists(path):
        pytest.skip("analytics not built")
    s = pd.read_csv(path)
    for col in ("lag1_acf_detrended", "mk_hamed_rao_p", "mk_tfpw_p", "mk_prewhitened_p"):
        assert col in s.columns
    assert len(s) == 25


def test_persistence_null_is_calibrated():
    """On trendless AR(1) series with the true persistence, the simulated
    p-value should reject about 5% of the time, not far more."""
    import trend_sensitivity as TS
    rng = np.random.default_rng(7)
    old = TS.N_SIM
    TS.N_SIM = 500
    try:
        Y = TS.ar1(200, 32, 0.8, rng)
        rate = np.mean([TS.persistence_null_p(y, 0.8, rng) < 0.05 for y in Y])
    finally:
        TS.N_SIM = old
    assert rate < 0.10


def test_ar2_fit_recovers_coefficients():
    import trend_sensitivity as TS
    rng = np.random.default_rng(11)
    y = TS.ar2(1, 5000, 0.5, -0.3, rng)[0]
    phi1, phi2 = TS.fit_ar2(y)
    assert phi1 == pytest.approx(0.5, abs=0.05)
    assert phi2 == pytest.approx(-0.3, abs=0.05)


def test_bias_correction_moves_persistence_up():
    import trend_sensitivity as TS
    assert TS.bias_corrected_ar1(0.5, 32) > 0.5
    assert TS.bias_corrected_ar1(0.94, 32) <= 0.95


def test_structural_break_tells_step_from_trend():
    import trend_sensitivity as TS
    yrs = np.arange(1995, 2027)
    rng = np.random.default_rng(5)
    step = np.where(yrs >= 2010, 1.0, 0.0) + rng.normal(0, 0.05, len(yrs))
    trend = 0.03 * (yrs - 1995) + rng.normal(0, 0.05, len(yrs))
    s = TS.structural_break(yrs, step)
    t = TS.structural_break(yrs, trend)
    assert s["preferred_by_bic"] == "step" and s["break_first_year_after"] == 2010
    assert t["preferred_by_bic"] == "linear trend"
    assert t["sen_slope_per_decade_before"] > 0 and t["sen_slope_per_decade_after"] > 0


def test_trend_sensitivity_outputs_present():
    for f in ("robust_trend_sensitivity.csv", "robust_trend_breaks.csv", "robust_trend_test_size.csv"):
        path = os.path.join(ROOT, "processed_data", "analytics", f)
        if not os.path.exists(path):
            pytest.skip("analytics not built")
        assert len(pd.read_csv(path)) > 0
