"""Affordability engine (EquiCity 2.0, stage 2).

Question answered: for a household at income percentile P, buying flat type F
in town T in year Y with a standard HDB loan, how much of its income goes to
the mortgage -- and is that within the 30% Mortgage Servicing Ratio cap?

DEFINITIONS
-----------
monthly instalment  = annuity payment on (LTV x price) at rate r over n years
MSR                 = instalment / gross monthly household income
Affordability index = 100 x income / qualifying income, where qualifying
                      income = instalment / MSR cap. 100 means a household
                      at this percentile exactly meets the cap; above 100 it
                      has headroom; below 100 it cannot borrow the full LTV.
                      (Same construction as the US National Association of
                      Realtors' Housing Affordability Index, with Singapore's
                      30% MSR in place of NAR's 25% qualifying ratio.)
required percentile = the household income percentile whose income exactly
                      meets the cap; 100 minus it is the share of resident
                      employed households that could finance this flat.

INCOME SOURCE
-------------
SingStat, "Monthly Household Employment Income (Excluding Employer CPF
Contributions) Among Resident Employed Households at Selected Percentiles,
2000-2025" (data.gov.sg d_650c42ad53d69ad3f6f47f04c21616fb). This is a
HOUSEHOLD-level series, which replaces v1's per-household-MEMBER series
multiplied by an assumed 3.09 persons. That approximation was close at the
median (2025: $10,627 vs actual $10,591) but 26% too high at the 10th
percentile ($3,501 vs $2,779) and 5% too high at the 90th, so it overstated
affordability for exactly the households the question is about.

Excluding employer CPF is the right basis: MSR is assessed on gross monthly
income, which includes the employee's CPF contribution but not the
employer's.

HELD FIXED, ON PURPOSE
----------------------
Financing terms are held at today's rules for every year by default, so the
time series isolates the race between prices and incomes. Real historical
terms differed: HDB-loan LTV was 90% until 16 Dec 2021, 85% until 30 Sep
2022, 80% until 20 Aug 2024 and 75% since; the 30% MSR has applied to HDB
flats since 2013; and since 30 Sep 2022 HDB assesses loan eligibility at a 3%
interest floor rather than the 2.6% charged. `historical_ltv()` gives the
HDB-loan LTV in force in a given year. Change the inputs to explore them.

NOT MODELLED: CPF housing grants, existing CPF savings, sale proceeds of a
previous flat, household income ceilings for HDB loans, cash-over-valuation.
Results describe financing capacity for a first purchase, not eligibility.
"""

import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .store import RAW

INCOME_FILE = "singstat_household_employment_income_percentiles_excl_employer_cpf_2000_2025.csv"
CENSUS_FILE = "census2020_resident_households_by_planning_area_and_income_from_work.csv"
PERCENTILES = [10, 20, 30, 40, 50, 60, 70, 80, 90]


@dataclass(frozen=True)
class Financing:
    rate: float = 0.026      # HDB concessionary rate (CPF OA + 0.1%)
    years: int = 25          # HDB loan maximum tenure
    ltv: float = 0.75        # HDB loan LTV since 20 Aug 2024
    msr_cap: float = 0.30    # Mortgage Servicing Ratio cap for HDB flats


PRESETS = {
    "HDB loan (2.6%, 25 yrs, 75% LTV)": Financing(),
    "Bank loan at 4% stress rate (30 yrs, 75% LTV)": Financing(rate=0.04, years=30),
    "HDB loan, assessed at the 3% eligibility floor": Financing(rate=0.03),
    "HDB loan, pre-Dec-2021 terms (90% LTV)": Financing(ltv=0.90),
}


def historical_ltv(year: int) -> float:
    """HDB-loan LTV in force for most of `year` (changes: 16 Dec 2021 -> 85%,
    30 Sep 2022 -> 80%, 20 Aug 2024 -> 75%)."""
    if year <= 2021:
        return 0.90
    if year == 2022:
        return 0.85
    if year == 2023:
        return 0.80
    if year == 2024:
        return 0.80
    return 0.75


def annuity_payment(principal, rate_annual, years):
    """Monthly payment on a fully amortising loan."""
    principal = np.asarray(principal, dtype=float)
    r = rate_annual / 12
    n = years * 12
    if r == 0:
        return principal / n
    return principal * r / (1 - (1 + r) ** -n)


@lru_cache(maxsize=1)
def income_table() -> pd.DataFrame:
    """Long table: year, percentile (int), monthly household income."""
    df = pd.read_csv(os.path.join(RAW, INCOME_FILE))
    df["percentile"] = df["Dollar"].map(lambda s: int(re.match(r"(\d+)", s).group(1)))
    long = df.drop(columns=["Dollar"]).melt("percentile", var_name="year", value_name="income")
    long["year"] = long["year"].astype(int)
    long["income"] = long["income"].astype(float)
    return long.sort_values(["year", "percentile"]).reset_index(drop=True)


def latest_income_year() -> int:
    return int(income_table()["year"].max())


def _curve(year):
    t = income_table()
    y = min(max(int(year), int(t["year"].min())), int(t["year"].max()))
    g = t[t["year"] == y]
    return g["percentile"].to_numpy(float), np.log(g["income"].to_numpy(float)), y


def income_at(year, percentile) -> tuple[float, int]:
    """Monthly household income at a percentile, interpolated in log income.
    Years outside 2000-2025 are clamped to the nearest available year; the
    year actually used is returned so callers can flag it."""
    p, li, used = _curve(year)
    pc = min(max(float(percentile), p[0]), p[-1])
    return float(np.exp(np.interp(pc, p, li))), used


def percentile_for_income(year, monthly_income) -> tuple[float, str]:
    """Inverse of income_at. Returns (percentile, flag) where flag is '',
    '<' (below the 10th percentile) or '>' (above the 90th)."""
    p, li, _ = _curve(year)
    x = math.log(max(monthly_income, 1.0))
    if x < li[0]:
        return 10.0, "<"
    if x > li[-1]:
        return 90.0, ">"
    return float(np.interp(x, li, p)), ""


def evaluate(price, year, percentile, fin: Financing = Financing()) -> dict:
    """All affordability measures for one flat price."""
    pay = float(annuity_payment(price * fin.ltv, fin.rate, fin.years))
    inc, inc_year = income_at(year, percentile)
    req = pay / fin.msr_cap
    req_pct, flag = percentile_for_income(year, req)
    return {
        "monthly_payment": pay,
        "monthly_income": inc,
        "income_year_used": inc_year,
        "msr_pct": pay / inc * 100,
        "affordability_index": inc / req * 100,
        "required_income": req,
        "required_percentile": req_pct,
        "required_percentile_flag": flag,
        "share_households_qualifying_pct": (
            100 - req_pct if flag == "" else (">90" if flag == "<" else "<10")),
        "price_to_annual_income": price / (inc * 12),
        "downpayment": price * (1 - fin.ltv),
        "downpayment_years_income": price * (1 - fin.ltv) / (inc * 12),
    }


def affordability_frame(prices: pd.DataFrame, percentile: float,
                        fin: Financing = Financing(), price_col="price") -> pd.DataFrame:
    """Vectorised `evaluate` over a frame with `year` and a price column.
    Rows with years before the income series starts (2000) are dropped."""
    t = income_table()
    df = prices[prices["year"] >= int(t["year"].min())].copy()
    years = sorted(df["year"].unique())
    inc_map = {y: income_at(y, percentile)[0] for y in years}
    df["income_year_used"] = df["year"].clip(upper=int(t["year"].max()))
    df["monthly_income"] = df["year"].map(inc_map)
    df["monthly_payment"] = annuity_payment(df[price_col].to_numpy(float) * fin.ltv, fin.rate, fin.years)
    df["msr_pct"] = df["monthly_payment"] / df["monthly_income"] * 100
    df["affordability_index"] = fin.msr_cap * 100 / df["msr_pct"] * 100
    df["required_income"] = df["monthly_payment"] / fin.msr_cap
    rp = [percentile_for_income(y, r) for y, r in zip(df["year"], df["required_income"])]
    df["required_percentile"] = [r[0] for r in rp]
    df["required_percentile_flag"] = [r[1] for r in rp]
    df["price_to_annual_income"] = df[price_col] / (df["monthly_income"] * 12)
    df["within_cap"] = df["msr_pct"] <= fin.msr_cap * 100
    return df


# ---------------------------------------------------------------------------
# Local (resident-based) income lens: Census 2020 by planning area.
# ---------------------------------------------------------------------------

# Census planning area(s) whose households best represent each HDB town.
TOWN_TO_PLANNING_AREAS = {
    "Kallang/Whampoa": ["Kallang", "Novena"],
    "Central Area": ["Outram", "Downtown Core"],
}
_BRACKET_RE = re.compile(r"^(Below_1_000|\d+_\d+_\d+_\d+|20_000andOver)$")


def _bracket_bounds(col):
    if col == "Below_1_000":
        return 0.0, 1000.0
    if col == "20_000andOver":
        return 20000.0, 30000.0  # open top bracket; only matters above P85+
    a, b, c, d = col.split("_")
    return float(a + b), float(c + d) + 1.0


def median_from_brackets(counts: dict) -> float:
    cols = sorted(counts, key=lambda c: _bracket_bounds(c)[0])
    total = sum(counts[c] for c in cols)
    half, cum = total / 2, 0.0
    for c in cols:
        lo, hi = _bracket_bounds(c)
        n = counts[c]
        if cum + n >= half and n > 0:
            return lo + (half - cum) / n * (hi - lo)
        cum += n
    return float("nan")


@lru_cache(maxsize=1)
def census_local_income() -> pd.DataFrame:
    """Median monthly household income from work among EMPLOYED resident
    households, by planning area (Census 2020, includes employer CPF), and
    each area's ratio to the national median.

    CAVEAT: planning-area households include condominium and landed
    residents, so areas like Bukit Timah, Bishan, Queenstown and Marine
    Parade read richer than their HDB residents are. Use as a relative
    lens, not as the income of HDB buyers.
    """
    c = pd.read_csv(os.path.join(RAW, CENSUS_FILE))
    brackets = [k for k in c.columns if _BRACKET_RE.match(k)]
    rows = []
    for _, r in c.iterrows():
        counts = {b: float(r[b]) for b in brackets}
        rows.append({"planning_area": r["Number"], "median": median_from_brackets(counts),
                     "employed_households": sum(counts.values())})
    out = pd.DataFrame(rows)
    nat = float(out.loc[out["planning_area"] == "Total", "median"].iloc[0])
    out["ratio_to_national"] = out["median"] / nat
    return out


def town_local_income_ratio(town: str) -> float | None:
    ci = census_local_income().set_index("planning_area")
    areas = TOWN_TO_PLANNING_AREAS.get(town, [town])
    areas = [a for a in areas if a in ci.index]
    if not areas:
        return None
    w = ci.loc[areas, "employed_households"]
    return float((ci.loc[areas, "ratio_to_national"] * w).sum() / w.sum())
