"""EquiCity Policy Lab: mechanical what-if experiments (EquiCity 3.0).

Question: holding prices and incomes as observed, what happens to the
modelled ability of households at each income percentile to finance a
flat if financing rules change?

MECHANICAL ONLY. Every experiment here changes the financing arithmetic and
nothing else: prices do not respond, incomes do not respond, buyers do not
change behaviour. A real interest-rate rise or grant increase would also
move prices (a larger grant is partly capitalised into what sellers ask),
and EquiCity has no validated model of that response, so it does not
pretend to. Read every result as "the first-round financing effect".

No experiment on BTO supply is offered for the same reason: turning a
supply change into a resale-price change needs an estimated, validated
supply-price relationship that this project does not have.

Savings are the one input with no public distribution by income
percentile. By default they are assumed NOT to bind (only the income and
LTV tests apply, as in v2's Affordability Index). The user can instead set
liquid assets (cash + CPF OA) as a multiple of annual household income,
which is an assumption, labelled as one.
"""

import numpy as np
import pandas as pd

from . import affordability as A
from .scenario import Household, Loan, max_loan_by_income, max_price


def household_at(year: int, pct: float, grants: float, assets_multiple: float | None,
                 cash_share: float = 0.4) -> Household:
    inc, _ = A.income_at(year, pct)
    if assets_multiple is None:  # savings assumed not to bind
        return Household(income=inc, cash=1e9, cpf_oa=1e9, grants=grants, other_upfront=0.0)
    assets = assets_multiple * inc * 12
    return Household(income=inc, cash=assets * cash_share, cpf_oa=assets * (1 - cash_share), grants=grants)


def max_price_at(year, pct, loan: Loan, grants=0.0, assets_multiple=None) -> float:
    h = household_at(year, pct, grants, assets_multiple)
    if assets_multiple is None:
        # Savings assumed not to bind: the household can finance price P if
        # its income supports the loan it needs, min(LTV x P, P - grants).
        L = max_loan_by_income(h, loan)
        g = min(grants, 10 ** 9)
        if loan.ltv >= 1 or L / loan.ltv >= g / (1 - loan.ltv):
            return L / loan.ltv
        return L + g
    return max_price(h, loan)["price"]


def min_percentile(price, year, loan: Loan, grants=0.0, assets_multiple=None) -> tuple[float, str]:
    """Lowest income percentile (10-90) whose household can finance `price`."""
    if max_price_at(year, 90, loan, grants, assets_multiple) < price:
        return 90.0, ">"
    if max_price_at(year, 10, loan, grants, assets_multiple) >= price:
        return 10.0, "<"
    lo, hi = 10.0, 90.0
    for _ in range(25):
        mid = (lo + hi) / 2
        if max_price_at(year, mid, loan, grants, assets_multiple) >= price:
            hi = mid
        else:
            lo = mid
    return hi, ""


def percentile_ladder(year, loan, grants=0.0, assets_multiple=None, pcts=range(10, 91, 10)) -> pd.DataFrame:
    return pd.DataFrame([{"percentile": p, "max_price": max_price_at(year, p, loan, grants, assets_multiple),
                          "income": A.income_at(year, p)[0]} for p in pcts])


def towns_within_reach(town_prices: pd.DataFrame, ladder: pd.DataFrame) -> pd.DataFrame:
    """For each percentile, how many towns' reference flats are affordable."""
    rows = []
    for _, r in ladder.iterrows():
        ok = town_prices["price"] <= r["max_price"]
        rows.append({"percentile": r["percentile"], "towns_within_reach": int(ok.sum()),
                     "of_towns": len(town_prices)})
    return pd.DataFrame(rows)


def frontier_over_time(nat_prices: pd.DataFrame, loan, grants=0.0, assets_multiple=None) -> pd.DataFrame:
    """Minimum percentile able to finance the national reference flat, by year."""
    rows = []
    for _, r in nat_prices.iterrows():
        y = int(r["year"])
        if y < 2000:
            continue
        p, flag = min_percentile(float(r["price"]), y, loan, grants, assets_multiple)
        rows.append({"year": y, "min_percentile": p, "flag": flag,
                     "share_able_pct": (100 - p) if flag == "" else (0.0 if flag == ">" else 90.0)})
    return pd.DataFrame(rows)
