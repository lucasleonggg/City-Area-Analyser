"""Affordability Scenario Engine (EquiCity 3.0).

A household-level financing calculator: given a household's income, debts,
savings, CPF balance and grants, and a set of loan terms, what is the most it
could pay for a resale flat, what would the purchase cost upfront and every
month, and which towns' flats are within reach?

THIS IS A SCENARIO CALCULATOR, NOT AN ELIGIBILITY CHECK. Grants, CPF
balances and debts are user-entered assumptions. HDB's actual HDB Flat
Eligibility (HFE) letter assesses citizenship, income ceilings, first-timer
status, remaining lease, age and credit history, none of which is modelled.

RULES APPLIED (defaults; every one is adjustable)
-------------------------------------------------
* Loan by income: monthly instalment, computed at the ASSESSMENT rate,
  must not exceed MSR x gross income (30%, HDB flats). For bank loans the
  total of all debt payments must also be within TDSR x income (55%).
  Since 30 Sep 2022 HDB assesses loan eligibility at a 3% floor and banks
  at a 4% medium-term rate, both above what is actually charged.
* Loan by value: at most LTV x price (75% for both HDB and bank loans
  since 20 Aug 2024).
* Grants (CPF housing grants) are credited to CPF and reduce what must be
  financed; they cannot exceed the price.
* Downpayment: price - grants - loan. It can come from CPF OA or cash,
  except that a bank loan needs at least 5% of the price in cash.
* Buyer's Stamp Duty: residential rates from 15 Feb 2023 (1% on the first
  $180k, 2% next $180k, 3% next $640k, 4% next $500k, 5% next $1.5M, 6%
  above); payable from CPF.
* Other upfront costs (legal fees, option fee, valuation): a user-set cash
  amount, $5,000 by default.

Not modelled: cash-over-valuation, renovation, CPF accrued-interest
refunds on resale, the HDB loan income ceiling, ABSD (0% for a citizen's
first property).
"""

from dataclasses import dataclass, field

import numpy as np

BSD_BANDS = [(180_000, 0.01), (180_000, 0.02), (640_000, 0.03), (500_000, 0.04),
             (1_500_000, 0.05), (float("inf"), 0.06)]


def buyers_stamp_duty(price: float) -> float:
    left, duty = float(price), 0.0
    for width, rate in BSD_BANDS:
        part = min(left, width)
        duty += part * rate
        left -= part
        if left <= 0:
            break
    return duty


def annuity_payment(principal, rate, years):
    r, n = rate / 12, years * 12
    return principal / n if r == 0 else principal * r / (1 - (1 + r) ** -n)


def annuity_principal(payment, rate, years):
    r, n = rate / 12, years * 12
    return payment * n if r == 0 else payment * (1 - (1 + r) ** -n) / r


@dataclass(frozen=True)
class Loan:
    kind: str = "HDB"               # "HDB" or "Bank"
    rate: float = 0.026             # charged
    assess_rate: float = 0.03       # used for eligibility
    years: int = 25
    ltv: float = 0.75
    msr: float = 0.30
    tdsr: float = 0.55              # bank loans only
    min_cash_pct: float = 0.0       # 5% for bank loans


LOAN_PRESETS = {
    "HDB loan (2.6%, assessed at 3%, 25 yrs)": Loan(),
    "Bank loan (3.0%, assessed at 4%, 30 yrs)": Loan(kind="Bank", rate=0.03, assess_rate=0.04, years=30,
                                                    min_cash_pct=0.05),
}


@dataclass(frozen=True)
class Household:
    income: float = 8000.0          # gross monthly household income
    other_debt: float = 0.0         # monthly payments on car / personal loans etc.
    cash: float = 30_000.0
    cpf_oa: float = 60_000.0
    grants: float = 0.0
    other_upfront: float = 5_000.0  # legal fees, option fee, valuation (cash)


def max_loan_by_income(h: Household, loan: Loan) -> float:
    cap = loan.msr * h.income
    if loan.kind == "Bank":
        cap = min(cap, loan.tdsr * h.income - h.other_debt)
    return max(0.0, annuity_principal(max(cap, 0.0), loan.assess_rate, loan.years))


def evaluate(price: float, h: Household, loan: Loan) -> dict:
    """Everything about buying at `price`. `feasible` is True when the
    income test, the LTV cap and the upfront cash/CPF test all pass."""
    grants = min(h.grants, price)
    loan_cap_income = max_loan_by_income(h, loan)
    loan_amt = max(0.0, min(loan_cap_income, loan.ltv * price, price - grants))
    down = price - grants - loan_amt
    bsd = buyers_stamp_duty(price)
    cash_min = loan.min_cash_pct * price if loan_amt > 0 else 0.0
    # CPF can pay the downpayment beyond the cash minimum, and the BSD
    from_cpf = min(h.cpf_oa, max(0.0, down - cash_min) + bsd)
    cash_needed = down + bsd - from_cpf + h.other_upfront
    pay = annuity_payment(loan_amt, loan.rate, loan.years)
    binding = ("income (MSR/TDSR)" if loan_amt == loan_cap_income and loan_cap_income < loan.ltv * price
               else "loan-to-value cap")
    return {
        "price": price, "grants_used": grants, "loan": loan_amt, "downpayment": down,
        "stamp_duty": bsd, "cpf_used": from_cpf, "cash_needed": cash_needed,
        "cash_shortfall": max(0.0, cash_needed - h.cash),
        "monthly_payment": pay, "mortgage_to_income_pct": pay / h.income * 100 if h.income else np.inf,
        "loan_limited_by": binding, "feasible": cash_needed <= h.cash + 1e-6,
    }


def max_price(h: Household, loan: Loan, hi: float = 3_000_000.0) -> dict:
    """Largest price the household can complete, by bisection (feasibility
    is monotone in price)."""
    lo = 0.0
    if not evaluate(1.0, h, loan)["feasible"]:
        return evaluate(0.0, h, loan) | {"price": 0.0, "binding": "upfront costs exceed cash"}
    for _ in range(60):
        mid = (lo + hi) / 2
        if evaluate(mid, h, loan)["feasible"]:
            lo = mid
        else:
            hi = mid
    out = evaluate(lo, h, loan)
    # Beyond this price the extra cost lands on the downpayment, because the
    # loan is already capped -- by income or by LTV.
    out["binding"] = f"savings for the downpayment, with the loan capped by {out['loan_limited_by']}"
    return out
