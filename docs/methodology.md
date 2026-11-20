# EquiCity methodology

Version 3.2 · November 2026 · Lucas Leong Ting Wai

This document describes how every number in EquiCity is produced. Each
section names the script that implements it. The whole pipeline reruns with
`bash reproduce.sh`; inputs are checked against `raw_data/MANIFEST.csv`.

## 1. Data preparation (`pipeline/merge.py`, `clean.py`, `common.py`)

Five HDB resale files (Jan 1990 – Jul 2026) are merged by content, not by
filename (the original downloads were not named by the period they
contain). Town, flat type and
flat model spellings are standardised; remaining lease is parsed from text
where published and computed from lease start otherwise; the storey range is
reduced to its midpoint. Result: 982,011 transactions across 27 towns.

## 2. Constant-quality prices: yearly hedonic regressions (`pipeline/hedonic.py`)

For each calendar year *y*, by OLS on that year's transactions:

    log P = α_month + γ_town + δ_flat type + μ_flat model
            + β1·lease + β2·lease²/100 + β3·log(area) + β4·storey + β5·storey²/100 + ε

Flat models with fewer than 30 sales that year are pooled into "Other".
Coefficients are re-estimated every year, so the market's valuation of
lease, size and floor level may drift (Section 6 shows that it does).

**Reference flat.** Each year's fit prices the same flat in every town: 70
years lease left, storey 8, the flat type's standard area (2-room 45, 3-room
67, 4-room 93, 5-room 113, Executive 145 sqm) and its most common model over
the whole sample. This **constant-quality price** separates location from
flat mix, which raw medians confound.

**Location premium.** exp(γ_town − γ̄) − 1, where γ̄ is the
transaction-weighted average town effect using **fixed** 2015–2025 weights,
so shifts in where flats sell do not move the national level.

**National index.** α_month + γ̄ + the 4-room reference flat's attribute
terms, rebased to 1Q2009 = 100 (HDB's base). Validated against HDB's
official Resale Price Index: correlation of quarterly changes 0.83; levels
drift about 3% apart by 2026.

## 3. Affordability (`dashboard/analysis/affordability.py`)

For a household at income percentile *p* in year *y* buying a flat at price *P*:

* instalment = annuity payment on LTV·P at rate *r* over *n* years
* MSR = instalment / gross monthly household income (SingStat, excluding
  employer CPF — the right basis, since MSR is assessed on gross income);
* Affordability index = 100 × income / (instalment / MSR cap), the US
  National Association of Realtors' construction with Singapore's 30% cap;
* required percentile = the income percentile whose income exactly meets
  the cap (log-linear interpolation between published percentiles).

Defaults are today's HDB-loan terms (2.6%, 25 years, 75% LTV, 30% MSR),
held fixed across years to isolate the race between prices and incomes.
Incomes for 2026 use 2025, the latest published.

## 4. Scenario Engine and Policy Lab (`analysis/scenario.py`, `analysis/policy.py`)

**Scenario Engine.** The loan is the smallest of: the income limit (MSR,
and for bank loans TDSR net of other debts, at the *assessment* rate — 3%
for HDB loans, 4% for bank loans since Sep 2022); the LTV limit; and price
minus grants. The downpayment (price − grants − loan) and buyer's stamp
duty (2023 residential bands) are paid from CPF OA then cash, with at least
5% of the price in cash for bank loans; other upfront costs are cash. The
maximum price is found by bisection on feasibility. It is a calculator, not
an eligibility check.

**Policy Lab.** Applies the same arithmetic to households at each income
percentile, under baseline and user-chosen rules (rate, LTV, grant, MSR
cap), against observed constant-quality prices. Only the first-round
financing effect is computed: prices and behaviour do not respond. No
supply experiment is offered because no validated supply–price relationship
is available.

## 5. Spatial statistics (`analysis/spatial.py`, `pipeline/spatial_robustness.py`)

* **Global Moran's I** with permutation inference (999 reshuffles).
* **Getis-Ord Gi\*** with conditional permutation p-values (the focal unit
  held fixed) and Benjamini–Hochberg FDR across units.
* **Theil T** with exact between/within-region decomposition (URA regions).
* **Block level.** Each block's location premium = mean within-town
  residual from that year's hedonic fit + its town's premium, pooled over
  the last three years (≥5 sales); Gi\* with a 1 km band, analytic z, FDR.
* **Robustness.** Moran's I of the town premium every year 1995–2026 under
  25 specifications — 24 spatial-weight definitions (kNN k=3–10, distance
  bands 4/6/8 km, inverse distance; binary and row-standardised) plus a
  balanced panel of the 22 towns present every year — with Sen's slope and a
  Mann–Kendall trend test. Adjacent years' values are not independent, so
  the primary test is the Hamed–Rao (1998) corrected Mann–Kendall; trend-free
  and plain pre-whitening are reported as sensitivity checks only.
  `pipeline/trend_sensitivity.py` then (i) re-runs Hamed–Rao with lags 1–3
  only; (ii) simulates 4,000 trendless series per specification to ask how
  often persistence alone produces a rise as consistent as the observed one,
  under the estimated AR(1), a pessimistic AR(1) of 0.9, an estimated AR(2),
  and a bias-corrected AR(1) whose coefficient is redrawn from its sampling
  distribution for every simulated series (parameter uncertainty);
  (iii) fits the best single mean-shift break (segments ≥ 6 years) and
  compares it with a linear trend by BIC, with Sen slopes on each side; and
  (iv) measures each test's false-trend rate on trendless series of length
  32. The 25 specifications share data and neighbour structures: they are
  robustness checks, not independent replications;
  block-level Moran's I by 3-year window with
  0.5/1/2 km bands; 100 bootstrap re-fits of the hedonic regression per year
  (transactions resampled within year) for 95% intervals on premia and on
  Moran's I, and for the share of re-fits in which each town is a Gi\* hot
  or cold spot.

## 6. Market trends (`analysis/temporal.py`)

Town and region constant-quality series; growth between any two years with
a robust z-score (median and 1.4826×MAD across towns, |z| ≥ 2 flagged); the
lease curve exp(f(L) − f(90)) from each year's lease coefficients, drawn
only over leases actually traded that year.

## 7. Historical rail access (`pipeline/rail_history.py`)

Each transaction is matched to the nearest MRT/LRT station **open in its
month of sale** (184 stations; coordinates from LTA, opening dates from
Wikipedia, LRT dates hand-checked; interchanges take the earliest service).
Town access = share of *existing* blocks within 500 m / 1 km.

**Event study (descriptive).** Treated: blocks that first came within 1 km
of an open station between 1996 and 2022 and existed at least 8 years
before. Outcome: block-year mean within-town hedonic residual. Comparison:
same-town, same-year blocks more than 1 km from any open station and at
least 6 years from getting one. Path normalised to year −1; 95% intervals
from 300 bootstrap resamples of treated blocks. Because treated blocks were
already rising before opening, parallel trends fail and the result is not
causal.

## 8. Price model (`pipeline/price_model.py`, `audit_price_model.py`)

**Target.** log(price) − log(national index in the sale month). At
prediction the latest known index is applied. **Learner.** Histogram
gradient boosting (500 iterations, 63 leaves, learning rate 0.08). **Features.**
town, flat type, flat model (native categoricals), floor area, storey,
remaining lease, lease start, sale date, time-ordered street and block
encodings, block latitude/longitude, time-correct distance to rail,
distance to the CBD. **Intervals.** Conformalized quantile regression:
10th/90th percentile models widened by the margin that covers 80% of the
six months before the origin, using only information available then.

**Validation.**
* Ablation back-test: 8 annual origins (Jan 2019 – Jan 2026), 10-year
  training window, next 6 months tested, market index frozen at the last
  value before the origin. Seven model variants from a naive baseline to
  the served model.
* Horizon back-test: 14 half-yearly origins, served model, tested out to 24
  months; error and interval coverage by horizon.
* Market baselines: random walk, 12- and 36-month trend forecasts of the
  national index at 6/12/24 months from 300 monthly origins (2000–2024),
  each index rebuilt only from data before its origin.
* Subgroups: error, bias and coverage by town, flat type, region, lease band,
  price band and origin.

## 9. Explanations (`analysis/explain.py`, `pipeline/build_explanations.py`)

Exact interventional Shapley values over seven driver groups (2⁷ = 128
coalitions) against the served model, with a 100-row background of recent
sales; log contributions converted to dollars with the logarithmic-mean
(LMDI) rule, which adds up exactly and never blows up. Pairwise comparison
uses flat B as a single-row baseline. Partial dependence over 400 recent
sales. All are associations, not causal effects.
