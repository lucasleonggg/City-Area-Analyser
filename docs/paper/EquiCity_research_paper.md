# The Evolution of Spatial Inequality and Housing Affordability in Singapore's HDB Resale Market, 1990–2026

**Lucas Leong Ting Wai** · Working paper, version 3.2 (draft) · November 2026

*Not peer reviewed. Comments on the methods are actively sought. All results
reproduce from public data with `bash reproduce.sh` in the EquiCity
repository; every number below is recomputed in
`notebooks/reproduce_headline_findings.ipynb`.*

---

## Abstract

Using all 982,011 HDB resale transactions from January 1990 to July 2026, I
estimate constant-quality prices for an identical reference flat in every
town and year with yearly hedonic regressions, combine them with household
income percentiles from SingStat, and ask how unequal resale affordability is
across space and how that has changed. Three findings stand out. First, for
the median resident employed household, a constant-quality 4-room flat
financed with a standard HDB loan costs 19.7% of income in 2026, up from
15.4% in 2019 and close to the 2013 peak of 21.2%; at the 30th income
percentile it is within the 30% Mortgage Servicing Ratio cap in 9 of 26
towns, down from 16 in 2019. Second, location premia have become more
spatially clustered at town level — Moran's I rose from about 0.48 in
1995–99 to 0.60 in 2022–26 — and this rise holds under all 25 robustness
specifications (24 definitions of which towns count as neighbours, plus a
balanced panel that excludes new towns) and under bootstrap resampling, and stays significant after correcting
for serial correlation in all but the balanced panel. Yet the same statistic computed across blocks shows
no comparable rise: the intensification is a between-town, increasingly
between-region phenomenon (the between-region share of cross-town price
inequality rose from about 71% to 78–80%), not a sharpening within
neighbourhoods. Third, the dispersion of prices across towns peaked in 2019
and narrowed during the 2020–26 boom, as the cheapest northern towns
appreciated fastest (North region +73% versus Central +52%, 2019–26).
Clustering and convergence coexist: cheap towns rose together. Supporting
analyses document a steepening lease discount and a descriptive association
between new rail stations and nearby prices, which pre-opening trends prevent
me from interpreting causally.

## 1. Introduction

Most Singaporean households live in HDB flats, and the resale market is where
households who cannot or do not wait for a new flat buy. Debate about resale
affordability usually turns on national figures — the Resale Price Index, the
ratio of a median price to a median income. National figures hide two things
this paper sets out to measure: *where* affordability differs, and *for whom*.

**Research question.** To what extent does Singapore's HDB resale market
exhibit spatial inequality in housing affordability, and how has that
inequality evolved over time?

The question needs three ingredients that are easy to get wrong.

1. **A price that compares like with like.** Town medians mix location with
   the kind of flats that happen to sell: a young town sells young flats.
   I use yearly hedonic regressions to price one reference flat everywhere.
2. **A definition of affordability tied to how flats are actually financed.**
   I use the binding constraint in Singapore's rules, the 30% Mortgage
   Servicing Ratio (MSR), evaluated across the household income distribution
   rather than at the median only.
3. **Spatial statistics that are tested, not just mapped.** With 26 towns,
   any map shows patterns. I test for clustering globally before locating
   it, correct for multiple testing, and check whether conclusions survive
   different definitions of "nearby", different spatial units, and sampling
   uncertainty.

**Contributions.** (i) A constant-quality town × year price panel for
1990–2026 built only from public data, validated against HDB's official
index. (ii) An affordability measure across the income distribution,
by town and year. (iii) A robustness-tested account of how spatial clustering
of prices has changed, including the finding that the change is between
towns and regions rather than within neighbourhoods. (iv) An open, fully
reproducible platform (EquiCity) that lets others re-run every step and
explore financing scenarios.

## 2. Data

* **Transactions.** HDB resale flat prices from Data.gov.sg, January 1990 –
  July 2026: 982,011 sales with town, street, block, flat type and model,
  floor area, storey range, lease start and price.
* **Incomes.** SingStat's monthly household employment income (excluding
  employer CPF) among resident employed households at the 10th–90th
  percentiles, 2000–2025. Excluding employer CPF matches the gross income on
  which MSR is assessed. A household-level series is used; per-member
  income multiplied by an average household size, as in this project's
  first version, overstates the 10th percentile by about a quarter.
* **Local incomes.** Census 2020 household income from work by planning area
  (used only as a secondary lens).
* **Space.** Block coordinates from OneMap (9,830 of 9,982 blocks); MRT/LRT
  station coordinates from LTA with opening dates compiled from public
  sources (184 stations).
* **Validation.** HDB's Resale Price Index, 1Q1990–3Q2026.

## 3. Methods

Full details, with the script implementing each step, are in
`docs/methodology.md`. In brief:

**Constant-quality prices.** For each year, log price is regressed on month,
town, flat type and flat model indicators and on lease (quadratic), log floor
area and storey (quadratic). Each year's fit prices a reference flat (70 years
lease left, storey 8, standard size and model) in every town. A town's
*location premium* is its effect relative to a fixed-weight national average.

**Affordability.** The monthly instalment on a 75%-LTV, 25-year loan at 2.6%
is divided by gross household income at each percentile. A flat is
*financeable* for a household if the instalment is within 30% of income.
Terms are held at today's rules for all years so the series isolates prices
against incomes.

**Spatial inequality.** Global Moran's I (permutation inference) on town
location premia each year; Getis-Ord Gi\* for local hot and cold spots with
conditional permutation p-values and Benjamini–Hochberg correction; Theil T
decomposed into between- and within-region parts; the coefficient of
variation of constant-quality prices. Robustness: 25 specifications (24
spatial-weight definitions plus a balanced panel); block-level Moran's I in 3-year windows
at three distance bands; 100 bootstrap re-fits per year. Trends in the yearly
Moran's I series are tested with Mann–Kendall; because adjacent years are
correlated, the Hamed–Rao correction is the primary test, checked against
simulations of how often persistence alone (AR(1), AR(2), and AR(1) with
parameter uncertainty) produces a comparable rise, and against a
single-break step model.

## 4. Results

### 4.1 The constant-quality index reproduces the official one

EquiCity's national index, built independently from the public transaction
file, tracks HDB's Resale Price Index closely: the correlation of quarterly
changes is 0.83, and the turning points (1996–97 peak, 2013 peak, 2019
trough, 2020–25 boom, 2025–26 plateau) coincide. Levels drift about 3% apart
by 2026, so the paper compares changes and cross-sections, not levels with
official figures. This validation matters because every subsequent result
rests on the same hedonic machinery.

### 4.2 Affordability has worsened since 2019, most for lower incomes

| Year | Constant-quality 4-room | Mortgage share of income, P50 | at P20 | Income percentile needed |
|---|---|---|---|---|
| 2000 | $229,000 | 19.5% | 39.0% | P30 |
| 2005 | $209,000 | 16.4% | 35.5% | P25 |
| 2013 | $437,000 | 21.2% | 48.7% | P36 |
| 2019 | $378,000 | 15.4% | 35.8% | P25 |
| 2023 | $525,000 | 18.5% | 42.9% | P30 |
| 2026 | $613,000 | 19.7% | 45.0% | P32 |

*HDB loan at 2.6%, 25 years, 75% LTV. 2026 uses 2025 incomes.*

For the median household the reference 4-room is financeable throughout,
but the margin has narrowed sharply since 2019. The burden falls unevenly:
at the 20th percentile the mortgage would take 45% of income, far above the
cap. The number of towns in which a 30th-percentile household can finance the
reference 4-room fell from 16 of 26 in 2019 to 9 in 2026 — better than the
2013 low of 2, but the direction since 2019 is clear.

### 4.3 Location premia have become more clustered — between towns

Town-level Moran's I of the location premium rose steadily over three
decades.

* **Size and uncertainty.** With k = 5 nearest neighbours, I rose from 0.45
  (1995) to 0.60 (2026); 95% bootstrap intervals are at most ±0.013, so the
  change is far larger than sampling noise.
* **Specification.** Clustering in 2022–26 exceeds 1995–99 in all 25
  specifications: 24 spatial-weight definitions (kNN with k = 3–10,
  distance bands of 4, 6 and 8 km, inverse distance, each binary and
  row-standardised) plus a balanced panel,
  and every one shows an upward trend that the ordinary (unadjusted)
  Mann–Kendall test calls significant; the next point shows how much of that
  survives a correction for serial correlation. The *size* of the rise
  depends on the specification — from +0.03 with inverse-distance weights to
  +0.21 with a 4 km band — so the honest claim is directional.
* **Serial correlation.** The ordinary Mann–Kendall test treats the 32
  yearly values as independent, but adjacent years are correlated (lag-1
  autocorrelation of the detrended series up to 0.82), which overstates
  significance. The primary adjusted test is the Hamed–Rao (1998) variance
  correction: the rise remains significant in 24 of 25 specifications
  (p ≤ 0.0001), the exception being the balanced panel, the most
  autocorrelated series, at p = 0.057 (0.070 if only lags 1–3 are used).
* **Short-series checks.** Thirty-two observations are few for estimating
  serial dependence. On simulated trendless series of that length, every
  Mann–Kendall variant over-rejects when persistence is high: at an AR(1)
  coefficient of 0.8, Hamed–Rao reports a false trend 34% of the time and
  trend-free pre-whitening 60%, so the latter's 25 of 25 is not cited as
  support. I therefore ask directly how often persistence alone, with no
  trend, produces a rise as consistent as the observed one (4,000
  simulations per specification). With each specification's estimated AR(1)
  persistence, p ≤ 0.0002 for 24 of 25; with an AR(2) instead, p ≤ 0.0002
  for 24 of 25; with the AR(1) coefficient bias-corrected for the short
  sample and drawn afresh for every simulation to carry its estimation
  uncertainty, p ≤ 0.001 for 24 of 25; and with a deliberately pessimistic
  persistence of 0.9, 23 of 25 stay below 0.05 (inverse-distance weights
  reach 0.055). The balanced panel fails all four (p = 0.07–0.14).
* **Steady rise or one-off shift?** A single level shift could mimic a
  trend. Fitting the best one-break step model to each series, a linear
  trend is preferred by BIC in 22 of 25 specifications, and the Sen slope is
  positive on both sides of the best break in all 25. The exceptions are
  kNN with k = 10 and, notably, the balanced panel, which is flat until 2018
  and rises after 2019 (0.11 per decade): on the 22 continuously present
  towns the increase is concentrated in the recent boom rather than spread
  over three decades.
* **What this adds up to.** The 25 specifications share the same
  transactions and overlapping neighbour structures, so they are robustness
  checks, not 25 independent confirmations. For the full set of towns the
  rise survives every check above; on the balanced panel alone it is
  suggestive, recent, and not established.
* **New towns.** Restricting to the 22 towns with 30+ sales in every year
  from 1995 still gives a rise (0.46 → 0.56), so the direction is not an
  artefact of peripheral new towns entering the sample, although on this
  smaller panel the trend is only borderline once serial correlation is
  allowed for.
* **A different unit.** At block level the picture changes. Moran's I across
  roughly 4,900–6,900 blocks with a 1 km band is high (about 0.65–0.70) but
  essentially flat from 1996–98 to 2023–25. Neighbouring blocks have always
  been priced alike; what has changed is how towns relate to *other towns*.

The Theil decomposition points the same way: the share of cross-town price
inequality lying between URA regions rose from about 71% in the mid-1990s to
78–80% in 2024–26. The spatial divide in resale prices is increasingly a
regional one — Central versus the rest — rather than a patchwork.

**Local hot spots are fragile.** Gi\* labels change with the number of
neighbours and across bootstrap re-fits: in 2026 no town is a significant hot
spot in more than half of 100 re-fits (Queenstown and Bukit Merah 50%), and no
town is a cold spot in more than 28% (Woodlands). The global trend is
robust; the identity of individual "hot spots" is not, and I do not rely
on it.

### 4.4 Clustering rose while the gap narrowed

Price dispersion across towns (coefficient of variation of the reference
4-room) rose from about 13% in the mid-1990s to a peak of 23.5% in 2019, then
fell to 17–19% in 2021–26. The 2020–26 boom lifted the cheapest regions most:
North +73%, North-East +65%, West +62%, East +59%, Central +52% (2019–26).
Sembawang's growth was flagged as unusually fast relative to other towns
(robust z ≥ 2), Queenstown's and Bukit Merah's as unusually slow.

These two facts are consistent. Moran's I measures whether *neighbours are
alike*, the coefficient of variation measures *how far apart* towns are.
When neighbouring cheap towns all rise together, the gap narrows while the
regional pattern sharpens. Spatial inequality in this market has become
more *organised* even as it became, since 2019, less *extreme*.

### 4.5 The lease discount has steepened

Holding everything else equal, a flat with 60 years left sold for 88% of a
90-year flat's price in 2004, but for 72–76% in every year since 2019. The
discount steepened most in 2017–19, a period of prominent public discussion
of lease expiry. Because older flats also differ in design, condition and
estate, the coefficient describes a market regularity rather than the causal
effect of the lease alone.

### 4.6 Rail access: large gains, a modest and non-causal price association

Measured against the network open at the time, the share of existing HDB
blocks within 1 km of a station rose from 56% in 1990 to 85% in 2026, with
large gains in Bedok, Hougang, Kallang/Whampoa, Marine Parade and Bukit Timah
from the North East, Circle, Downtown and Thomson–East Coast lines. Blocks that first came within
1 km of an open station between 1996 and 2022 (1,802 blocks) sold for about
0.9% more than same-town comparison blocks in the opening year and 3.9% more
six years later, relative to the year before opening. But these blocks were
already gaining before opening (−1.2% six years before, rising steadily to
zero), consistent with prices responding to announcements years ahead or with
lines being routed to areas that were changing anyway. The parallel-trends
condition a difference-in-differences design needs does not hold, so this is
a description, not an estimate of what stations cause.

## 5. Alternative explanations

* **Composition.** Rising clustering could reflect which towns exist, not
  how prices relate. On the balanced panel of 22 towns the rise is smaller
  (0.46 → 0.56), concentrated after 2019 and, once serial correlation is
  allowed for, only borderline significant, so new-town entry cannot be
  fully ruled out as a contributor to the long-run rise, though the
  direction is the same.
* **Measurement.** The reference-flat approach assumes the town premium is
  the same for all flat types. If central towns' premia are larger for small
  flats, the regional divide for those flats is understated, not overstated.
* **Mechanical correlation with distance to the centre.** Clustering of
  premia is partly a smooth centre–periphery gradient. The block-level
  stability shows the gradient has not become *sharper* within
  neighbourhoods; the rising town-level statistic reflects towns sorting more
  cleanly along it.
* **Policy changes.** Financing terms tightened after 2021 (HDB-loan LTV from
  90% to 75%). Holding terms fixed, as here, understates the deterioration
  for buyers who need the maximum loan.

## 6. Limitations

The full register is in `docs/limitations.md`. The most important: resale
prices only (no BTO prices); incomes for resident employed households only,
bounded at the 10th and 90th percentiles; grants, CPF savings and sale
proceeds not modelled in the affordability series (they are user inputs in
EquiCity's Scenario Engine); town premia shared across flat types; equal
weighting of towns in spatial statistics; straight-line distances; 2026
incomes not yet published.

## 7. Conclusion

Resale affordability in Singapore has deteriorated since 2019, with the
burden concentrated on lower-income households: the share of towns where a
30th-percentile household can finance a constant-quality 4-room has almost
halved. Spatially, the market has become more clearly divided along regional
lines even as the gap between cheap and dear towns narrowed during the
recent boom. The first conclusion is about levels and is robust to method;
the second is about structure, and is robust in direction but not in size or
in which towns are singled out as hot spots. Policy discussion that relies on
national averages, or on maps of individual hot spots, misses both.

## Reproducibility statement

All code, aggregated data, and the served model are in the EquiCity
repository. `bash reproduce.sh` rebuilds every table from the raw inputs,
whose SHA-256 hashes are recorded in `raw_data/MANIFEST.csv`; dependency
versions are pinned exactly; a continuous-integration workflow runs the
test suite on Python 3.12 and 3.13 on every change.

## References

Ang, B. W. (2004). Decomposition analysis for policymaking in energy: which
is the preferred method? *Energy Policy*, 32(9), 1131–1139.

Anselin, L. (1995). Local indicators of spatial association — LISA.
*Geographical Analysis*, 27(2), 93–115.

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate.
*Journal of the Royal Statistical Society B*, 57(1), 289–300.

Getis, A., & Ord, J. K. (1992). The analysis of spatial association by use of
distance statistics. *Geographical Analysis*, 24(3), 189–206.

Hamed, K. H., & Rao, A. R. (1998). A modified Mann–Kendall trend test for
autocorrelated data. *Journal of Hydrology*, 204(1–4), 182–196.

Mann, H. B. (1945). Nonparametric tests against trend. *Econometrica*, 13(3),
245–259.

Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized quantile
regression. *Advances in Neural Information Processing Systems* 32.

Rosen, S. (1974). Hedonic prices and implicit markets. *Journal of Political
Economy*, 82(1), 34–55.

Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's
tau. *Journal of the American Statistical Association*, 63(324), 1379–1389.

Shapley, L. S. (1953). A value for n-person games. In *Contributions to the
Theory of Games II*, 307–317.

Theil, H. (1967). *Economics and Information Theory*. North-Holland.

von Storch, H. (1995). Misuses of statistical analysis in climate research. In
*Analysis of Climate Variability* (pp. 11–26). Springer.

Yue, S., Pilon, P., Phinney, B., & Cavadias, G. (2002). The influence of
autocorrelation on the ability to detect trend in hydrological series.
*Hydrological Processes*, 16(9), 1807–1829.

Data: Housing & Development Board, Singapore Department of Statistics, Land
Transport Authority, Singapore Land Authority (OneMap), via Data.gov.sg.
