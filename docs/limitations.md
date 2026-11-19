# Assumptions and limitations register

Version 3.0 · November 2026. Each entry: what is assumed or missing, which
results it affects, which way it is likely to bias them, and what would fix it.
Status: **Open** (affects current results), **Mitigated** (reduced, not removed),
**Fixed** (addressed in this version).

## A. Data

| # | Item | Affects | Likely direction | Status / fix |
|---|---|---|---|---|
| A1 | Resale transactions only; BTO prices are not in the data | All affordability results | Overstates the cost of entry for first-timers who get a BTO flat | Open. Add BTO launch prices if a verified source is found |
| A2 | 2026 incomes not yet published; 2025 distribution used for 2026 | 2026 affordability | Slightly pessimistic if 2026 incomes rose | Open, self-correcting when SingStat publishes |
| A3 | Income percentiles cover resident **employed** households only, P10–P90 | Shares of households able to finance a flat | Excludes retiree and non-working households; extremes are bounded at "<10th" / ">90th" | Open, stated on every chart |
| A4 | Census 2020 planning-area incomes include condo and landed residents and employer CPF | Local-income lens only | Makes central areas look richer than their HDB residents | Open, labelled as a relative lens |
| A5 | 136 blocks had no OneMap match; 16 rejected by the 5 km sanity check | Block-level analysis, geo features | Negligible (1.5% of blocks); those sales fall back to street/town | Mitigated |
| A6 | Station opening dates come from Wikipedia; LRT dates hand-checked | Time-correct rail access, event study | Small timing errors possible | Mitigated. Cross-check against LTA/NLB records |
| A7 | Ten Mile Junction LRT (1999–2019) absent from the 2025 LTA file | Rail access near Bukit Panjang | Understates access there, 1999–2018 | Open |
| A8 | Straight-line distances, not walking distance or journey time | Rail access, CBD distance, v1 commute proxy | Noise; under-states barriers such as expressways | Open. OneMap routing API would fix |
| A9 | Raw HDB filenames do not match their contents | None (pipeline reads by content) | — | Documented |

## B. Method

| # | Item | Affects | Likely direction | Status / fix |
|---|---|---|---|---|
| B1 | Town premium is shared across flat types (no town × type interaction) | Constant-quality prices by flat type | Smooths real differences, e.g. small flats in central towns | Open. Add interaction where cells are large enough |
| B2 | Reference flat (70 yrs lease, storey 8, standard size/model) is priced even where few such flats sell | Thin town-years (Bukit Timah, Central Area, early Punggol/Sengkang) | Higher variance | Mitigated: towns with <30 sales excluded; bootstrap intervals published |
| B3 | Log-price OLS gives a median-like level; no smearing correction | Price levels | ~0.5–1% low, uniform; no effect on comparisons | Documented |
| B4 | EquiCity's index drifts ~3% above HDB's official RPI by 2026 | Levels compared with official figures | Small | Documented; changes track closely (r = 0.83) |
| B5 | Equal weight per town in spatial statistics, not population-weighted | Moran's I, Theil, CV | Small towns count as much as large ones | Open. Population-weighted variant would test it |
| B6 | Town-level spatial statistics have only 26 units | Moran's I, Gi* | Gi* labels are fragile (see robustness); global trend is robust | Mitigated: 24 weight specs, bootstrap, block-level check |
| B7 | Block-level clustering does **not** show the rising trend seen at town level | Interpretation of "clustering intensified" | The rise is between towns/regions, not within neighbourhoods | Reported as a finding |

## C. Affordability, Scenario Engine and Policy Lab

| # | Item | Affects | Status |
|---|---|---|---|
| C1 | Financing terms held at today's rules for every year unless changed | Affordability time series | By design, to isolate prices vs incomes; historical LTVs available |
| C2 | CPF grants, savings and sale proceeds not modelled in the Affordability Index | "Cannot finance" means cannot borrow the full loan on income alone | Open there; modelled as user assumptions in the Scenario Engine |
| C3 | Scenario Engine is not an eligibility check (citizenship, income ceilings, first-timer status, age, remaining-lease rules, credit) | Scenario Engine | Stated on the page |
| C4 | Policy Lab is first-round arithmetic: prices and behaviour do not respond | Policy Lab | Stated; overstates lasting gains from looser credit or bigger grants |
| C5 | No BTO-supply experiment: no validated supply → price relationship | Policy Lab | Deliberately not built |
| C6 | Savings by income percentile have no public distribution | Policy Lab | Default assumes savings do not bind; a labelled multiple-of-income assumption is optional |

## D. Price model

| # | Item | Affects | Status |
|---|---|---|---|
| D1 | The model values flats at the current market level; it does not forecast the market | Any use beyond ~6 months | Stated; future values are scenarios with a historical market-error band |
| D2 | 80% intervals are calibrated for the next 6 months | Interval use further out | Coverage by horizon published |
| D3 | No renovation, condition, view, facing, exact unit or MOP information | Individual valuations | Open (not in public data) |
| D4 | Random forest in the back-test uses 30 trees (v1 used 100) | v1 comparison | Documented; forest error changes little beyond ~30 trees |
| D5 | Target encodings were random out-of-fold in v2 | v2 training | **Fixed** in v3: time-ordered |
| D6 | v2 used today's rail network for historical sales | v2 geo feature | **Fixed** in v3: time-correct |

## E. Interpretation

| # | Item | Status |
|---|---|---|
| E1 | Shapley values, partial dependence and location premia are associations, not causal effects | Stated on every explanation |
| E2 | The station event study shows rising prices **before** opening, so parallel trends fail; it is descriptive | Stated; a causal design needs announcement dates |
| E3 | The Access & Amenity composite is a choice of weights, not a finding (Spearman 0.70–0.94 across weightings) | Stated (v1 page) |
