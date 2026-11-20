# EquiCity data dictionary and source register

Version 3.0 · November 2026. Every input is listed with its source and
SHA-256 hash in `raw_data/MANIFEST.csv`; `python pipeline/manifest.py --check`
verifies them.

## 1. Source register

| Input | Publisher | Coverage | Used for |
|---|---|---|---|
| HDB resale flat prices (`raw_data/hdb-*.csv`) | HDB via Data.gov.sg | Jan 1990 – Jul 2026, 982,011 transactions | Everything |
| Household employment income at selected percentiles, excl. employer CPF (`singstat_household_…_2000_2025.csv`) | SingStat via Data.gov.sg `d_650c42ad53d69ad3f6f47f04c21616fb` | 2000–2025, P10–P90, resident employed households | Affordability, Scenario Engine, Policy Lab |
| Resident households by planning area and household income from work (`census2020_…csv`) | SingStat Census 2020 via Data.gov.sg `d_4e150bbd35c3145ea43019212c7b46c2` | 2020, 30 planning areas | Local-income lens |
| HDB Resale Price Index (`hdb_resale_price_index_official_1q2009_100.csv`) | HDB via Data.gov.sg `d_14f63e595975691e7c24a27ae4c07c79` | 1Q1990 – 3Q2026 | Validation of EquiCity's index |
| MRT/LRT station exits (`lta-mrt-station-exits.geojson`) | LTA via Data.gov.sg | Aug 2025 snapshot | Station coordinates |
| Station opening dates (`rail_stations_opening_dates.csv`) | Derived: Wikipedia station lists and articles, LRT dates hand-checked | 184 stations, 1987–2026 | Time-correct rail access |
| Block coordinates (`hdb_block_coordinates.csv`) | Derived: OneMap Search API (SLA) | 9,830 of 9,982 blocks | Block-level analysis, rail and CBD distance |
| Hospitals, polyclinics, estate age, v1 station list | Various (see v1 README) | 2024–2025 | Access & Amenity Index (supporting page) |
| BTO supply pipeline (`bto-supply-pipeline.csv`) | v1 file, provenance unverified | 2021+ | **Not used** in any v3 analysis |

## 2. Derived variables

| Variable | Definition |
|---|---|
| `month_idx` | Months since January 1990 (Jan 1990 = 0) |
| `t` | Decimal sale date, e.g. 2026.5 = July 2026 |
| `storey_mid` | Midpoint of the published storey range (e.g. "07 TO 09" → 8) |
| `remaining_lease` | Years left on the 99-year lease at sale, from the published value or lease start |
| `block_key` | `"<street name> \| <block>"`, the block identifier |
| `region` | URA planning region of the town (Central, East, North, North-East, West) |
| `te_street`, `te_block` | Smoothed mean of index-deflated log price per sqm of **earlier** sales on the street / in the block (time-ordered target encoding; pseudo-counts 30 and 10) |
| `dist_mrt_km` | Straight-line distance to the nearest MRT/LRT station **open in the month of sale** |
| `dist_cbd_km` | Straight-line distance to Raffles Place |
| Reference flat | 70 years lease left, storey 8, flat type's standard area (2R 45, 3R 67, 4R 93, 5R 113, Exec 145 sqm) and most common model |

## 3. Analytics tables (`processed_data/analytics/`)

| Table | Produced by | Key columns |
|---|---|---|
| `hedonic_town_year_type.csv` | `build_analytics.py` | `price_ref` constant-quality price; `town_effect`, `town_se` (log); `location_premium_pct` vs fixed-weight national; `n_town`, `n_cell` sales behind it |
| `hedonic_national_month.csv` | `build_analytics.py` | `log_index`, `index_2009q1_100`, `price_ref_4room` |
| `hedonic_region_year.csv` | `build_analytics.py` | `price_ref_4room` by URA region |
| `lease_curve.csv` | `build_analytics.py` | `rel_price_vs_90` by remaining lease; `in_support` = lease range actually traded |
| `hedonic_fit_summary.csv` | `build_analytics.py` | `r2`, `resid_sd_log`, `area_elasticity`, `lease_60_vs_90`, `storey_14_vs_5_pct` |
| `observed_town_type_year.csv` | `build_analytics.py` | raw medians and counts |
| `lease_band_year.csv` | `build_analytics.py` | median $/sqm by lease band |
| `recent_sales.csv` | `build_analytics.py` | last 24 months of transactions (predictor comparables) |
| `block_premium.csv` | `build_analytics.py` | `premium_pct`, `within_town_pct`, `gi_z`, `p_fdr`, `hotspot` |
| `rail_access_town_year.csv` | `build_analytics.py` | `mean_dist_km`, `share_within_500m`, `share_within_1km` (existing blocks only) |
| `rail_events.csv`, `rail_event_study.csv` | `build_analytics.py` | event year; `effect_pct`, `lo95_pct`, `hi95_pct` by years from opening `k` |
| `audit_backtest_overall.csv`, `audit_backtest_by_origin.csv` | `audit_price_model.py` | `mae`, `rmse` ($), `mape_pct`, `median_ape_pct`, `within_5pct`, `within_10pct`, `bias_pct`, `r2`; `mode` = forecast (frozen index) or nowcast (observed index) |
| `audit_subgroups.csv` | `audit_price_model.py` | error, bias, `coverage80_pct`, `median_width_pct` by town / type / region / lease / price band / origin |
| `audit_horizons.csv`, `audit_halfyear_origins.csv` | `audit_price_model.py` | error and coverage by months ahead |
| `audit_legacy_split.csv` | `audit_price_model.py` | v1's train≤2018 test re-run |
| `market_forecast_baselines.csv`, `market_forecast_errors.csv.gz` | `audit_price_model.py` | index forecast errors (%) by method and horizon |
| `global_shapley.csv`, `shapley_sample.csv`, `pdp_*.csv`, `model_town_premium.csv` | `build_explanations.py` | model explanations |
| `robust_moran_specs.csv`, `robust_moran_summary.csv` | `spatial_robustness.py` | Moran's I by year and weight specification; Sen slope; Mann-Kendall z and p (original, assumes independent years); `lag1_acf_detrended` (lag-1 autocorrelation after removing the Sen trend); `mk_hamed_rao_p` (**primary**, corrected for serial correlation); `mk_tfpw_p`, `mk_prewhitened_p` (sensitivity only; over-reject or under-reject at n = 32) |
| `robust_trend_sensitivity.csv` | `trend_sensitivity.py` | Per specification: fitted AR(1) persistence and its small-sample bias correction; AR(2) coefficients; Hamed–Rao p with all significant lags and with lags 1–3; simulated p-values = share of 4,000 trendless series whose Mann–Kendall S is at least the observed S (one-sided): `sim_p_fitted_phi` (AR(1) as estimated), `sim_p_phi_0_9` (AR(1), pessimistic 0.9), `sim_p_ar2` (AR(2) as estimated), `sim_p_phi_uncertain` (bias-corrected AR(1) drawn from its sampling distribution for every series) |
| `robust_trend_breaks.csv` | `trend_sensitivity.py` | Per specification: best single mean-shift break (each side ≥ 6 years), BIC of the step model and of a linear trend, the model BIC prefers, and the Sen slope and Mann–Kendall p on each side of the break |
| `robust_trend_test_size.csv` | `trend_sensitivity.py` | False-trend rate at 5% of each test on 1,000 trendless AR(1) series of length 32, by persistence (ideal 0.05) |
| `robust_block_moran.csv` | `spatial_robustness.py` | block-level Moran's I by 3-year window and band |
| `robust_moran_ci.csv`, `robust_premium_ci.csv` | `spatial_robustness.py` | bootstrap 95% intervals |
| `robust_gi_stability.csv`, `robust_gi_by_k.csv` | `spatial_robustness.py` | share of bootstrap re-fits flagging each town; labels by k |
| `robust_theil_by_type.csv` | `spatial_robustness.py` | Theil T and between-region share by flat type |

## 4. The served model (`processed_data/price_model_v2.joblib`)

Stored as scikit-learn estimators, numpy arrays and plain Python types only
(no pandas objects), with the library versions used to train it. Contents:
point model, two quantile models and the conformal margin; category lists;
street/block encodings; the national log index; a 400-row background sample
for Shapley values; the block lookup table; stations open at the last data
month.
