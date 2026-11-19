# EquiCity 3.0 — what changed from v2, and why

v3 does not add more of the same. It makes v2 harder to criticise: reproducible
from a clean machine, corrected where v2 was anachronistic or leaky, tested for
robustness, and documented well enough for someone else to check.

## Reproducibility

| v2 problem | v3 fix |
|---|---|
| A reviewer's environment failed 3 of 40 tests: loading the model needed **pyarrow** (pandas 3 pickles strings as pyarrow arrays), and the model was trained with scikit-learn 1.9.1 while the environment had 1.8.0 | Model saved as estimators + numpy + plain Python only (`analysis/bundle_io.py`); a test loads it with pyarrow blocked. Every runtime dependency pinned exactly; the model records its training versions and the dashboard warns on mismatch |
| No automated tests on change | `.github/workflows/tests.yml` runs all tests on Python 3.12 and 3.13 from a clean install on every push |
| Inputs not pinned | `raw_data/MANIFEST.csv`: SHA-256 and source of every input; `pipeline/manifest.py --check` |
| No single rebuild path | `bash reproduce.sh` rebuilds everything from `raw_data/` |

Verified: clean install on Python 3.12 from `dashboard/requirements.txt`, on a
copy of the repo without the 132 MB transaction file — all tests pass,
including a headless render of every dashboard page.

## Corrections

| v2 | v3 |
|---|---|
| Distance to rail used **today's** network for every sale | Distance to the nearest station **open in the month of sale** (184 stations with opening dates) |
| Street/block target encodings used **random** out-of-fold splits (a 2017 sale could be encoded with 2018 comparables) | **Time-ordered** encodings: each sale sees only earlier sales |

The served model's out-of-time error moved from 4.18% to **4.10%** and RMSE
from $34.9k to **$32.0k**; 80% intervals cover **80.3%**.

## New capabilities

* **Affordability Scenario Engine** — household income, debts, cash, CPF,
  grants, HDB or bank loan (MSR, TDSR, LTV, assessment rate, 5% cash rule,
  stamp duty) → maximum price, upfront cash, monthly cost, and the share of
  each town's recent sales within reach. Labelled as a calculator, not an
  eligibility check.
* **Policy Lab** — mechanical what-ifs on rate, LTV, grant and MSR cap across
  the income distribution and over time. Shows that, holding savings out,
  grants do nothing for the borrowing constraint and a higher LTV can only
  hurt; both work through savings. No BTO-supply experiment (no validated
  supply–price model).
* **Historical Transport Access** — rail access by town and year; descriptive
  event study around station openings (+0.9% in the opening year, +3.9% six
  years on, but a pre-opening upward trend rules out a causal reading).
* **Spatial robustness** — 25 weight specifications, block-level windows,
  balanced panel, 100 bootstrap re-fits per year, Gi\* stability.
* **Validation** — 14 half-yearly origins out to 24 months (error 4.2% →
  12.0%, coverage 80% → 25%), market-index forecast baselines (a year ahead,
  ~5–7% error), predictor split into current valuation / scenario future value
  / market forecast.
* **Research & Data** — paper draft, methodology, data dictionary,
  limitations register, executed notebook, downloadable tables.

## What the robustness work changed in the findings

* Town-level clustering of location premia **rose in all 25 specifications**
  and on a balanced panel; bootstrap intervals (at most ±0.013) are far narrower than
  the rise. The direction is robust; the size varies (+0.03 to +0.21).
* **New:** at **block level** clustering did **not** rise (1 km band, about
  0.65–0.70 throughout). The intensification is between towns and regions,
  not within neighbourhoods.
* **New:** individual Gi\* hot spots are fragile (no town flagged in more than
  about half of bootstrap re-fits; labels change with k). v2's hot-spot maps
  should be read as indicative only.

---

# EquiCity 2.0 — what changed from v1, and why

Central research question:

> To what extent does Singapore's HDB resale market exhibit spatial inequality
> in housing affordability, and how has that inequality evolved over time?

Every number below was produced by the code in this repo
(`pipeline/build_analytics.py`, `pipeline/audit_price_model.py`,
`pipeline/build_explanations.py`) on the 982,011-transaction file.

---

## 1. Audit of v1: problems found

| # | Problem | Consequence | Fix |
|---|---|---|---|
| 1 | Price model predicted **raw dollars with `year` as a feature**. Trees cannot extrapolate, so every post-training sale is priced at the last training year's level. | Chronological R² 0.33, $120k mean error. | Predict price **relative to a market index**; apply the latest known index. |
| 2 | v1 headline accuracy (R² 0.94) came from a **random split**, which leaks future prices into training. | Overstated accuracy. | Rolling-origin back-test: train on the past, price the next 6 months, 8 origins 2019–2026. |
| 3 | Explanations used a **different model** (6-feature forest) from the one making predictions (38-feature forest). | The Shapley bars did not explain the number shown. | Exact Shapley against the served model. |
| 4 | Affordability used **per-member income × assumed 3.09 persons**. | 26% too high at the 10th percentile (2025: $3,501 vs real $2,779). Overstated affordability for exactly the households that matter. | SingStat **household-level** percentile series (data.gov.sg `d_650c42ad…`). |
| 5 | Affordability compared town **median prices**, which mix location with flat mix (young towns sell young flats). | Town comparisons confounded. | **Constant-quality** hedonic prices for an identical reference flat. |
| 6 | Spatial analysis on **26 hand-typed town centroids** only. | Coarse; no within-town signal. | Geocoded **9,830 of 9,982 blocks** (98.5%) via OneMap; block-level hot spots. |
| 7 | Gi* run on one variable with no **global** test first. | Local hot spots read without knowing whether any clustering exists. | Moran's I (permutation) on every variable, every year; Gi* generic. |
| 8 | **k-means cluster labels hard-coded** to cluster IDs ("0 = Underserved"). IDs are arbitrary. | Labels attached to whichever group k-means numbered 0. | Clusters ranked by mean score and described from their own centroids. |
| 9 | `plotly>=5.18` with `px.scatter_mapbox`, which **no longer exists in plotly 7**. | A fresh deploy crashes every map page. | `px.scatter_map`, `plotly>=5.24`. |
| 10 | Stale wording: "least / most equitable", "six dimensions" (there are five), "partial transit coverage" (all 187 stations are now loaded). | Contradicted the README's own naming note. | Corrected. |

## 2. Retained / modified / built / dropped

**Retained as is**: data cleaning pipeline, amenity loaders (LTA rail incl. LRT,
hospitals, all 28 polyclinics, estate age), access composite with its weight
sensitivity, permutation Gi* with FDR, the Intel benchmark and its honest
negative result, the test suite.

**Modified**: access map page (bugs 8–10), Intel page (now benchmark-only),
requirements pins, test runner (discovers every `tests/test_*.py`).

**Built from scratch**

| Module | Files |
|---|---|
| Hedonic backbone (37 yearly regressions) | `pipeline/hedonic.py`, `pipeline/build_analytics.py` |
| Model audit + served v2 model + CQR intervals | `pipeline/price_model.py`, `pipeline/audit_price_model.py` |
| Block geocoding | `pipeline/geocode_blocks.py`, `raw_data/hdb_block_coordinates.csv` |
| Global explanations, PDPs | `pipeline/build_explanations.py` |
| Affordability engine | `dashboard/analysis/affordability.py` |
| Spatial statistics (Moran's I, Gi*, Theil, Gini) | `dashboard/analysis/spatial.py` |
| Temporal helpers | `dashboard/analysis/temporal.py` |
| Explainer (exact group Shapley, pairwise compare) | `dashboard/analysis/explain.py` |
| Seven new pages | `dashboard/views/{overview,predictor,affordability_index,spatial_inequality,market_trends,explainability,model_validation}.py` |
| 21 new tests | `tests/test_v2.py` |

**Dropped**: `views/price_trends.py` and `views/equity_analysis.py`
(superseded by Market Trends and Affordability Index; raw medians are still a
tab on Market Trends).

**Deliberately not built** (low value for the research question): XGBoost as a
separate dependency (scikit-learn's HistGradientBoosting is the same family and
needs nothing extra); deep learning; a block-level income estimate (no public
data at that resolution — inventing one would repeat v1's fabricated-factor
mistake).

## 3. Results

### Price model (rolling back-test, 98,074 sales, 8 origins 2019–2026)

| Model | Mean abs. % error | Within 10% | R² |
|---|---|---|---|
| Naive: last 12 months' town × type median | 12.0% | 54% | 0.71 |
| v1 random forest | 9.4% | 63% | 0.84 |
| Linear hedonic | 7.4% | 73% | 0.91 |
| v1 forest, market-indexed target | 7.9% | 71% | 0.89 |
| Gradient boosting, indexed | 4.8% | 89% | 0.96 |
| + street/block comparable-sales level | 4.3% | 93% | 0.96 |
| **+ block coordinates & rail distance (v2)** | **4.2%** | **93%** | **0.97** |

* **80% intervals cover 80.2%** of out-of-time sales (76–85% across origins,
  76–84% across flat types).
* **v1's chronological test explained.** Re-run with v2 and the market frozen at
  Dec 2018: R² 0.29, no better than v1. Give the same 2018-trained model the
  observed market level: R² 0.87 up to 7.5 years out. The failure was asking a
  cross-sectional model to forecast the market, not the algorithm.
* Weakest: Marine Parade, Bishan, Bukit Timah (thin, heterogeneous central
  markets); leases under 50 years.

### Market index validation
EquiCity's constant-quality index, built independently from the public
transaction file, tracks HDB's official RPI (1Q2009 = 100): quarterly-change
correlation 0.83; levels drift ~3% apart by 2026.

### Research findings (4-room reference flat, HDB loan at 2.6% / 25 yrs / 75% LTV)

1. **Affordability for the median household** is close to its 2013 low and
   worse than any point in 2015–2022: mortgage share of income 19.7% in 2026
   (2013: 21.2%; 2019 trough: 15.4%). The income percentile needed to meet the
   30% MSR cap rose from P25 (2019) to P32.
2. **Lower-income households were hit harder.** At the 30th percentile the
   reference 4-room is within the cap in **9 of 26 towns** (2019: 16; 2013: 2).
3. **Spatial clustering of prices has intensified.** Moran's I of the
   location premium rose from about 0.40 (1995) to 0.57–0.60 (2022–2026),
   significant in every year (p ≤ 0.001). Expensive towns increasingly sit
   next to expensive towns.
4. **The gap is increasingly regional.** The share of cross-town price
   inequality lying *between* URA regions (Theil decomposition) rose from about
   71% to 78–80%.
5. **But dispersion narrowed during the 2021–26 boom.** Cross-town price CV
   peaked at 23.5% (2019) and fell to about 17–19%, as cheaper northern and
   western towns grew fastest (Sembawang flagged as unusually fast 2019–26;
   Queenstown and Bukit Merah unusually slow). Clustering rose while the gap
   compressed — the cheap towns rose together.
6. **Lease decay is priced more steeply.** All else equal, a flat with 60
   years left sold for 88% of a 90-year flat in 2004 and about 72–76% since
   2019.
7. **Block level.** Location premium is strongly clustered at block scale too
   (Moran's I 0.64, 5,957 blocks, 1 km band).

These are associations. The hedonic and machine-learning location premia agree
(Spearman ρ = 0.94 across towns), which makes the spatial pattern robust to
method, not causal.

## 4. Known limitations (new in v2)

* Financing terms are held at today's rules for every year unless changed;
  CPF grants, savings and sale proceeds are not modelled.
* Affordability is buyer-side (national income distribution). The Census 2020
  local-income lens includes private-housing residents.
* 2026 uses the 2025 income distribution (2026 not yet published).
* Distance to rail uses today's network for historical sales, so lines opened
  after a sale (e.g. Thomson-East Coast) are anachronistic in the back-test.
* Town effects are shared across flat types (no town × type interaction).
* Block geocoding: 136 blocks returned no OneMap match and 16 were rejected
  by the sanity check; their sales fall back to street/town location.
