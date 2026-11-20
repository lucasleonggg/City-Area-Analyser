# EquiCity 3.0 — Singapore HDB Resale Affordability & Spatial Inequality

> **Research question.** To what extent does Singapore's HDB resale market
> exhibit spatial inequality in housing affordability, and how has that
> inequality evolved over time?

A Streamlit platform built on 982,011 real HDB resale transactions
(Data.gov.sg, 1990–2026), SingStat household income percentiles (2000–2025),
OneMap block coordinates and LTA rail data. **See [CHANGES.md](CHANGES.md) for
what changed in each version, [docs/paper](docs/paper/EquiCity_research_paper.md)
for the research paper draft, and [docs/](docs/) for the methodology, data
dictionary and limitations register.**

| Module | What it answers |
|---|---|
| Price Predictor | How much might this flat sell for? (with an 80% interval) |
| Affordability Index | Can a household at percentile P finance a flat in town T under the 30% MSR cap? |
| Spatial Inequality | Where are disparities concentrated (Moran's I, Gi*, town and block level), and are they widening? |
| Scenario Engine | With this income, savings, CPF and grants, what can a household afford, and where? |
| Policy Lab | What do rate, LTV, grant or MSR changes do, mechanically, across the income distribution? |
| Transport Access | How did rail access change, and what happened to nearby prices? |
| Market Trends | How have constant-quality prices, regional gaps and the lease discount changed? |
| Research & Data | Paper, methodology, data dictionary, limitations, downloads |
| Explainability | What drives the model, and why do two similar flats differ? (exact Shapley) |
| Model Validation | How accurate is the model when tested on the future? |

Headline: the served price model's out-of-time error is **4.1%** (v1: 9.4%),
and its 80% intervals cover **80.3%** of real future sales.

## Rebuilding everything

```bash
pip install -r dashboard/requirements.txt -r requirements-validation.txt   # exact pins
bash reproduce.sh                  # verify inputs, rebuild every table and the model, run tests
streamlit run dashboard/app.py     # ?page=scenario, ?page=policy ... deep-link to a module
python -m pytest tests/            # 219 tests, incl. 150 checks of the trend tests against
                                   # pymannkendall and a headless render of every page
```

Inputs are checked against SHA-256 hashes in `raw_data/MANIFEST.csv`.
CI (`.github/workflows/tests.yml`) runs the tests on Python 3.12 and 3.13 on
every push.

The pipeline needs `processed_data/hdb-all-cleaned.csv` (gitignored, ~120 MB).
The dashboard does not: it reads only `processed_data/analytics/*.csv`,
`processed_data/price_model_v2.joblib` and `raw_data/`.

---

# v1 documentation (still accurate for the supporting pages)

## Singapore HDB Access & Affordability Analyser

A Streamlit dashboard analysing HDB resale prices and town-level
access-and-affordability across Singapore, built on public Data.gov.sg,
LTA and public-healthcare datasets.

*Previously titled "Singapore City Equity Analyser". Renamed because the
composite it computes is an amenity-and-affordability index, not an equity or
deprivation index, and the old title claimed something the analysis does not
support. The `equity_*` module and package names are retained for import
stability; every user-facing label now matches what is actually measured.*

---

## Read this first: what the composite score is and is not

The composite is a weighted mean of five **amenity and affordability**
factors. High score = cheaper relative to income, newer stock, more rail
stations per capita, closer to a hospital, shorter distance to a job centre.

**That is not the same thing as an equity or deprivation index.** Mature
central estates (Bishan, Ang Mo Kio, Serangoon, Toa Payoh) score low here
mainly because they are *expensive with older flats*, not because their
residents are underserved. Read a low score as "expensive, older stock".

The equal weighting is a **choice, not a finding**. The dashboard ships a
sensitivity analysis (`compute_weight_sensitivity`) showing the ranking is
genuinely weighting-dependent — Spearman 0.70–0.94 against the default, with
rank shifts up to 15 places. Cite the composite with that caveat attached.

---

## Quick start

```bash
pip install -r dashboard/requirements.txt
python pipeline/train_transaction_model.py      # trains + benchmarks (slow)
streamlit run dashboard/app.py
```

---

## Data provenance

| Factor | Source | Status |
|---|---|---|
| Resale prices | Data.gov.sg HDB resale, 1990–2026, 982,011 txns | Real |
| Income | SingStat monthly household income per member, 2000–2025 | Real |
| Rail stations | LTA MRT Station Exit GEOJSON (Aug 2025), 187 stations incl. 41 LRT | Real |
| Hospitals | 28 operational, real coordinates | Real |
| Polyclinics | SingHealth (10) + NUHS (8) + NHG (10) = 28 | Real |
| Estate age | `estate-maturity.csv` | Real |
| Commute | Straight-line distance to 4 job centres | **Proxy** |
| Population | HDB Annual Report 2023/24 (31 Mar 2024) | Real (1 derived) |
| Education, green space | — | **Removed, not fabricated** |

No factor is randomly generated. An earlier version of this project generated
every equity factor with `np.random.default_rng(42)`; that has been fully
removed and replaced with the sources above.

---

## Known limitations (please read before citing outputs)

1. **Commute is a distance proxy, not journey time.** `Jurong East`'s town
   centroid is byte-identical to the Jurong East Regional Centre job-centre
   coordinate, so its distance is exactly 0 and its commute collapses to the
   fixed 12-minute overhead. This is circular; the feature table flags it via
   `commute_is_degenerate`. Replacing the proxy with real LTA journey-planner
   times would fix it properly.
2. **(v1 composite only) Affordability depends on an assumed household size.**
   v2's Affordability Index page uses SingStat's household-level series instead. The income series
   is *monthly, per household member*. Converting to an annual per-household
   figure uses `AVG_HOUSEHOLD_SIZE = 3.09`. The ranking is unaffected (it
   scales all towns identically before normalisation); the displayed ratio and
   the 3x/5x threshold lines are not.
3. **Gi\* significance bands are indicative.** n = 26 towns, normal
   approximation, no permutation inference.
4. **NHG polyclinics have no real coordinates.** They are assigned to towns by
   name (e.g. "Hougang Polyclinic" → Hougang) and carry town-centroid
   placeholder coordinates. They are counted, never distance-measured.
5. **5 rail stations are missing** (Keppel, Cantonment, Prince Edward Road,
   Punggol Coast, Hume) — too new for the Aug 2025 LTA snapshot.
6. **Estate age for Bukit Timah is unavailable** and is median-imputed;
   `estate_age_imputed` flags it. Kallang/Whampoa and Central Area are mapped
   to `Kallang` and `Pearl's Hill (Chinatown)` respectively in the source file.
7. **Raw filenames.** The original Data.gov.sg downloads were not named by
   the period they contain; the files in `raw_data/` now are (checked 3.2:
   each file's first and last month match its name). The pipeline reads by
   content, not filename, either way.
8. **Duplicated `year`/`quarter` columns.** Older builds of
   `hdb-all-cleaned.csv` carried them; the current pipeline does not (checked
   by a full rebuild in 3.2), and the loader still drops `.1` duplicates if
   an old file is present.

---

## Model validation

> **v1 only.** The served model is now v2; its rolling back-test is on the
> Model Validation page and in CHANGES.md. This section documents the v1
> forest, which the Intel benchmark still uses.

`pipeline/train_transaction_model.py` reports, in
`processed_data/model_metrics.json`:

- **Random 80/20 split** — optimistic; prices are strongly time-trended so a
  random split leaks future information into training.
- **Chronological split** — the honest test: train on early years, predict
  later ones.
- **5-fold CV** on a 120k subsample.
- **MAE and RMSE in dollars**, alongside R².

An in-sample R² is also reported but **clearly labelled as in-sample**. A
depth-14 forest on 982k rows fitting itself to R² ≈ 0.94 is not evidence of
predictive quality — the earlier version of this project reported exactly that
number with no split at all.

---

## Intel technology

Uses **Intel(R) Extension for Scikit-learn** (`sklearnex`), which patches
`RandomForestRegressor` / `KMeans` / `MinMaxScaler` onto Intel's oneDAL
backend with no API change, and evaluates **ONNX Runtime** and **OpenVINO**
for batch inference.

Everything below is measured, on one machine (1 vCPU, Intel Xeon @ 2.80GHz),
both arms back to back, 5 repeats each. Nothing here is projected.

### Training: two workloads, opposite results

| Workload | Stock scikit-learn | Intel oneDAL | Result |
|---|---|---|---|
| RandomForest fit, 982k x 38, 12 trees | 51.74s ± 0.69 | 56.34s ± 1.94 | **1.09x slower** |
| KMeans fit, 250k x 38, k=4, n_init=10 | 2.07s ± 0.10 | 1.33s ± 0.06 | **1.56x faster** |

Both gaps exceed combined run-to-run variance, so neither is noise.

The forest result is negative and is reported as measured. The mechanism is
`cpu_count: 1`. oneDAL's forest gains come from multi-threaded tree building
and vectorised split search; on a single vCPU there is no parallelism to
recover, so the dispatch overhead is paid for no benefit. KMeans is a
different shape of problem -- dense distance computation that vectorises
*within* a core -- and there the extension wins clearly on the same hardware.

That contrast is the actual finding. A single number would have been an
anecdote in either direction; two workloads pointing opposite ways, with a
mechanism that predicts which goes which way, is a result. The open question
this leaves is what the forest arm does on a multi-core box, where the
prediction is that the sign flips. That has not been measured here and is not
claimed.

The KMeans arm also records **inertia**, because a clustering speedup that
lands on a worse local optimum is not a speedup. oneDAL's solution is 0.13%
higher inertia -- marginally worse, not materially so.

The 250k rows are a **seeded random sample** of the real transaction data, not
a head slice. This matters: rows 0-250k of the CSV span 2000-2016 (median
2003) against 1990-2026 (median 2006) for the full set, because the file is
concatenated per era. Benchmarked on that head slice the speedup reads 1.81x;
on a representative sample it is 1.56x. The lower number is the honest one.

**Scope of the KMeans result.** This is *not* the dashboard's town
clustering. That call runs on ~23 towns x 6 factors, finishes in
microseconds, and no backend would change it measurably. Nothing measured
here makes the live dashboard faster. It is the same algorithm on real rows
from the same dataset, scaled to a size where the backend is measurable -- a
benchmark of oneDAL's KMeans path, not of this application. State that in the
pitch, not only here.

### Inference: OpenVINO cannot run this model

`skl2onnx` lowers the forest to a single `ai.onnx.ml.TreeEnsembleRegressor`
node. OpenVINO's ONNX frontend implements the neural-network operator set and
has no conversion rule for the classical-ML tree operators, so `read_model()`
fails outright:

```
OpConversionFailure: Model wasn't fully converted.
-- No conversion rule found for operations: ai.onnx.ml.TreeEnsembleRegressor
```

This is not a conversion bug and not a tuning problem -- the same ONNX file
runs correctly under ONNX Runtime and reproduces sklearn to a mean absolute
difference of $0.03. OpenVINO is simply the wrong runtime for a decision-tree
ensemble. The failure is recorded as structured data in
`openvino_benchmark.json` rather than omitted.

The runtime that does work:

| Runtime | Throughput | Speedup | Agreement vs sklearn |
|---|---|---|---|
| stock scikit-learn | 565,477 rows/s | 1.00x | — |
| ONNX Runtime | 678,510 rows/s | **1.20x** | MAE $0.03 |
| OpenVINO | — | — | cannot load model |

Batch, not single-row: a 40-tree forest predicting one row takes microseconds
and the measurement would be dominated by Python call overhead.

### Reproducing

```bash
pip install scikit-learn-intelex onnxruntime skl2onnx onnx openvino

# training benchmark, both arms
python pipeline/train_transaction_model.py --stages full,kmeans --trees 12 --repeats 5

# inference benchmark -- self-check gate first
python pipeline/export_onnx.py --self-check
python pipeline/export_onnx.py
```

Training results land in `processed_data/intel_benchmark.json` (tagged with
`workload`, `cpu_count` and `platform` so mismatched comparisons are visible),
inference results in `processed_data/openvino_benchmark.json`. Both render on
the *Transaction-Level Price Model* page. The dashboard refuses to display a
speedup when the two arms used different configurations, when either arm was
run with `--repeats 1`, or when the gap is inside combined variance.

**Install note:** on Debian-derived Pythons, `pip` may place `sklearnex` in
`/usr/local/lib/python3.12/site-packages` while only `dist-packages` is on
`sys.path`. The install reports success and `import sklearnex` then fails.
If that happens, set
`PYTHONPATH=/usr/local/lib/python3.12/site-packages`. An earlier version of
this README misattributed that failure to a lack of network access.

---

## Layout

```
dashboard/
  app.py            entry point
  views/            page modules (named views/, NOT pages/, so Streamlit's
                    auto-multipage nav doesn't collide with the sidebar radio)
  data/loader.py    all data loading + geocoding
  equity/           factor computation, Gi*, Shapley
pipeline/
  merge.py clean.py aggregate.py       raw -> processed
  train_transaction_model.py           models + Intel benchmark
raw_data/           source datasets
processed_data/     derived outputs (regenerable)
```
