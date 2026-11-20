# Deploying to Streamlit Community Cloud

Free, public URL, no card required. ~30 minutes end to end.

## Before you start

Three blockers were fixed in this repo. Do not undo them:

1. **`processed_data/*.joblib` is no longer gitignored.** The dashboard loads
   `transaction_price_model.joblib` at runtime. Ignoring it deploys a broken
   page. Both models are under GitHub's 100MB per-file limit.
2. **Every runtime dependency is pinned exactly** (`==`) in
   `dashboard/requirements.txt`, including `scikit-learn==1.9.1`, the version
   that trained the served model. The model file records its training
   versions and the Model Validation page warns if the deployment differs. An unpinned `>=` resolves to whatever is newest and unpickling across
   minor versions is not guaranteed to reconstruct the estimator.
3. **Benchmark tooling moved to `requirements-dev.txt`.** `openvino`,
   `onnxruntime`, `skl2onnx` and `scikit-learn-intelex` total several hundred MB
   and the dashboard never imports them. Leaving them in the runtime file makes
   every build slow and may exhaust the free tier's build resources.

`processed_data/hdb-all-cleaned.csv` (127MB) stays ignored — it exceeds
GitHub's 100MB hard limit and nothing needs it at runtime.

## Steps

1. **Create the GitHub repo.** Public (private works but public is one less
   permission step). Do not initialise it with a README.

2. **Push.**
   ```bash
   cd HDB_Access_Affordability_Analyser
   git init
   git add -A
   git status --short          # confirm hdb-all-cleaned.csv is NOT listed
   git commit -m "Singapore HDB Access & Affordability Analyser"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
   Expect ~110MB. If the push is rejected for file size, something re-added the
   127MB CSV — check `git status` before committing, not after.

3. **Deploy.** Go to https://share.streamlit.io, sign in with GitHub,
   "Create app" → "Deploy a public app from GitHub".
   - Repository: `<you>/<repo>`
   - Branch: `main`
   - **Main file path: `dashboard/app.py`** ← this exact path
   - Advanced settings → Python version **3.12** (tested in CI, along with 3.13)

   Streamlit puts the main file's directory on `sys.path`, which is why
   `dashboard/app.py` resolves `import views.overview` correctly. Verified.

4. **Watch the build log.** It installs `dashboard/requirements.txt`. First
   build takes 3-6 minutes. If it stalls past ~10, check that
   `requirements-dev.txt` was not picked up.

5. **Click every page.** Overview, the eight numbered modules, Model
   Validation, Research & Data and the three Supporting pages. All fourteen were verified to render
   (headless Streamlit AppTest + browser screenshots) from a checkout without
   the gitignored ~120 MB CSV. Maps use `px.scatter_map`, so `plotly>=5.24`
   is required; `scatter_mapbox` no longer exists in plotly 7.

6. **Put the URL in the submission form and on the video end card.**

## Known non-blocking warning

The live prediction widget logs:

```
UserWarning: X has feature names, but RandomForestRegressor was fitted without feature names
```

The model was fitted on a NumPy array; predict is called with a DataFrame.
Predictions are correct because the frame is reindexed onto
`bundle["feature_columns"]` first. It is noise in the log, not a wrong number.
Silence it by passing `.to_numpy()` at the call site if you want a clean log.

## If the build fails

- **`ModuleNotFoundError: views`** → main file path is wrong. Must be
  `dashboard/app.py`, not `app.py`.
- **`FileNotFoundError` on a `raw_data/*.csv`** → that file got gitignored.
  Check with `git ls-files raw_data | wc -l`.
- **`InconsistentVersionWarning` then bad predictions** → the sklearn pin no
  longer matches the training environment. Retrain or fix the pin.
- **`AttributeError: 'Styler' object has no attribute 'applymap'`** → you
  reverted the pandas 3.0 fix in `equity_map.py`.
