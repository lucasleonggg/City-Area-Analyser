"""Research & Data -- methodology, data, limitations, paper and downloads."""

import io
import os
import zipfile

import pandas as pd
import streamlit as st

from analysis import store

ROOT = store.ROOT
DOCS = [
    ("Research paper (draft)", "docs/paper/EquiCity_research_paper.md"),
    ("Methodology", "docs/methodology.md"),
    ("Data dictionary & sources", "docs/data_dictionary.md"),
    ("Assumptions & limitations register", "docs/limitations.md"),
    ("Version history of findings", "CHANGES.md"),
]

# Aggregated tables safe and useful to publish (no row-level personal data
# exists in this project; transactions are public and anonymous).
DATASETS = {
    "hedonic_town_year_type.csv": "Constant-quality price of a reference flat by town, flat type and year, with location premium",
    "hedonic_national_month.csv": "National constant-quality price index, monthly, 1Q2009 = 100",
    "hedonic_region_year.csv": "Constant-quality 4-room price by URA region and year",
    "lease_curve.csv": "Price by remaining lease relative to 90 years, re-estimated each year",
    "hedonic_fit_summary.csv": "Per-year hedonic fit diagnostics and key coefficients",
    "observed_town_type_year.csv": "Observed medians by town, flat type and year",
    "block_premium.csv": "Block-level location premium (last 3 years) with Gi* hot/cold spots",
    "rail_access_town_year.csv": "Time-correct distance to an open rail station by town and year",
    "rail_event_study.csv": "Descriptive event study around station openings",
    "robust_moran_summary.csv": "Moran's I trend under 24 spatial-weight specifications",
    "robust_moran_ci.csv": "Town-level Moran's I with bootstrap 95% intervals, by year",
    "robust_premium_ci.csv": "Town location premia with bootstrap 95% intervals, by year",
    "audit_backtest_overall.csv": "Price-model back-test results by model",
    "audit_subgroups.csv": "Back-test error and interval coverage by town, flat type, lease and price band",
    "audit_horizons.csv": "Back-test error and coverage by forecast horizon",
    "market_forecast_baselines.csv": "How well simple methods forecast the national index",
}


def _read(rel):
    p = os.path.join(ROOT, rel)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else None


def show():
    st.title("Research & Data")
    st.markdown(
        "Everything needed to check how EquiCity's results were produced: the paper, the methods, the "
        "data and its sources, every assumption, and the aggregated datasets behind each chart. The "
        "full pipeline reruns from the inputs with one command (`bash reproduce.sh`); inputs are "
        "verified against SHA-256 hashes in `raw_data/MANIFEST.csv`.")

    tabs = st.tabs([d[0] for d in DOCS] + ["Download data"])
    for tab, (title, rel) in zip(tabs, DOCS):
        with tab:
            text = _read(rel)
            if text is None:
                st.info(f"`{rel}` not found.")
                continue
            st.download_button(f"Download {title} (Markdown)", text, file_name=os.path.basename(rel),
                               mime="text/markdown", key=f"dl_{rel}")
            st.markdown(text)

    with tabs[-1]:
        st.markdown("Aggregated tables (CSV). Each is produced by a named script in `pipeline/`; "
                    "column definitions are in the Data dictionary tab.")
        rows = []
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for name, desc in DATASETS.items():
                if store.exists(name):
                    z.write(store.path(name), arcname=name)
                    rows.append({"file": name, "description": desc,
                                 "rows": len(store.table(name)), "KB": round(os.path.getsize(store.path(name)) / 1024)})
            for rel in [d[1] for d in DOCS]:
                if os.path.exists(os.path.join(ROOT, rel)):
                    z.write(os.path.join(ROOT, rel), arcname=f"docs/{os.path.basename(rel)}")
        st.download_button("Download all tables + documents (zip)", buf.getvalue(),
                           file_name="equicity_research_data.zip", mime="application/zip")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        pick = st.selectbox("Preview / download one table", [r["file"] for r in rows])
        if pick:
            df = store.table(pick)
            st.download_button(f"Download {pick}", df.to_csv(index=False), file_name=pick, mime="text/csv")
            st.dataframe(df.head(200), width="stretch", hide_index=True)
        nb = os.path.join(ROOT, "notebooks", "reproduce_headline_findings.ipynb")
        if os.path.exists(nb):
            st.download_button("Download the reproducibility notebook (.ipynb)", open(nb, "rb").read(),
                               file_name="reproduce_headline_findings.ipynb")
