"""EquiCity 3.0 — Singapore HDB Resale Affordability & Spatial Inequality.

Entry point with page navigation.
"""

import streamlit as st

from views import (affordability_index, equity_map, explainability, map_view, market_trends,
                   model_validation, overview, policy_lab, predictor, research, scenario_engine,
                   spatial_inequality, transaction_model, transport_access)

st.set_page_config(
    page_title="EquiCity 3.0 — HDB Affordability & Spatial Inequality",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("EquiCity 3.0")
st.sidebar.markdown(
    "Spatial inequality in Singapore HDB resale affordability, built on 982k real "
    "transactions (Data.gov.sg, 1990–2026) and SingStat household incomes.")

PAGES = {
    "Overview": overview,
    "1 · Price Predictor": predictor,
    "2 · Affordability Index": affordability_index,
    "3 · Scenario Engine": scenario_engine,
    "4 · Policy Lab": policy_lab,
    "5 · Spatial Inequality": spatial_inequality,
    "6 · Transport Access": transport_access,
    "7 · Market Trends": market_trends,
    "8 · Explainability": explainability,
    "Model Validation": model_validation,
    "Research & Data": research,
    "Supporting · Access & Amenity Index": equity_map,
    "Supporting · Amenity Map": map_view,
    "Supporting · Intel Benchmark": transaction_model,
}

# ?page=<slug> deep-links to a module, e.g. ?page=affordability
SLUGS = {"overview": 0, "predictor": 1, "affordability": 2, "scenario": 3, "policy": 4, "spatial": 5,
         "transport": 6, "trends": 7, "explain": 8, "validation": 9, "research": 10, "access": 11,
         "amenity-map": 12, "intel": 13}
start = SLUGS.get(str(st.query_params.get("page", "overview")).lower(), 0)
page = st.sidebar.radio("Navigate", list(PAGES), index=start)
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit + Plotly | [Data.gov.sg](https://data.gov.sg) | "
                   "[OneMap](https://www.onemap.gov.sg)")
PAGES[page].show()
