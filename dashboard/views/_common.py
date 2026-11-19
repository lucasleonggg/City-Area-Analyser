"""Small helpers shared by the EquiCity 2.0 pages."""

import numpy as np
import pandas as pd
import streamlit as st

from analysis import store
from data.loader import TOWN_COORDS

FLAT_TYPES = ["2-Room", "3-Room", "4-Room", "5-Room", "Executive"]
REGION_COLORS = {"Central": "#5B4B8A", "East": "#2A9D8F", "North": "#E9C46A",
                 "North-East": "#F4A261", "West": "#E76F51"}


def money(x, digits=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"${x:,.{digits}f}"


def need(*tables) -> bool:
    """Show a build instruction instead of crashing when an analytics table
    has not been generated yet."""
    missing = [t for t in tables if not store.exists(t)]
    if missing:
        st.warning(
            "Missing analytics tables: " + ", ".join(f"`{m}`" for m in missing)
            + ". Run `python pipeline/build_analytics.py` (and, for model tables, "
            "`python pipeline/audit_price_model.py` then "
            "`python pipeline/build_explanations.py`).")
        return False
    return True


def town_coords_df() -> pd.DataFrame:
    return (pd.DataFrame.from_dict(TOWN_COORDS, orient="index", columns=["lat", "lng"])
              .rename_axis("town").reset_index())


@st.cache_data
def fixed_town_weights() -> dict:
    """Same fixed weights the pipeline uses: 2015-2025 transaction shares."""
    t = store.table("hedonic_town_year_type.csv")
    t = t[(t["flat_type"] == "4-Room") & t["year"].between(2015, 2025)]
    w = t.groupby("town")["n_town"].sum()
    return (w / w.sum()).to_dict()


@st.cache_data
def national_reference_price(flat_type: str) -> pd.DataFrame:
    """Fixed-weight national constant-quality price by year for a flat type."""
    t = store.table("hedonic_town_year_type.csv")
    t = t[t["flat_type"] == flat_type]
    w = fixed_town_weights()
    t = t.assign(w=t["town"].map(w).fillna(0.0))
    g = t.groupby("year").apply(
        lambda d: np.exp((d["log_price_ref"] * d["w"]).sum() / d["w"].sum()) if d["w"].sum() > 0
        else np.exp(d["log_price_ref"].mean()), include_groups=False)
    return g.rename("price").reset_index()


def method_note(text: str, title: str = "How this is calculated"):
    with st.expander(title):
        st.markdown(text)
