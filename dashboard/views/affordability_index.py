"""2 · Affordability Index -- how affordable is housing, where, and for whom?"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import affordability as A
from analysis import store
from views._common import (FLAT_TYPES, method_note, money, national_reference_price,
                           need, town_coords_df)


def _controls(key):
    c1, c2, c3 = st.columns([1, 1, 1.4])
    flat_type = c1.selectbox("Flat type", FLAT_TYPES, index=2, key=f"{key}_ft")
    pct = c2.slider("Household income percentile", 10, 90, 50, 5, key=f"{key}_pct",
                    help="Resident employed households, monthly household employment income "
                         "excluding employer CPF (SingStat).")
    preset = c3.selectbox("Financing", list(A.PRESETS), key=f"{key}_preset")
    fin = A.PRESETS[preset]
    with st.expander("Adjust financing terms"):
        a, b, c, d = st.columns(4)
        rate = a.number_input("Interest rate (%)", 0.0, 10.0, fin.rate * 100, 0.1, key=f"{key}_r") / 100
        years = b.number_input("Tenure (years)", 5, 35, fin.years, 1, key=f"{key}_y")
        ltv = c.number_input("Loan-to-value (%)", 10.0, 95.0, fin.ltv * 100, 5.0, key=f"{key}_l") / 100
        cap = d.number_input("MSR cap (%)", 10.0, 60.0, fin.msr_cap * 100, 1.0, key=f"{key}_m") / 100
    return flat_type, pct, A.Financing(rate=rate, years=int(years), ltv=ltv, msr_cap=cap)


def _town_frame(flat_type, basis, min_cell):
    if basis.startswith("Constant"):
        t = store.table("hedonic_town_year_type.csv")
        t = t[t["flat_type"] == flat_type].rename(columns={"price_ref": "price"})
        t = t[t["n_town"] >= 30]
    else:
        t = store.table("observed_town_type_year.csv")
        t = t[(t["flat_type"] == flat_type) & (t["n"] >= min_cell)].rename(
            columns={"median_price": "price", "n": "n_cell"})
    return t


def show():
    st.title("Affordability Index")
    st.markdown(
        "For a household at a chosen income percentile, buying a flat with a standard loan: "
        "what share of income goes to the mortgage, and does it fit under Singapore's "
        "**30% Mortgage Servicing Ratio (MSR)** cap? An index of **100** means the household "
        "exactly meets the cap; **above 100** it has headroom; **below 100** it cannot borrow "
        "the full loan.")
    if not need("hedonic_town_year_type.csv", "observed_town_type_year.csv"):
        return

    flat_type, pct, fin = _controls("aff")
    c1, c2 = st.columns([1.4, 1])
    basis = c1.radio("Price basis", ["Constant-quality reference flat (recommended)",
                                     "Median of flats that actually sold"], horizontal=False)
    year = c2.slider("Year", 2000, 2026, 2026)

    t = _town_frame(flat_type, basis, min_cell=20)
    af = A.affordability_frame(t, pct, fin)
    cur = af[af["year"] == year].merge(town_coords_df(), on="town", how="left")
    if cur.empty:
        st.info("No towns have enough sales of this flat type in that year.")
        return
    inc_year = int(cur["income_year_used"].iloc[0])
    inc = float(cur["monthly_income"].iloc[0])

    nat = national_reference_price(flat_type)
    nat_row = nat[nat["year"] == year]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"P{pct} household income", money(inc) + "/mo",
              help=f"SingStat {inc_year}" + (" (latest available; carried forward)" if inc_year != year else ""))
    if not nat_row.empty:
        ev = A.evaluate(float(nat_row["price"].iloc[0]), year, pct, fin)
        k2.metric(f"National {flat_type} (reference)", money(nat_row["price"].iloc[0]))
        k3.metric("Mortgage share of income", f"{ev['msr_pct']:.1f}%",
                  delta=f"{ev['msr_pct'] - fin.msr_cap * 100:+.1f} pp vs cap", delta_color="inverse")
        k4.metric("Towns within MSR cap", f"{int(cur['within_cap'].sum())} of {len(cur)}")
    if inc_year != year:
        st.caption(f"Income for {year} is not yet published; the {inc_year} income distribution is used.")

    # ---- map ----
    st.subheader(f"Where is a {flat_type} affordable for a P{pct} household? ({year})")
    fig = px.scatter_map(
        cur, lat="lat", lon="lng", color="affordability_index", size="price",
        hover_name="town",
        hover_data={"price": ":$,.0f", "msr_pct": ":.1f", "affordability_index": ":.0f",
                    "required_percentile": ":.0f", "lat": False, "lng": False},
        color_continuous_scale="RdYlGn", color_continuous_midpoint=100,
        map_style="open-street-map", center={"lat": 1.355, "lon": 103.82}, zoom=10.2,
        height=520, size_max=24, labels={"affordability_index": "Index (100 = at cap)"})
    st.plotly_chart(fig, width="stretch")

    # ---- ranking ----
    c1, c2 = st.columns(2)
    rank = cur.sort_values("msr_pct")
    fig = px.bar(rank, x="msr_pct", y="town", orientation="h", color="region",
                 labels={"msr_pct": "Mortgage as % of income", "town": ""},
                 title="Mortgage share of income, by town", height=640)
    fig.update_yaxes(categoryorder="array", categoryarray=rank["town"].tolist())
    fig.add_vline(x=fin.msr_cap * 100, line_dash="dash", line_color="black",
                  annotation_text=f"{fin.msr_cap * 100:.0f}% MSR cap")
    c1.plotly_chart(fig, width="stretch")
    rq = cur.sort_values("required_percentile")
    rq = rq.assign(label=np.where(rq["required_percentile_flag"] == ">", ">90",
                         np.where(rq["required_percentile_flag"] == "<", "<10",
                                  rq["required_percentile"].round(0).astype(int).astype(str))))
    fig = px.bar(rq, x="required_percentile", y="town", orientation="h", color="region", text="label",
                 labels={"required_percentile": "Income percentile needed to meet the cap", "town": ""},
                 title="Minimum income percentile that can finance it", height=640)
    fig.update_yaxes(categoryorder="array", categoryarray=rq["town"].tolist())
    fig.add_vline(x=pct, line_dash="dot", annotation_text=f"selected P{pct}")
    c2.plotly_chart(fig, width="stretch")
    st.caption("A required percentile of 40 means the household income at the 40th percentile "
               "is just enough, so about 60% of resident employed households could finance it. "
               "'>90' means only households above the 90th percentile could.")

    # ---- over time ----
    st.subheader("How has affordability changed?")
    rows = []
    for p in [20, 50, 80]:
        a = A.affordability_frame(nat.assign(year=nat["year"]), p, fin)
        rows.append(a.assign(percentile=f"P{p}"))
    ts = pd.concat(rows)
    fig = px.line(ts, x="year", y="msr_pct", color="percentile", markers=True,
                  labels={"msr_pct": "Mortgage as % of income", "year": ""},
                  title=f"National constant-quality {flat_type}: mortgage share of income by household percentile")
    fig.add_hline(y=fin.msr_cap * 100, line_dash="dash", annotation_text="MSR cap")
    st.plotly_chart(fig, width="stretch")

    towns_sel = st.multiselect("Compare towns over time", sorted(af["town"].unique()),
                               default=["Punggol", "Queenstown", "Woodlands", "Bishan"])
    if towns_sel:
        fig = px.line(af[af["town"].isin(towns_sel)], x="year", y="affordability_index", color="town",
                      labels={"affordability_index": f"Affordability index, P{pct}", "year": ""})
        fig.add_hline(y=100, line_dash="dash")
        st.plotly_chart(fig, width="stretch")

    within = af.groupby("year").agg(towns=("town", "size"), within=("within_cap", "sum")).reset_index()
    within["share_pct"] = within["within"] / within["towns"] * 100
    fig = px.bar(within, x="year", y="share_pct",
                 labels={"share_pct": "% of towns within the cap", "year": ""},
                 title=f"Share of towns where a P{pct} household can finance a {flat_type}")
    st.plotly_chart(fig, width="stretch")

    # ---- local income lens ----
    st.subheader("A second lens: local residents' incomes (Census 2020)")
    st.markdown(
        "Everything above uses the **national** income distribution, which is the right lens for "
        "a buyer who can buy anywhere. It also means differences between towns come only from "
        "prices. The Census 2020 gives household income by planning area, so we can ask a "
        "different question: *could the people who already live here afford a flat here?*")
    ci = A.census_local_income()
    rows = []
    t20 = _town_frame(flat_type, basis, 20)
    t20 = t20[t20["year"] == 2020]
    nat_med_2020, _ = A.income_at(2020, 50)
    for _, r in t20.iterrows():
        ratio = A.town_local_income_ratio(r["town"])
        if ratio is None:
            continue
        local = nat_med_2020 * ratio
        pay = float(A.annuity_payment(r["price"] * fin.ltv, fin.rate, fin.years))
        rows.append({"town": r["town"], "local_income_vs_national": ratio, "local_median_income": local,
                     "msr_local_pct": pay / local * 100,
                     "msr_national_pct": pay / nat_med_2020 * 100})
    if rows:
        li = pd.DataFrame(rows).sort_values("msr_local_pct")
        fig = go.Figure()
        fig.add_bar(y=li["town"], x=li["msr_national_pct"], orientation="h", name="National median household")
        fig.add_bar(y=li["town"], x=li["msr_local_pct"], orientation="h", name="Local median household")
        fig.add_vline(x=fin.msr_cap * 100, line_dash="dash")
        fig.update_layout(barmode="group", height=700, xaxis_title="Mortgage as % of income (2020)",
                          title=f"{flat_type} in 2020: national vs local income")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Census planning-area incomes include condominium and landed households and employer "
            "CPF, so areas such as Bukit Timah, Bishan and Marine Parade look richer than their HDB "
            "residents are. The local figure is scaled from the area's ratio to the national Census "
            "median. Treat this as a relative lens, not a measure of HDB buyers' incomes.")

    method_note(A.__doc__.split("INCOME SOURCE")[0] + "\n\n**Income source.** SingStat household "
                "employment income percentiles, 2000-2025 (data.gov.sg). v1 used per-member income x 3.09; "
                "the household series replaces that assumption.\n\n**Held fixed.** Financing terms are "
                "held at the selected values for every year so the chart isolates prices vs incomes. "
                "Not modelled: CPF grants, existing savings, sale proceeds, income ceilings.")
