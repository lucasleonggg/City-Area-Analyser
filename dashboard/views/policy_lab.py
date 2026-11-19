"""EquiCity Policy Lab -- mechanical what-if financing experiments."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import policy as P
from analysis import store
from analysis.scenario import Loan
from views._common import FLAT_TYPES, method_note, money, national_reference_price, need

def _scenario_controls(key, base: Loan, base_grant):
    c1, c2, c3, c4 = st.columns(4)
    rate = c1.slider("Interest rate (%)", 0.5, 8.0, 3.5, 0.1, key=f"{key}r",
                     help="Starts at 3.5% so the comparison shows something; set 2.6 to match the baseline.") / 100
    ltv = c2.slider("Loan-to-value (%)", 50, 95, int(base.ltv * 100), 5, key=f"{key}l") / 100
    grant = c3.slider("Grant ($)", 0, 230_000, base_grant, 5_000, key=f"{key}g")
    msr = c4.slider("MSR cap (%)", 20, 45, int(base.msr * 100), 1, key=f"{key}m") / 100
    # eligibility assessed at the higher of the charged rate and HDB's 3% floor
    return Loan(rate=rate, assess_rate=max(rate, 0.03), years=25, ltv=ltv, msr=msr), grant


def show():
    st.title("Policy Lab")
    st.markdown(
        "Change the financing rules and see the **mechanical** effect on which households could "
        "finance a flat, holding observed prices and incomes fixed.")
    st.warning(
        "**First-round arithmetic only.** Prices, incomes and behaviour do not respond here. In reality "
        "a cheaper loan or a bigger grant also pushes up what sellers ask, so these results overstate "
        "the lasting gain for buyers. EquiCity has no validated model of that response and does not "
        "pretend to. A BTO-supply experiment is deliberately not offered for the same reason.")
    if not need("hedonic_town_year_type.csv"):
        return

    c1, c2 = st.columns(2)
    ft = c1.selectbox("Flat type", FLAT_TYPES, index=2)
    sav = c2.radio("Savings", ["Assume savings do not bind (income and LTV tests only)",
                               "Liquid assets = a multiple of annual income (assumption)"])
    mult = None
    if sav.startswith("Liquid"):
        mult = st.slider("Cash + CPF OA, in years of household income", 0.0, 3.0, 0.75, 0.25)
    else:
        st.info("In this mode the downpayment is assumed to be covered, so only the loan the household must "
                "service matters. Two consequences follow from the arithmetic: **grants have no effect** unless "
                "they exceed the downpayment (they replace savings, not borrowing), and **a higher LTV can only "
                "hurt** (a bigger loan to service). Grants and LTV work through savings: switch to the "
                "liquid-assets assumption to see them.")

    base = Loan()
    st.subheader("Baseline")
    st.caption("Today's HDB loan: 2.6% charged, assessed at HDB's 3% eligibility floor, 25 years, 75% LTV, "
               "30% MSR, no grant. (The Affordability Index page uses the 2.6% charged rate, so its shares are "
               "slightly higher.)")
    st.subheader("Scenario")
    scen, grant = _scenario_controls("s", base, 0)

    t = store.table("hedonic_town_year_type.csv")
    yr = int(t["year"].max())
    inc_year = min(yr, 2025)
    towns = t[(t["flat_type"] == ft) & (t["year"] == yr) & (t["n_town"] >= 30)].rename(columns={"price_ref": "price"})

    lb = P.percentile_ladder(inc_year, base, 0, mult)
    ls = P.percentile_ladder(inc_year, scen, grant, mult)
    rb, rsn = P.towns_within_reach(towns, lb), P.towns_within_reach(towns, ls)
    comp = rb.merge(rsn, on=["percentile", "of_towns"], suffixes=("_baseline", "_scenario"))
    comp = comp.merge(lb[["percentile", "max_price"]], on="percentile").merge(
        ls[["percentile", "max_price"]], on="percentile", suffixes=("_baseline", "_scenario"))

    st.header(f"Who can finance a {ft}, and where? ({yr} prices, {inc_year} incomes)")
    fig = go.Figure()
    fig.add_bar(x=[f"P{int(p)}" for p in comp["percentile"]], y=comp["towns_within_reach_baseline"], name="Baseline")
    fig.add_bar(x=[f"P{int(p)}" for p in comp["percentile"]], y=comp["towns_within_reach_scenario"], name="Scenario")
    fig.update_layout(barmode="group", yaxis_title=f"Towns (of {len(towns)}) where the reference flat is affordable",
                      xaxis_title="Household income percentile", height=420)
    st.plotly_chart(fig, width="stretch")
    show_tbl = comp.assign(**{
        "max_price_baseline": comp["max_price_baseline"].map(money),
        "max_price_scenario": comp["max_price_scenario"].map(money)})
    st.dataframe(show_tbl, width="stretch", hide_index=True)

    st.header("The affordability frontier over time")
    nat = national_reference_price(ft)
    fb = P.frontier_over_time(nat, base, 0, mult).assign(case="Baseline")
    fs = P.frontier_over_time(nat, scen, grant, mult).assign(case="Scenario")
    both = pd.concat([fb, fs])
    fig = px.line(both, x="year", y="share_able_pct", color="case", markers=True,
                  labels={"share_able_pct": "% of resident employed households able to finance it", "year": ""},
                  title=f"National constant-quality {ft}: share of households able to finance it")
    st.plotly_chart(fig, width="stretch")
    st.caption("Shares are bounded by the published income percentiles (10th to 90th): 90% means 'at least "
               "90%', 0% means 'only households above the 90th percentile'. Years after 2025 use 2025 incomes.")
    last = both[both["year"] == both["year"].max()].set_index("case")["share_able_pct"]
    if len(last) == 2:
        st.metric(f"Change in households able to finance it, {int(both['year'].max())}",
                  f"{last['Scenario'] - last['Baseline']:+.0f} percentage points")

    method_note(P.__doc__)
