"""Affordability Scenario Engine -- what could THIS household afford, and where?"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import scenario as SC
from analysis import store
from views._common import FLAT_TYPES, method_note, money, need, town_coords_df


@st.cache_data
def _recent():
    return store.table("recent_sales.csv")


def _inputs():
    st.subheader("Household")
    st.caption("Starts with an illustrative first-timer couple: $7,500 a month, $40k cash, $60k CPF, and the "
               "$80k CPF Housing Grant for a 2- to 4-room resale flat. Change any figure.")
    c1, c2, c3 = st.columns(3)
    income = c1.number_input("Gross monthly household income ($)", 0, 50_000, 7_500, 250)
    debt = c2.number_input("Other monthly loan repayments ($)", 0, 20_000, 0, 50,
                           help="Car loan, personal loans, credit-card minimums. Counted under TDSR for bank loans.")
    cash = c3.number_input("Cash savings available ($)", 0, 2_000_000, 40_000, 5_000)
    c1, c2, c3 = st.columns(3)
    cpf = c1.number_input("CPF Ordinary Account available ($)", 0, 2_000_000, 60_000, 5_000)
    grants = c2.number_input("CPF housing grants assumed ($)", 0, 230_000, 80_000, 5_000,
                             help="Enter what you expect to qualify for. Indicatively, first-timer families buying "
                                  "resale can stack a CPF Housing Grant (up to $80k for 2-4 room, $50k for 5-room+), "
                                  "an income-tiered Enhanced CPF Housing Grant (up to $120k) and a Proximity Housing "
                                  "Grant (up to $30k). Check HDB's HFE letter for your actual amounts.")
    other = c3.number_input("Other upfront cash costs ($)", 0, 100_000, 5_000, 500,
                            help="Legal fees, option fee, valuation fee and similar.")
    st.subheader("Loan")
    preset = st.selectbox("Loan type", list(SC.LOAN_PRESETS))
    L = SC.LOAN_PRESETS[preset]
    with st.expander("Adjust loan terms"):
        a, b, c, d = st.columns(4)
        rate = a.number_input("Interest rate charged (%)", 0.0, 10.0, L.rate * 100, 0.1) / 100
        assess = b.number_input("Assessment rate (%)", 0.0, 10.0, L.assess_rate * 100, 0.1) / 100
        years = c.number_input("Tenure (years)", 5, 35, L.years, 1)
        ltv = d.number_input("Loan-to-value (%)", 10.0, 95.0, L.ltv * 100, 5.0) / 100
        a, b, _, _ = st.columns(4)
        msr = a.number_input("MSR cap (%)", 10.0, 60.0, L.msr * 100, 1.0) / 100
        tdsr = b.number_input("TDSR cap (%) (bank loans)", 10.0, 80.0, L.tdsr * 100, 1.0) / 100
    loan = SC.Loan(kind=L.kind, rate=rate, assess_rate=assess, years=int(years), ltv=ltv, msr=msr,
                   tdsr=tdsr, min_cash_pct=L.min_cash_pct)
    h = SC.Household(income=income, other_debt=debt, cash=cash, cpf_oa=cpf, grants=grants, other_upfront=other)
    return h, loan


def show():
    st.title("Affordability Scenario Engine")
    st.markdown(
        "Enter a household's finances and loan terms to see the most it could pay for a resale flat, "
        "what that would cost upfront and every month, and **which towns' flats are within reach**.")
    st.warning(
        "**A scenario calculator, not an eligibility check.** Grants, CPF and savings are your own "
        "assumptions. HDB's HDB Flat Eligibility (HFE) letter decides real loan and grant eligibility, "
        "using rules this tool does not model (citizenship, income ceilings, first-timer status, age, "
        "remaining lease, credit history).")
    if not need("hedonic_town_year_type.csv", "recent_sales.csv"):
        return
    h, loan = _inputs()
    best = SC.max_price(h, loan)

    st.divider()
    st.subheader("Result")
    a, b, c, d = st.columns(4)
    a.metric("Maximum price", money(round(best["price"], -3)))
    b.metric("Loan", money(round(best["loan"], -3)))
    c.metric("Monthly instalment", money(best["monthly_payment"]))
    d.metric("Mortgage ÷ income", f"{best['mortgage_to_income_pct']:.1f}%")
    st.caption(f"What stops a higher price: {best['binding']}.")

    st.subheader("Check a specific price")
    price = st.number_input("Flat price ($)", 50_000, 3_000_000, int(max(best["price"], 60_000) // 10_000 * 10_000), 10_000)
    e = SC.evaluate(price, h, loan)
    rows = [("Price", e["price"]), ("less grants applied", -e["grants_used"]), ("less loan", -e["loan"]),
            ("Downpayment", e["downpayment"]), ("plus buyer's stamp duty", e["stamp_duty"]),
            ("plus other upfront costs", h.other_upfront), ("less paid from CPF OA", -e["cpf_used"]),
            ("Cash needed", e["cash_needed"]), ("Cash available", h.cash)]
    st.table(pd.DataFrame(rows, columns=["", "$"]).assign(**{"$": lambda d: d["$"].map(lambda v: f"{v:,.0f}")})
             .set_index(""))
    if e["feasible"]:
        st.success(f"Feasible. Monthly instalment {money(e['monthly_payment'])} "
                   f"({e['mortgage_to_income_pct']:.1f}% of income); loan limited by {e['loan_limited_by']}.")
    else:
        st.error(f"Short by {money(e['cash_shortfall'])} of cash. The loan is limited by {e['loan_limited_by']}, "
                 "so the rest of the price falls on the downpayment.")

    st.subheader("Where is that within reach?")
    ft = st.selectbox("Flat type", FLAT_TYPES, index=2)
    t = store.table("hedonic_town_year_type.csv")
    yr = int(t["year"].max())
    t = t[(t["flat_type"] == ft) & (t["year"] == yr) & (t["n_town"] >= 30)].rename(columns={"price_ref": "price"})
    rs = _recent()
    rs = rs[(rs["flat_type"] == ft) & (rs["month"] >= rs["month"].max()[:4] + "-01")]
    share = rs.groupby("town")["resale_price"].apply(lambda p: (p <= best["price"]).mean() * 100)
    t["share_of_sales_affordable_pct"] = t["town"].map(share)
    t["within_reach"] = t["price"] <= best["price"]
    t = t.merge(town_coords_df(), on="town")
    n_reach = int(t["within_reach"].sum())
    cheapest = t.sort_values("price").iloc[0]
    if (t["share_of_sales_affordable_pct"].fillna(0) < 1).all():
        st.error(f"Almost none of this year's {ft} resale flats are within reach in any town. The cheapest "
                 f"reference {ft} ({cheapest['town']}, {money(cheapest['price'])}) is "
                 f"{money(cheapest['price'] - best['price'])} above this household's maximum. The limit is "
                 f"{best['binding']}.")
    else:
        st.markdown(f"The reference {ft} is within reach in **{n_reach} of {len(t)} towns**; the map shows the "
                    f"share of each town's actual {yr} {ft} sales at or below {money(round(best['price'], -3))}.")
    fig = px.scatter_map(t, lat="lat", lon="lng", color="share_of_sales_affordable_pct", size="price",
                         hover_name="town", range_color=(0, 100), color_continuous_scale="RdYlGn",
                         hover_data={"price": ":$,.0f", "share_of_sales_affordable_pct": ":.0f",
                                     "lat": False, "lng": False},
                         map_style="open-street-map", center={"lat": 1.355, "lon": 103.82}, zoom=10.2, height=500,
                         labels={"share_of_sales_affordable_pct": f"% of {yr} {ft} sales at or below max price"})
    st.plotly_chart(fig, width="stretch")
    s = t.sort_values("share_of_sales_affordable_pct")
    fig = px.bar(s, x="share_of_sales_affordable_pct", y="town", orientation="h", color="region", height=640,
                 labels={"share_of_sales_affordable_pct": f"% of {yr} {ft} resale transactions within reach", "town": ""})
    fig.update_yaxes(categoryorder="array", categoryarray=s["town"].tolist())
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Bars use actual {yr} transactions of this flat type in each town, so they count real flats "
               f"you could have bought. The map's bubble size is the constant-quality reference price.")

    method_note(SC.__doc__)
