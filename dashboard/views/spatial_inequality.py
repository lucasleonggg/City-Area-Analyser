"""3 · Spatial Inequality -- where are disparities concentrated, and are they
growing?"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import affordability as A
from analysis import spatial as S
from analysis import store
from analysis.temporal import growth_table
from views._common import method_note, need, town_coords_df

VARIABLES = {
    "Location premium (constant quality)": "location_premium_pct",
    "Affordability index (P50 household, 4-room)": "affordability_index",
    "5-year constant-quality price growth": "growth_5y_pct",
}


@st.cache_data
def town_panel() -> pd.DataFrame:
    """Town x year panel of the spatial variables, 4-room reference flat."""
    t = store.table("hedonic_town_year_type.csv")
    t = t[(t["flat_type"] == "4-Room") & (t["n_town"] >= 30)].copy()
    t = t.rename(columns={"price_ref": "price"})
    af = A.affordability_frame(t, 50)
    t = t.merge(af[["year", "town", "affordability_index", "msr_pct"]], on=["year", "town"], how="left")
    lp = t.pivot(index="year", columns="town", values="log_price_ref")
    g5 = (np.exp(lp - lp.shift(5)) - 1) * 100
    g5 = g5.stack().rename("growth_5y_pct").reset_index()
    t = t.merge(g5, on=["year", "town"], how="left")
    return t.merge(town_coords_df(), on="town", how="left")


@st.cache_data
def moran_series(col: str, k: int = 5) -> pd.DataFrame:
    p = town_panel()
    rows = []
    for y, g in p.groupby("year"):
        g = g.dropna(subset=[col, "lat"])
        if len(g) < 15:
            continue
        W = S.knn_weights(g[["lat", "lng"]].to_numpy(), k=k)
        m = S.morans_i(g[col].to_numpy(), W, n_perm=999)
        x = g["price"].to_numpy()
        ineq = S.inequality(x, g["region"].to_numpy())
        rows.append({"year": y, "n_towns": len(g), "morans_I": m["I"], "p_sim": m["p_sim"],
                     "cv_price_pct": ineq["cv_pct"], "p90_p10_price": ineq["p90_p10_ratio"],
                     "gini_price": ineq["gini"], "theil_between_region_share_pct": ineq["between_share_pct"]})
    return pd.DataFrame(rows)


@st.cache_data
def gi_for(col: str, year: int, k: int = 5) -> pd.DataFrame:
    g = town_panel()
    g = g[(g["year"] == year)].dropna(subset=[col, "lat"]).reset_index(drop=True)
    W = S.knn_weights(g[["lat", "lng"]].to_numpy(), k=k, include_self=True)
    gi = S.getis_ord_gi(g[col].to_numpy(), W, n_perm=4999)
    Wm = S.knn_weights(g[["lat", "lng"]].to_numpy(), k=k)
    m = S.morans_i(g[col].to_numpy(), Wm, n_perm=999)
    return pd.concat([g, gi], axis=1), m


def show():
    st.title("Spatial Inequality")
    st.markdown(
        "Are expensive and unaffordable towns **clustered** together, and is the gap between "
        "places **widening**? Two tests, in order: **global Moran's I** asks whether there is "
        "any spatial clustering at all; only then does **Getis-Ord Gi\\*** say where the hot and "
        "cold spots are. With 26 towns, chance alone produces patterns, so both use permutation "
        "inference and Gi\\* uses a false-discovery-rate correction across towns.")
    if not need("hedonic_town_year_type.csv"):
        return

    c1, c2, c3 = st.columns([1.6, 1, 1])
    var_label = c1.selectbox("Variable", list(VARIABLES))
    col = VARIABLES[var_label]
    year = c2.slider("Year", 2000, 2026, 2026)
    k = c3.slider("Neighbours (k)", 3, 8, 5)
    gi, m = gi_for(col, year, k)

    a, b, c = st.columns(3)
    a.metric("Moran's I", f"{m['I']:.3f}", help="0 = no spatial pattern; >0 similar towns sit together.")
    b.metric("Permutation p-value", f"{m['p_sim']:.3f}")
    c.metric("Significant Gi* spots (FDR<5%)", int((gi["p_fdr"] < 0.05).sum()))
    if m["p_sim"] >= 0.05:
        st.info("No statistically significant global clustering for this variable and year: "
                "read the local hot/cold spots below with that in mind.")

    fig = px.scatter_map(
        gi, lat="lat", lon="lng", color="gi_z", hover_name="town",
        hover_data={col: ":.1f", "gi_z": ":.2f", "p_fdr": ":.3f", "label": True, "lat": False, "lng": False},
        color_continuous_scale="RdBu_r", color_continuous_midpoint=0, size=np.abs(gi["gi_z"]) + 1,
        map_style="open-street-map", center={"lat": 1.355, "lon": 103.82}, zoom=10.2, height=520,
        labels={"gi_z": "Gi* z"}, title=f"Gi* hot (red) and cold (blue) spots: {var_label}, {year}")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(gi[["town", "region", col, "gi_z", "p_sim", "p_fdr", "label"]]
                 .sort_values("gi_z", ascending=False).round(3),
                 width="stretch", hide_index=True)

    # ---- over time ----
    st.header("Is spatial inequality changing?")
    ms = moran_series(col, k)
    ms["significant"] = np.where(ms["p_sim"] < 0.05, "p < 0.05", "not significant")
    fig = px.scatter(ms, x="year", y="morans_I", color="significant",
                     title=f"Global Moran's I over time: {var_label}",
                     labels={"morans_I": "Moran's I", "year": ""})
    fig.add_scatter(x=ms["year"], y=ms["morans_I"], mode="lines", line=dict(color="grey"), showlegend=False)
    fig.add_hline(y=-1 / (ms["n_towns"].median() - 1), line_dash="dot", annotation_text="E[I] under no pattern")
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    fig = px.line(ms, x="year", y=["cv_price_pct"], markers=True,
                  labels={"value": "Coefficient of variation (%)", "year": ""},
                  title="Dispersion of constant-quality 4-room prices across towns")
    c1.plotly_chart(fig, width="stretch")
    fig = px.line(ms, x="year", y="theil_between_region_share_pct", markers=True,
                  labels={"theil_between_region_share_pct": "% of inequality between regions", "year": ""},
                  title="How much of the gap is between regions (vs within them)?")
    c2.plotly_chart(fig, width="stretch")
    st.caption(
        "Price dispersion uses the same reference flat everywhere, so it measures how far apart "
        "locations are, not differences in flat mix. With a single national income distribution, "
        "cross-town inequality in the affordability index is exactly this price dispersion; the "
        "Affordability page's local-income lens adds incomes back in.")

    # ---- robustness ----
    st.header("How robust is the clustering trend?")
    if store.exists("robust_moran_summary.csv"):
        rs = store.table("robust_moran_summary.csv")
        n_up = int((rs["change"] > 0).sum())
        n_sig = int(((rs["mann_kendall_p"] < 0.05) & (rs["mann_kendall_z"] > 0)).sum())
        has_hr = "mk_hamed_rao_p" in rs.columns
        a, b, c = st.columns(3)
        a.metric("Specifications tested", len(rs),
                 help="24 spatial-weight definitions plus a balanced panel of towns present in every year. "
                      "They share the same transactions, so they are robustness checks, not independent replications.")
        b.metric("Show clustering rising 1995-99 → 2022-26", f"{n_up} of {len(rs)}")
        if has_hr:
            n_hr = int(((rs["mk_hamed_rao_p"] < 0.05) & (rs["change"] > 0)).sum())
            c.metric("Rising trend significant, corrected for serial correlation", f"{n_hr} of {len(rs)}",
                     help=f"Hamed-Rao corrected Mann-Kendall, p < 0.05. Adjacent years are correlated, so the "
                          f"ordinary test (significant in {n_sig} of {len(rs)}) overstates confidence.")
            bal = rs[rs["spec"].str.contains("balanced")]
            msg = (f"The exception is the balanced panel of towns present every year "
                   f"(Hamed-Rao p = {bal['mk_hamed_rao_p'].iloc[0]:.3f}): on that smaller set the rise is suggestive, "
                   f"not established. ") if len(bal) and bal["mk_hamed_rao_p"].iloc[0] >= 0.05 else ""
            if store.exists("robust_trend_sensitivity.csv"):
                ts = store.table("robust_trend_sensitivity.csv")
                k = len(ts)
                n_ = lambda c: int((ts[c] < 0.05).sum()) if c in ts.columns else None  # noqa: E731
                msg += (f"With only 32 yearly values every trend test over-rejects when persistence is high, so "
                        f"simulations ask directly how often persistence alone would produce a rise this consistent. "
                        f"p < 0.05 in {n_('sim_p_fitted_phi')} of {k} specifications under the estimated AR(1)")
                if "sim_p_ar2" in ts.columns:
                    msg += (f", {n_('sim_p_ar2')} of {k} under an AR(2), {n_('sim_p_phi_uncertain')} of {k} with the "
                            f"persistence estimate's own uncertainty included")
                msg += f", and {n_('sim_p_phi_0_9')} of {k} even at a pessimistic 0.9."
            if store.exists("robust_trend_breaks.csv"):
                br = store.table("robust_trend_breaks.csv")
                both = int(((br["sen_slope_per_decade_before"] > 0) & (br["sen_slope_per_decade_after"] > 0)).sum())
                msg += (f" A steady rise rather than a one-off jump: a linear trend beats the best single level shift "
                        f"in {int((br['preferred_by_bic'] == 'linear trend').sum())} of {len(br)} specifications, and "
                        f"clustering rises on both sides of the break in {both} of {len(br)}. The specifications share "
                        f"the same data, so these are robustness checks, not independent confirmations.")
            st.caption(msg)
        else:
            c.metric("Rising trend significant (Mann-Kendall p<0.05)", f"{n_sig} of {len(rs)}")
        specs = store.table("robust_moran_specs.csv")
        fig = px.line(specs, x="year", y="I", color="spec", height=480,
                      labels={"I": "Moran's I of location premium", "year": "", "spec": "Weights"},
                      title="Moran's I under every specification (town level)")
        st.plotly_chart(fig, width="stretch")
        show = rs.sort_values("change").round(3)
        for col in ("mann_kendall_p", "mk_hamed_rao_p", "mk_tfpw_p", "mk_prewhitened_p"):
            if col in rs.columns:
                show[col] = rs.loc[show.index, col].map(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")
        st.dataframe(show, width="stretch", hide_index=True)
    if store.exists("robust_moran_ci.csv"):
        ci = store.table("robust_moran_ci.csv")
        fig = go.Figure()
        fig.add_scatter(x=ci["year"], y=ci["hi95"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=ci["year"], y=ci["lo95"], mode="lines", fill="tonexty", line=dict(width=0),
                        fillcolor="rgba(31,119,180,0.2)", name="95% bootstrap interval")
        fig.add_scatter(x=ci["year"], y=ci["I"], mode="lines+markers", name="Moran's I (k=5)")
        fig.update_layout(height=380, yaxis_title="Moran's I", title="Sampling uncertainty (100 bootstrap re-fits per year)")
        st.plotly_chart(fig, width="stretch")
    if store.exists("robust_block_moran.csv"):
        bm = store.table("robust_block_moran.csv")
        bm = bm[~bm["window"].str.contains("partial")]  # a single part-year has too few blocks to compare
        fig = px.line(bm, x="window", y="I", color=bm["band_km"].astype(str) + " km", markers=True,
                      labels={"I": "Moran's I", "window": "3-year window", "color": "distance band"},
                      title="A different unit: block-level clustering of location premia")
        st.plotly_chart(fig, width="stretch")
    if store.exists("robust_premium_ci.csv"):
        pc = store.table("robust_premium_ci.csv")
        pc = pc[pc["year"] == year].sort_values("premium_pct")
        if not pc.empty:
            fig = go.Figure(go.Scatter(x=pc["premium_pct"], y=pc["town"], mode="markers",
                                       error_x=dict(type="data", symmetric=False,
                                                    array=pc["hi95_pct"] - pc["premium_pct"],
                                                    arrayminus=pc["premium_pct"] - pc["lo95_pct"])))
            fig.update_layout(height=650, xaxis_title=f"Location premium vs national, {year} (%, 95% CI)",
                              title="Town location premia with bootstrap intervals")
            st.plotly_chart(fig, width="stretch")
    if store.exists("robust_gi_stability.csv"):
        gs = store.table("robust_gi_stability.csv")
        yrs = sorted(gs["year"].unique())
        gy = st.selectbox("Gi* stability for year", yrs, index=len(yrs) - 1)
        g = gs[gs["year"] == gy]
        g = g[(g["share_hot_pct"] > 0) | (g["share_cold_pct"] > 0)].sort_values("share_hot_pct")
        st.markdown(f"Share of 100 bootstrap re-fits in which each town is a significant (FDR 5%) "
                    f"hot or cold spot, {gy}. Towns never flagged are omitted.")
        st.dataframe(g, width="stretch", hide_index=True)
    if store.exists("robust_gi_by_k.csv"):
        gk = store.table("robust_gi_by_k.csv")
        piv = gk.pivot(index="town", columns="k", values="label")
        piv = piv[(piv != "Not significant").any(axis=1)]
        st.markdown("Gi* labels for the latest year as the number of neighbours k changes:")
        st.dataframe(piv, width="stretch")
    if store.exists("robust_theil_by_type.csv"):
        th = store.table("robust_theil_by_type.csv")
        fig = px.line(th, x="year", y="between_share_pct", color="flat_type", markers=True,
                      labels={"between_share_pct": "% of inequality between regions", "year": ""},
                      title="Theil decomposition by flat type (observed town medians)")
        st.plotly_chart(fig, width="stretch")
        st.caption("Uses observed medians because constant-quality prices differ across flat types only by "
                   "a constant each year, which leaves the Theil index unchanged.")

    # ---- growth ----
    st.header("Where did prices grow unusually fast?")
    c1, c2 = st.columns(2)
    start = c1.selectbox("From", list(range(2000, 2026)), index=19)
    end = c2.selectbox("To", list(range(2001, 2027)), index=25)
    if end > start:
        g = growth_table(start, end)
        gs = g.sort_values("total_growth_pct")
        fig = px.bar(gs, x="total_growth_pct", y="town", color="flag",
                     orientation="h", height=650,
                     color_discrete_map={"Unusually fast": "#d62728", "Unusually slow": "#1f77b4", "": "#aaaaaa"},
                     labels={"total_growth_pct": f"Constant-quality growth {start}-{end} (%)", "town": ""})
        fig.update_yaxes(categoryorder="array", categoryarray=gs["town"].tolist())
        st.plotly_chart(fig, width="stretch")
        st.caption("Flags use a robust z-score (median and MAD across towns); |z| >= 2 is flagged.")

    # ---- block level ----
    st.header("Block-level hot spots")
    if not store.exists("block_premium.csv"):
        st.info("Block coordinates not built yet: run `python pipeline/geocode_blocks.py` "
                "then `python pipeline/build_analytics.py`.")
    else:
        bp = store.table("block_premium.csv")
        meta = store.meta().get("block_morans_i") or {}
        a, b, c = st.columns(3)
        a.metric("Blocks analysed", f"{len(bp):,}")
        if meta:
            b.metric("Block-level Moran's I", f"{meta['I']:.3f}", help=f"{meta['band_km']} km band, {meta['years']}")
            c.metric("Permutation p", f"{meta['p_sim']:.3f}")
        view = st.radio("Show", ["Location premium", "Gi* hot / cold spots"], horizontal=True)
        towns = st.multiselect("Filter towns (empty = all)", sorted(bp["town"].unique()))
        sub = bp[bp["town"].isin(towns)] if towns else bp
        if view == "Location premium":
            fig = px.scatter_map(sub, lat="lat", lon="lng", color="premium_pct",
                                    color_continuous_scale="RdYlBu_r", range_color=(-30, 40),
                                    hover_name="block_key", hover_data={"town": True, "n": True,
                                                                       "premium_pct": ":.1f", "within_town_pct": ":.1f",
                                                                       "lat": False, "lng": False},
                                    map_style="carto-positron", center={"lat": 1.355, "lon": 103.82},
                                    zoom=10.4, height=620, labels={"premium_pct": "Premium vs national (%)"})
        else:
            fig = px.scatter_map(sub, lat="lat", lon="lng", color="hotspot",
                                    color_discrete_map={"Hot spot": "#d62728", "Cold spot": "#1f77b4",
                                                        "Not significant": "#cccccc"},
                                    hover_name="block_key", hover_data={"gi_z": ":.2f", "town": True,
                                                                       "lat": False, "lng": False},
                                    map_style="carto-positron", center={"lat": 1.355, "lon": 103.82},
                                    zoom=10.4, height=620)
        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Premium = how much more (or less) a block's flats sold for than the national "
            "constant-quality average, after controlling for flat type, model, size, floor and lease "
            "with that year's hedonic regression; pooled over the last three years, blocks with 5+ sales. "
            "Gi* uses a 1 km distance band with an analytic z and BH-FDR across all blocks.")

    method_note(
        "**Moran's I** (global): correlation between a town's value and its k nearest neighbours' "
        "values, tested by reshuffling values across towns 999 times.\n\n"
        "**Getis-Ord Gi\\*** (local): whether a town *and its neighbours* are jointly higher or lower "
        "than average; conditional permutation p-values (the focal town is held fixed), then "
        "Benjamini-Hochberg FDR across towns.\n\n"
        "**Theil T decomposition**: total inequality across towns = between-region part + "
        "within-region part, exactly; regions are URA's Central, East, North, North-East, West.\n\n"
        "**Unit of analysis**: towns with 30+ sales in the year. Each town counts once (not "
        "population-weighted).")
