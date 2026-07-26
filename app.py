"""
Country Development Explorer & Predictor
==========================================
Streamlit app built on top of the K-Means clustering + XGBoost classification
project. Two tabs:
  1. Explore Countries  - view all 167 countries and their assigned tiers
  2. Predict & Simulate - "what-if" policy simulator, fund planner, and
                           nearest-neighbor country benchmarking

Required files:
  - data/country_data_clustered.csv (from notebook 1)
  - xgb_country_classifier.pkl      (from notebook 2)
  - cluster_mapping.json            (from notebook 2)

Run with:
    streamlit run app.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from st_keyup import st_keyup

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Country Development Explorer",
    page_icon="🌍",
    layout="wide",
)

FEATURES = [
    "child_mortality_per_1000",
    "exports_gdp_percent",
    "health_spending_gdp_percent",
    "imports_gdp_percent",
    "net_income_per_capita_usd",
    "inflation_rate_percent",
    "life_expectancy_years",
    "total_fertility_rate",
    "gdp_per_capita_usd",
]

# Reference only (radar chart context) - NOT used for gap direction anymore,
# see Issue #4 fix below.
LOWER_IS_BETTER = {"child_mortality_per_1000", "inflation_rate_percent", "total_fertility_rate"}

TIER_ORDER = ["Under Developed", "Developing", "Developed"]


# ----------------------------------------------------------------------------
# DATA / MODEL LOADING
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/country_data_clustered.csv")
    return df


@st.cache_resource
def load_model():
    model = joblib.load("xgb_country_classifier.pkl")
    with open("cluster_mapping.json") as f:
        raw_mapping = json.load(f)
    mapping = {int(k): v for k, v in raw_mapping.items()}
    return model, mapping


@st.cache_resource
def fit_neighbor_index(df):
    """Fit a StandardScaler on the raw feature space so hypothetical inputs
    and real countries are compared on a level playing field."""
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(df[FEATURES])
    return scaler, scaled_matrix


try:
    df = load_data()
    model, cluster_mapping = load_model()
    scaler, scaled_matrix = fit_neighbor_index(df)
except FileNotFoundError as e:
    st.error(
        "Missing required file: **{}**\n\n"
        "Make sure `data/country_data_clustered.csv`, `xgb_country_classifier.pkl`, "
        "and `cluster_mapping.json` are in the correct locations.".format(e.filename)
    )
    st.stop()

reverse_mapping = {v: k for k, v in cluster_mapping.items()}
tier_order_present = [t for t in TIER_ORDER if t in reverse_mapping]

# ----------------------------------------------------------------------------
# ISSUE #1 FIX — sidebar becomes an info/settings panel, sliders move to
# the main section (see Tab 2 below).
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏠 App Overview")
    st.caption(
        "Explore 167 countries grouped by K-Means clustering, then simulate "
        "policy changes using an XGBoost classifier trained on the same data."
    )

    st.markdown("### 📊 Dataset Info")
    st.caption(f"{len(df)} countries · {len(FEATURES)} indicators")

    st.markdown("### ⚙️ Settings")
    reset_clicked = st.button("🔄 Reset Inputs")

    st.markdown("### ℹ️ Model Info")
    st.caption("Clustering: K-Means (k=3) · Classifier: XGBoost, trained on the same 167 countries.")

    st.markdown("### 📄 Disclaimer")
    st.caption(
        "Educational/portfolio demo. Fund allocation impact estimates are "
        "simplified and illustrative — not based on real economic cost data."
    )

    st.markdown("---")
    st.caption("👤 Created by Maria Anwar — Portfolio Project")

st.title("🌍 Country Development Explorer")

tab1, tab2 = st.tabs(["📊 Explore Countries", "🔮 Predict & Simulate"])


# ----------------------------------------------------------------------------
# TAB 1 — EXPLORE COUNTRIES
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Country Clusters at a Glance")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.scatter(
            df,
            x="gdp_per_capita_usd",
            y="child_mortality_per_1000",
            color="country_status",
            hover_name="country",
            hover_data={c: True for c in FEATURES},
            title="GDP per Capita vs Child Mortality, colored by Development Tier",
            labels={
                "gdp_per_capita_usd": "GDP per Capita (USD)",
                "child_mortality_per_1000": "Child Mortality (per 1000)",
            },
        )
        fig.update_traces(marker=dict(size=9, opacity=0.8))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Countries per Tier**")
        st.bar_chart(df["country_status"].value_counts())

    st.markdown("---")
    st.subheader("Cluster Profile Summary")
    summary = df.groupby("country_status")[FEATURES].mean().round(2)

    st.dataframe(
        summary,
        column_config={
            "country_status": st.column_config.TextColumn("Tier"),
            "child_mortality_per_1000": st.column_config.NumberColumn("Child Mortality", format="%.1f /1k"),
            "exports_gdp_percent": st.column_config.NumberColumn("Exports (% GDP)", format="%.1f%%"),
            "health_spending_gdp_percent": st.column_config.NumberColumn("Health Spend (% GDP)", format="%.1f%%"),
            "imports_gdp_percent": st.column_config.NumberColumn("Imports (% GDP)", format="%.1f%%"),
            "net_income_per_capita_usd": st.column_config.NumberColumn("Net Income", format="$%d"),
            "inflation_rate_percent": st.column_config.NumberColumn("Inflation Rate", format="%.1f%%"),
            "life_expectancy_years": st.column_config.NumberColumn("Life Expectancy", format="%.1f yrs"),
            "total_fertility_rate": st.column_config.NumberColumn("Fertility Rate", format="%.2f"),
            "gdp_per_capita_usd": st.column_config.ProgressColumn(
                "GDP per Capita",
                format="$%d",
                min_value=0,
                max_value=float(df["gdp_per_capita_usd"].max())
            ),
        },
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Browse the Full Dataset")

    # ------------------------------------------------------------------
    # ISSUE #2 FIX — genuinely live "as-you-type" search (st.text_input
    # only updates on Enter/blur by default - st_keyup fires on every
    # keystroke instead) with a "starts with" match, case-insensitive.
    # ------------------------------------------------------------------
    search = st_keyup("Search for a country", key="country_search", debounce=150)
    search_clean = (search or "").strip()

    if search_clean:
        mask = df["country"].str.lower().str.startswith(search_clean.lower())
        filtered = df[mask]
    else:
        filtered = df  # empty search box -> always show full table

    if search_clean and filtered.empty:
        st.warning(f"No country found matching '{search_clean}'.")
    else:
        st.dataframe(
            filtered[["country", "country_status"] + FEATURES],
            column_config={
                "country": st.column_config.TextColumn("Country", help="Country Name"),
                "country_status": st.column_config.TextColumn("Tier"),
                "child_mortality_per_1000": st.column_config.NumberColumn("Child Mortality", format="%.1f"),
                "exports_gdp_percent": st.column_config.NumberColumn("Exports %", format="%.1f%%"),
                "health_spending_gdp_percent": st.column_config.NumberColumn("Health Spend %", format="%.1f%%"),
                "imports_gdp_percent": st.column_config.NumberColumn("Imports %", format="%.1f%%"),
                "net_income_per_capita_usd": st.column_config.NumberColumn("Income (USD)", format="$%d"),
                "inflation_rate_percent": st.column_config.NumberColumn("Inflation %", format="%.1f%%"),
                "life_expectancy_years": st.column_config.NumberColumn("Life Expectancy", format="%.1f yrs"),
                "total_fertility_rate": st.column_config.NumberColumn("Fertility Rate", format="%.2f"),
                "gdp_per_capita_usd": st.column_config.ProgressColumn(
                    "GDP / Capita",
                    format="$%d",
                    min_value=0,
                    max_value=float(df["gdp_per_capita_usd"].max())
                ),
            },
            hide_index=True,
            use_container_width=True,
            height=400,
        )


# ----------------------------------------------------------------------------
# TAB 2 — PREDICT & SIMULATE
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Build a Country Profile")

    country_list = ["-- Custom / Manual --"] + sorted(df["country"].unique().tolist())
    selected_country = st.selectbox("Prefill sliders from an existing country", country_list)

    if selected_country != "-- Custom / Manual --":
        defaults = df[df["country"] == selected_country].iloc[0]
    else:
        defaults = df[FEATURES].mean()

    # Active profile shown prominently, stays visible through prediction,
    # radar chart, fund calculator, and twins sections.
    if selected_country != "-- Custom / Manual --":
        st.info(f"📌 **Active Profile:** {selected_country} (adjust sidebar sliders below to simulate changes)")
    else:
        st.info("📌 **Active Profile:** Custom / Manual entry")

    # --- Sliders (sidebar) — no explicit key, matching the confirmed-working
    # version: value= is recomputed fresh from the selected country every
    # rerun, so switching countries updates the sliders correctly. ---------
    st.sidebar.markdown("### 🎚️ Adjust Country Indicators")
    st.sidebar.caption("Tweak values to simulate a policy change, then predict.")

    if reset_clicked:
        for k in ["pred_tier", "pred_proba", "input_values"]:
            st.session_state.pop(k, None)

    user_input = {}
    for feat in FEATURES:
        col_min = float(df[feat].min())
        col_max = float(df[feat].max())
        default_val = float(defaults[feat])
        step = round((col_max - col_min) / 200, 2) if col_max > col_min else 1.0
        user_input[feat] = st.sidebar.slider(
            feat.replace("_", " ").title(),
            min_value=col_min,
            max_value=col_max,
            value=default_val,
            step=step if step > 0 else 1.0,
        )

    input_df = pd.DataFrame([user_input])[FEATURES]

    predict_clicked = st.button("🔮 Predict Development Tier", type="primary")

    if predict_clicked:
        pred_class = int(model.predict(input_df)[0])
        pred_proba = model.predict_proba(input_df)[0]
        pred_tier = cluster_mapping[pred_class]

        st.session_state["pred_tier"] = pred_tier
        st.session_state["pred_proba"] = pred_proba
        st.session_state["input_values"] = user_input

    if "pred_tier" in st.session_state:
        pred_tier = st.session_state["pred_tier"]
        pred_proba = st.session_state["pred_proba"]
        user_input = st.session_state["input_values"]
        input_df = pd.DataFrame([user_input])[FEATURES]

        st.markdown("---")
        badge_color = {"Developed": "🟢", "Developing": "🟡", "Under Developed": "🔴"}
        st.markdown(f"## {badge_color.get(pred_tier, '⚪')} Predicted Tier: **{pred_tier}**")

        proba_df = pd.DataFrame({
            "Tier": [cluster_mapping[i] for i in range(len(pred_proba))],
            "Probability": pred_proba,
        }).sort_values("Probability", ascending=False)
        st.bar_chart(proba_df.set_index("Tier"))

        st.markdown("---")

        # ------------------------------------------------------------------
        # FEATURE A: What-If Comparison vs cluster average (design kept as-is)
        # ------------------------------------------------------------------
        st.subheader("📈 What-If Comparison: You vs. Your Predicted Tier's Average")

        cluster_avg = df[df["country_status"] == pred_tier][FEATURES].mean()

        radar_categories = FEATURES
        mins = df[FEATURES].min()
        maxs = df[FEATURES].max()

        user_norm = [(user_input[f] - mins[f]) / (maxs[f] - mins[f] + 1e-9) for f in radar_categories]
        avg_norm = [(cluster_avg[f] - mins[f]) / (maxs[f] - mins[f] + 1e-9) for f in radar_categories]

        categories_clean = [f.replace("_", " ").title() for f in radar_categories]
        categories_closed = categories_clean + [categories_clean[0]]

        user_norm_closed = user_norm + [user_norm[0]]
        avg_norm_closed = avg_norm + [avg_norm[0]]

        user_raw = [user_input[f] for f in radar_categories] + [user_input[radar_categories[0]]]
        avg_raw = [cluster_avg[f] for f in radar_categories] + [cluster_avg[radar_categories[0]]]

        radar_fig = go.Figure()

        radar_fig.add_trace(go.Scatterpolar(
            r=avg_norm_closed,
            theta=categories_closed,
            fill="toself",
            name=f"{pred_tier} Average",
            line=dict(color="#00E5FF", width=2),
            fillcolor="rgba(0, 229, 255, 0.15)",
            mode="lines+markers",
            customdata=avg_raw,
            hovertemplate="%{theta}<br>" + f"{pred_tier} Average: " + "%{customdata:.1f}<extra></extra>",
        ))

        radar_fig.add_trace(go.Scatterpolar(
            r=user_norm_closed,
            theta=categories_closed,
            fill="toself",
            name="Your Input",
            line=dict(color="#FF4081", width=3),
            fillcolor="rgba(255, 64, 129, 0.25)",
            mode="lines+markers",
            customdata=user_raw,
            hovertemplate="%{theta}<br>Your Input: %{customdata:.1f}<extra></extra>",
        ))

        radar_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor="rgba(255, 255, 255, 0.03)",
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    showticklabels=True,
                    tickfont=dict(color="#FFFFFF", size=11),
                    gridcolor="rgba(255, 255, 255, 0.2)",
                    linecolor="rgba(255, 255, 255, 0.4)",
                    dtick=0.2
                ),
                angularaxis=dict(
                    tickfont=dict(color="#FFFFFF", size=12),
                    gridcolor="rgba(255, 255, 255, 0.2)",
                    linecolor="rgba(255, 255, 255, 0.4)"
                )
            ),
            showlegend=True,
            legend=dict(
                font=dict(color="#FFFFFF", size=12),
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=80, r=80, t=40, b=80),
            title=dict(
                text="Normalized Comparison (0 = Dataset Min, 1 = Dataset Max)",
                font=dict(color="#CCCCCC", size=13)
            )
        )

        st.plotly_chart(radar_fig, use_container_width=True)

        st.markdown("---")

        # ------------------------------------------------------------------
        # FEATURE B: Resource Allocation Calculator
        # ------------------------------------------------------------------
        st.subheader("💰 Resource Allocation Calculator")

        if pred_tier == "Developed":
            st.success("This profile is already predicted as **Developed** — no gap to close.")
        else:
            current_idx = tier_order_present.index(pred_tier)
            next_tier = tier_order_present[current_idx + 1]
            next_avg = df[df["country_status"] == next_tier][FEATURES].mean()

            st.markdown(f"**Target: reach the '{next_tier}' tier**")

            # ISSUE #4 FIX — direction is now purely based on whether the
            # target average is above or below the current value. The old
            # LOWER_IS_BETTER branch was inverted for mortality/fertility,
            # incorrectly telling users to "increase" when they needed to
            # "decrease".
            # Direction is now based on the DEVELOPED tier average (the real
            # end goal - unambiguously the best value for every feature),
            # not just the immediate next tier. Some indicators (notably
            # health spending % of GDP) are NOT perfectly monotonic across
            # tiers in this dataset - e.g. "Developing" countries can show a
            # slightly LOWER health-spending percentage than "Under
            # Developed" ones (a low-GDP country can show a high % while
            # spending very little in real dollars). Comparing only to the
            # next tier in that case would wrongly recommend "decrease" for
            # something that should clearly increase. A caveat is shown
            # whenever the next-tier average disagrees with the final target.
            developed_avg = df[df["country_status"] == "Developed"][FEATURES].mean()

            gap_rows = []
            quirk_flags = []
            for feat in ["gdp_per_capita_usd", "child_mortality_per_1000",
                         "health_spending_gdp_percent", "total_fertility_rate"]:
                current_val = user_input[feat]
                target_val = next_avg[feat]          # nearer milestone, shown as the numeric target
                final_val = developed_avg[feat]       # real end goal, used to decide direction

                gap = target_val - current_val
                final_gap = final_val - current_val
                direction = "increase" if final_gap > 0 else "decrease"

                # Flag if the next tier's average disagrees with the final target's direction
                is_quirk = (final_gap > 0) != (gap > 0)
                if is_quirk:
                    quirk_flags.append(feat.replace("_", " ").title())

                gap_rows.append({
                    "Indicator": feat.replace("_", " ").title() + (" ⚠️" if is_quirk else ""),
                    "Current": round(current_val, 2),
                    f"{next_tier} Avg": round(target_val, 2),
                    "Gap": round(abs(gap), 2),
                    "Direction Needed": direction,
                })
            st.dataframe(pd.DataFrame(gap_rows), use_container_width=True)

            if quirk_flags:
                st.caption(
                    f"⚠️ **{', '.join(quirk_flags)}** — this dataset's '{next_tier}' average for this "
                    f"indicator happens to sit slightly worse than the 'Under Developed' average (a real "
                    f"data pattern, not an error). Direction shown above is based on the final Developed-"
                    f"tier target instead, so the recommendation stays logically consistent."
                )

            st.markdown("#### Simulate a Fund Allocation")
            st.caption(
                "⚠️ Simplified illustrative model — assumes a fixed budget fully allocated "
                "toward closing the gap can close 100% of it. Not based on real economic "
                "cost data. For demonstration purposes only."
            )

            fund_size = st.number_input("Total Available Fund (Million USD)", min_value=1, value=10, step=1)

            c1, c2, c3 = st.columns(3)
            with c1:
                pct_health = st.slider("Health %", 0, 100, 40)
            with c2:
                pct_education = st.slider("Education/Social %", 0, 100, 30)
            with c3:
                pct_econ = st.slider("Economic Infrastructure %", 0, 100, 30)

            total_pct = pct_health + pct_education + pct_econ
            if total_pct != 100:
                st.warning(f"Allocations sum to {total_pct}%, not 100%. Adjust sliders so they total 100%.")
            else:
                # REFERENCE_FUND is the illustrative benchmark size: a category
                # that receives $50M (the full amount if 100% of the fund were
                # put toward it) is assumed to fully close that category's gap.
                # A smaller allocation closes a proportional fraction of it
                # (e.g. $25M toward health = 50% of the way there). This number
                # isn't from real cost data - see the disclaimer above.
                REFERENCE_FUND = 50.0
                st.caption(f"ℹ️ Assumption: a full \\${REFERENCE_FUND:.0f}M allocated to one category is treated as enough to fully close that category's gap.")

                health_effect = min(1.0, (fund_size * pct_health / 100) / REFERENCE_FUND)
                econ_effect = min(1.0, (fund_size * pct_econ / 100) / REFERENCE_FUND)
                edu_effect = min(1.0, (fund_size * pct_education / 100) / REFERENCE_FUND)

                new_mortality = user_input["child_mortality_per_1000"] - \
                    health_effect * (user_input["child_mortality_per_1000"] - next_avg["child_mortality_per_1000"])
                new_gdp = user_input["gdp_per_capita_usd"] + \
                    econ_effect * (next_avg["gdp_per_capita_usd"] - user_input["gdp_per_capita_usd"])
                new_fertility = user_input["total_fertility_rate"] - \
                    edu_effect * (user_input["total_fertility_rate"] - next_avg["total_fertility_rate"])

                # ISSUE #5 FIX — propagate improvements to the other 6
                # features the model was trained on, instead of leaving
                # them frozen at their original values (which kept the
                # prediction stuck even after a big simulated investment).
                # Uses developed_avg (the final target) instead of next_avg
                # here specifically, since health spending % isn't always
                # monotonic between "Under Developed" and "Developing" in
                # this dataset - using next_avg could simulate MORE funding
                # pushing health spending DOWN, which makes no sense.
                new_health_spending = user_input["health_spending_gdp_percent"] + \
                    health_effect * (developed_avg["health_spending_gdp_percent"] - user_input["health_spending_gdp_percent"])

                new_life_expectancy = user_input["life_expectancy_years"] + \
                    health_effect * (next_avg["life_expectancy_years"] - user_input["life_expectancy_years"])

                gdp_before = user_input["gdp_per_capita_usd"]
                gdp_ratio = (new_gdp / gdp_before) if gdp_before > 0 else 1.0
                new_net_income = user_input["net_income_per_capita_usd"] * gdp_ratio

                st.markdown("**Estimated Impact After This Allocation:**")
                impact_cols = st.columns(3)
                impact_cols[0].metric("Child Mortality", f"{new_mortality:.1f}",
                                       f"{new_mortality - user_input['child_mortality_per_1000']:.1f}")
                impact_cols[1].metric("GDP per Capita ($)", f"{new_gdp:,.0f}",
                                       f"{new_gdp - user_input['gdp_per_capita_usd']:,.0f}")
                impact_cols[2].metric("Fertility Rate", f"{new_fertility:.2f}",
                                       f"{new_fertility - user_input['total_fertility_rate']:.2f}")

                impact_cols2 = st.columns(3)
                impact_cols2[0].metric("Life Expectancy", f"{new_life_expectancy:.1f} yrs",
                                        f"{new_life_expectancy - user_input['life_expectancy_years']:.1f}")
                impact_cols2[1].metric("Net Income ($)", f"{new_net_income:,.0f}",
                                        f"{new_net_income - user_input['net_income_per_capita_usd']:,.0f}")
                impact_cols2[2].metric("Health Spending (% GDP)", f"{new_health_spending:.1f}%",
                                        f"{new_health_spending - user_input['health_spending_gdp_percent']:.1f}")

                sim_input = dict(user_input)
                sim_input["child_mortality_per_1000"] = new_mortality
                sim_input["gdp_per_capita_usd"] = new_gdp
                sim_input["total_fertility_rate"] = new_fertility
                sim_input["health_spending_gdp_percent"] = new_health_spending
                sim_input["life_expectancy_years"] = new_life_expectancy
                sim_input["net_income_per_capita_usd"] = new_net_income
                sim_df = pd.DataFrame([sim_input])[FEATURES]
                sim_class = int(model.predict(sim_df)[0])
                sim_tier = cluster_mapping[sim_class]

                if sim_tier != pred_tier:
                    st.success(f"🎉 With this allocation, the predicted tier improves to **{sim_tier}**!")
                else:
                    st.info(f"With this allocation, the predicted tier remains **{sim_tier}** — try increasing the fund or reallocating.")

        st.markdown("---")

        # ------------------------------------------------------------------
        # FEATURE C: Nearest Neighbor Benchmark
        # ISSUE #6 FIX — switched from Euclidean-distance-relative-to-worst-
        # of-4-neighbors (which mathematically forces the last one toward
        # 0%) to cosine similarity computed against the WHOLE dataset. This
        # gives genuinely meaningful percentages, and a low-similarity note
        # is shown instead of silently padding out to 3 weak matches.
        # ------------------------------------------------------------------
        st.subheader("🌐 Economic Twins — Most Similar Real Countries")

        scaled_input = scaler.transform(input_df)
        sims = cosine_similarity(scaled_input, scaled_matrix)[0]  # -1 to 1, higher = more similar

        sim_series = pd.Series(sims, index=df.index)
        non_self_mask = sim_series < 0.999999  # exclude an exact self-match
        top_idx = sim_series[non_self_mask].sort_values(ascending=False).head(3).index

        neighbor_rows = []
        for idx in top_idx:
            row = df.loc[idx]
            similarity_pct = max(0, sim_series[idx] * 100)
            neighbor_rows.append((row, similarity_pct))

        if neighbor_rows and neighbor_rows[0][1] < 50:
            st.caption(
                "⚠️ No strongly similar country found in the dataset for this profile — "
                "showing the closest available matches, but similarity is low."
            )

        cols = st.columns(3)
        for col, (row, sim_pct) in zip(cols, neighbor_rows):
            with col:
                st.markdown(f"### {row['country']}")
                st.markdown(f"**Tier:** {row['country_status']}")
                st.metric("Similarity", f"{sim_pct:.0f}%")

        st.markdown("**Side-by-Side Comparison**")
        compare_data = {"Your Input": user_input}
        for row, _ in neighbor_rows:
            compare_data[row["country"]] = {f: row[f] for f in FEATURES}
        compare_df = pd.DataFrame(compare_data).round(2)
        st.dataframe(compare_df, use_container_width=True)

    else:
        st.info("Adjust the sliders in the sidebar and click **Predict Development Tier** to begin.")

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "© 2026 Maria Anwar · Country Development Explorer · Built for portfolio/educational purposes. "
    "Data and model outputs are illustrative and should not be used for real policy or investment decisions.  \n"
    "[GitHub](https://github.com/Maria-476) · [LinkedIn](https://www.linkedin.com/in/maria-anwar4786)"
)