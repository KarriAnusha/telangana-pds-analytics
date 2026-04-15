"""
Telangana PDS Analytics — Interactive Streamlit Dashboard
=========================================================
Provides:
 • District / Year filters
 • Geospatial map color-coded by Cluster ID
 • Shop Search tool (compare shop vs cluster average)
 • Cluster Profile Report
 • Geospatial Hotspot Map
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data_processing import (
    load_all_datasets, create_unified_dataset, engineer_features,
    compute_shop_level_features, attach_location_info,
    scale_features, run_pca, find_optimal_k,
    run_kmeans, run_dbscan, build_cluster_profiles,
    flag_suspicious_shops, identify_portability_hubs,
    pre_filter_extreme_outliers, assign_cluster_personas,
    CLUSTER_FEATURES,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telangana PDS Analytics",
    page_icon="📊",
    layout="wide",
)

DATA_ROOT = "data"
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")


# ── Data loading (cached) ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and processing data …")
def load_processed_data():
    """Try loading pre-processed CSVs first; otherwise run the full pipeline."""
    shop_path = os.path.join(PROCESSED_DIR, "shop_features_clustered.csv")
    unified_agg_path = os.path.join(PROCESSED_DIR, "unified_monthly_agg.csv")
    unified_path = os.path.join(PROCESSED_DIR, "unified_dataset.csv")
    profile_path = os.path.join(PROCESSED_DIR, "cluster_profiles.csv")
    suspicious_path = os.path.join(PROCESSED_DIR, "suspicious_shops.csv")
    hubs_path = os.path.join(PROCESSED_DIR, "portability_hubs.csv")
    diagnostics_path = os.path.join(PROCESSED_DIR, "cluster_diagnostics.csv")

    if os.path.exists(shop_path) and (os.path.exists(unified_agg_path) or os.path.exists(unified_path)):
        shop_features = pd.read_csv(shop_path)
        # Prefer lightweight aggregated file; fall back to full unified if needed
        if os.path.exists(unified_agg_path):
            unified = pd.read_csv(unified_agg_path)
        else:
            unified = pd.read_csv(unified_path)
        profiles = pd.read_csv(profile_path, index_col=0) if os.path.exists(profile_path) else None
        suspicious = pd.read_csv(suspicious_path) if os.path.exists(suspicious_path) else None
        hubs = pd.read_csv(hubs_path) if os.path.exists(hubs_path) else None
        diagnostics = pd.read_csv(diagnostics_path) if os.path.exists(diagnostics_path) else None
        return unified, shop_features, profiles, suspicious, hubs, diagnostics

    # Run pipeline from raw CSVs
    transactions, card_status, fps_locations = load_all_datasets(DATA_ROOT)
    unified = create_unified_dataset(transactions, card_status, fps_locations)
    unified = engineer_features(unified)

    shop_features = compute_shop_level_features(unified)
    shop_features = attach_location_info(shop_features, fps_locations)

    X_scaled, _, active_features = scale_features(shop_features)
    X_pca, _ = run_pca(X_scaled, n_components=3)
    shop_features["pca1"] = X_pca[:, 0]
    shop_features["pca2"] = X_pca[:, 1]
    shop_features["pca3"] = X_pca[:, 2] if X_pca.shape[1] >= 3 else 0

    km_labels, _ = run_kmeans(X_scaled, n_clusters=5)
    shop_features["kmeans_cluster"] = km_labels

    db_labels, _ = run_dbscan(X_scaled, eps=1.5, min_samples=5)
    shop_features["dbscan_cluster"] = db_labels

    profiles = build_cluster_profiles(shop_features, features=active_features)
    suspicious = flag_suspicious_shops(shop_features)
    hubs = identify_portability_hubs(shop_features)
    k_vals, inertias, sil_scores = find_optimal_k(X_scaled)
    diagnostics = pd.DataFrame({
        "k": k_vals,
        "inertia": inertias,
        "silhouette": sil_scores,
    })

    unified_export_cols = [
        "shopNo",
        "distCode",
        "distName",
        "year",
        "month",
        "noOfTrans",
        "otherShopTrans",
        "totalRcs",
        "riceQty",
        "wheatQty",
    ]
    unified_export = unified[[c for c in unified_export_cols if c in unified.columns]].copy()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    shop_features.to_csv(os.path.join(PROCESSED_DIR, "shop_features_clustered.csv"), index=False)
    unified_export.to_csv(os.path.join(PROCESSED_DIR, "unified_dataset.csv"), index=False)
    profiles.to_csv(os.path.join(PROCESSED_DIR, "cluster_profiles.csv"))
    suspicious.to_csv(os.path.join(PROCESSED_DIR, "suspicious_shops.csv"), index=False)
    hubs.to_csv(os.path.join(PROCESSED_DIR, "portability_hubs.csv"), index=False)
    diagnostics.to_csv(os.path.join(PROCESSED_DIR, "cluster_diagnostics.csv"), index=False)

    return unified_export, shop_features, profiles, suspicious, hubs, diagnostics


# ── Load data ────────────────────────────────────────────────────────────────
try:
    unified, shop_features, cluster_profiles, suspicious_shops, portability_hubs, diagnostics = load_processed_data()
    DATA_LOADED = True
except FileNotFoundError as e:
    DATA_LOADED = False
    st.error(
        "**Data not found.** Place your CSV files in the following folders and reload:\n"
        "- `data/transactions/`\n"
        "- `data/card_status/`\n"
        "- `data/fps_locations/`\n\n"
        "Or run the Jupyter notebook first to generate `data/processed/` files."
    )
    st.stop()


# ── Helpers ──────────────────────────────────────────────────────────────────
dist_col = "distName" if "distName" in shop_features.columns else "distCode"
has_coords = "latitude" in shop_features.columns and "longitude" in shop_features.columns
has_personas = "clusterPersona" in shop_features.columns

PERSONA_COLORS = {
    "Stable Rural Shops": "#2ca02c",
    "Active Urban Shops": "#1f77b4",
    "High-Volume Urban Hubs": "#ff7f0e",
    "Volatile Portability Hubs": "#d62728",
    "Anomalous High-Utilization": "#9467bd",
    "Extreme Anomaly (Pre-filtered)": "#e377c2",
}
CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ── Sidebar filters ─────────────────────────────────────────────────────────
st.sidebar.title("Filters")

districts = sorted(shop_features[dist_col].dropna().unique())
selected_districts = st.sidebar.multiselect("Select District(s)", districts, default=districts)

years = sorted(unified["year"].dropna().unique())
selected_years = st.sidebar.multiselect("Select Year(s)", years, default=years) if years else []

# Filter unified data by year; shop features by district
if "year" in unified.columns and selected_years:
    unified_filtered = unified[unified["year"].isin(selected_years)]
else:
    unified_filtered = unified.copy()
shop_filtered = shop_features[shop_features[dist_col].isin(selected_districts)]


# ── Title ────────────────────────────────────────────────────────────────────
st.title("Telangana PDS Analytics Dashboard")
st.markdown("Multi-Dimensional Shop Performance Clustering & Anomaly Profiling")

# ── KPI row ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Shops", f"{shop_filtered['shopNo'].nunique():,}")
c2.metric("Total Transactions", f"{unified_filtered['noOfTrans'].sum():,.0f}")
c3.metric("K-Means Clusters", shop_filtered["kmeans_cluster"].nunique())
c4.metric("DBSCAN Outliers", int((shop_filtered["dbscan_cluster"] == -1).sum()))

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
tab_map, tab_search, tab_profile, tab_hotspot, tab_policy, tab_risk = st.tabs(
    [
        "Geospatial Map",
        "Shop Search",
        "Cluster Profiles",
        "Hotspot Map",
        "Policy Impact",
        "Risk & Logistics",
    ]
)

# ── TAB 1: Geospatial Map (shops color-coded by Cluster ID) ─────────────────
with tab_map:
    st.subheader("Shops Color-Coded by Cluster Persona")
    if not has_coords:
        st.warning("Latitude/Longitude columns not found in the FPS location data.")
    else:
        map_df = shop_filtered.dropna(subset=["latitude", "longitude"]).copy()
        if map_df.empty:
            st.info("No shops with valid coordinates for the selected filters.")
        else:
            if has_personas:
                map_df["persona_label"] = map_df["clusterPersona"].fillna("Unknown")
            else:
                map_df["persona_label"] = map_df["kmeans_cluster"].astype(str)

            fig_map = px.scatter_mapbox(
                map_df,
                lat="latitude",
                lon="longitude",
                color="persona_label",
                color_discrete_map=PERSONA_COLORS if has_personas else None,
                hover_data={"shopNo": True, dist_col: True, "totalTransactions": ":,.0f",
                            "latitude": False, "longitude": False, "persona_label": False},
                category_orders={"persona_label": list(PERSONA_COLORS.keys())} if has_personas else None,
                zoom=6,
                center={"lat": map_df["latitude"].median(), "lon": map_df["longitude"].median()},
                mapbox_style="carto-positron",
                height=620,
                opacity=0.7,
            )
            fig_map.update_traces(marker_size=5)
            fig_map.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                legend_title_text="Persona",
            )
            st.plotly_chart(fig_map, use_container_width=True)

# ── TAB 2: Shop Search Tool ─────────────────────────────────────────────────
with tab_search:
    st.subheader("Shop Performance vs Cluster Average")
    shop_input = st.text_input("Enter Shop Number (shopNo)")

    if shop_input:
        try:
            shop_id = int(shop_input)
        except ValueError:
            shop_id = shop_input

        match = shop_features[shop_features["shopNo"] == shop_id]
        if match.empty:
            st.warning(f"Shop `{shop_input}` not found.")
        else:
            shop_row = match.iloc[0]
            cluster_id = int(shop_row["kmeans_cluster"])
            persona = shop_row.get("clusterPersona", f"Cluster {cluster_id}")
            st.success(f"Shop **{shop_id}** belongs to **{persona}** (Cluster {cluster_id})")

            cluster_avg = cluster_profiles.loc[cluster_id] if cluster_profiles is not None else None
            compare_features = [f for f in CLUSTER_FEATURES if f in shop_row.index]

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Shop Values**")
                st.dataframe(shop_row[compare_features].to_frame("Value"), use_container_width=True)

            with col_b:
                if cluster_avg is not None:
                    st.markdown(f"**Cluster {cluster_id} Average**")
                    avg_vals = cluster_avg[compare_features]
                    st.dataframe(avg_vals.to_frame("Cluster Avg"), use_container_width=True)

            if cluster_avg is not None:
                shop_vals = shop_row[compare_features].values.astype(float)
                avg_vals_arr = cluster_avg[compare_features].values.astype(float)
                fig = go.Figure()
                fig.add_trace(go.Bar(name=f"Shop {shop_id}", x=compare_features, y=shop_vals))
                fig.add_trace(go.Bar(name=f"Cluster {cluster_id} Avg", x=compare_features, y=avg_vals_arr))
                fig.update_layout(barmode="group", title="Shop vs Cluster Average",
                                  xaxis_tickangle=-45, height=450)
                st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: Cluster Profile Report ───────────────────────────────────────────
with tab_profile:
    st.subheader("K-Means Cluster Profile Report")
    if cluster_profiles is not None:
        numeric_cols = cluster_profiles.select_dtypes(include="number").columns
        fmt = {c: "{:.2f}" for c in numeric_cols}
        st.dataframe(cluster_profiles.style.format(fmt), use_container_width=True)

        profile_features = [f for f in CLUSTER_FEATURES if f in cluster_profiles.columns]
        fig = px.bar(
            cluster_profiles.reset_index().melt(
                id_vars=cluster_profiles.index.name or "kmeans_cluster",
                value_vars=profile_features,
            ),
            x=cluster_profiles.index.name or "kmeans_cluster",
            y="value",
            color="variable",
            barmode="group",
            title="Feature Means per Cluster",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("DBSCAN Outlier Summary")
    if "dbscan_cluster" in shop_features.columns:
        outliers = shop_features[shop_features["dbscan_cluster"] == -1]
        st.write(f"**{len(outliers)}** shops flagged as outliers by DBSCAN")
        if not outliers.empty:
            st.dataframe(
                outliers[["shopNo", dist_col] + CLUSTER_FEATURES].head(50),
                use_container_width=True,
            )

    st.subheader("Clustering Diagnostics")
    if diagnostics is not None and not diagnostics.empty:
        left, right = st.columns(2)
        with left:
            fig_elbow = px.line(
                diagnostics,
                x="k",
                y="inertia",
                markers=True,
                title="Elbow Curve (K vs Inertia)",
            )
            st.plotly_chart(fig_elbow, use_container_width=True)
        with right:
            fig_sil = px.line(
                diagnostics,
                x="k",
                y="silhouette",
                markers=True,
                title="Silhouette Score by K",
            )
            st.plotly_chart(fig_sil, use_container_width=True)

# ── TAB 4: Geospatial Hotspot Map ───────────────────────────────────────────
with tab_hotspot:
    st.subheader("Geospatial Hotspot Map — Cluster Concentrations")
    if not has_coords:
        st.warning("Latitude/Longitude columns not found.")
    else:
        hotspot_df = shop_filtered.dropna(subset=["latitude", "longitude"]).copy()
        if hotspot_df.empty:
            st.info("No valid coordinates for selected filters.")
        else:
            # Build cluster label options using persona names when available
            cluster_options = sorted(hotspot_df["kmeans_cluster"].unique())
            if has_personas:
                cluster_label_map = (
                    hotspot_df.dropna(subset=["clusterPersona"])
                    .drop_duplicates("kmeans_cluster")
                    .set_index("kmeans_cluster")["clusterPersona"]
                    .to_dict()
                )
                display_options = [cluster_label_map.get(c, f"Cluster {c}") for c in cluster_options]
            else:
                cluster_label_map = {}
                display_options = [f"Cluster {c}" for c in cluster_options]

            selected_label = st.selectbox("Highlight Cluster", display_options)
            selected_idx = display_options.index(selected_label)
            selected_cluster = cluster_options[selected_idx]
            highlight = hotspot_df[hotspot_df["kmeans_cluster"] == selected_cluster]

            # Split into two dataframes so Plotly draws "Other" first (behind)
            bg_df = hotspot_df[hotspot_df["kmeans_cluster"] != selected_cluster].copy()
            fg_df = highlight.copy()

            vivid_color = CLUSTER_COLORS[int(selected_cluster) % len(CLUSTER_COLORS)]

            fig_hot = go.Figure()

            # Background: faint, tiny dots
            fig_hot.add_trace(go.Scattermapbox(
                lat=bg_df["latitude"],
                lon=bg_df["longitude"],
                mode="markers",
                marker=dict(size=3, color="#d0d0d0", opacity=0.25),
                name="Other",
                hoverinfo="skip",
            ))

            # Foreground: vivid, larger dots with hover
            fig_hot.add_trace(go.Scattermapbox(
                lat=fg_df["latitude"],
                lon=fg_df["longitude"],
                mode="markers",
                marker=dict(size=7, color=vivid_color, opacity=0.85),
                name=selected_label,
                text=fg_df.apply(lambda r: f"Shop: {r['shopNo']}<br>District: {r.get(dist_col,'')}<br>Transactions: {r.get('totalTransactions',0):,.0f}", axis=1),
                hoverinfo="text",
            ))

            fig_hot.update_layout(
                mapbox=dict(
                    style="carto-positron",
                    zoom=6,
                    center=dict(lat=hotspot_df["latitude"].median(), lon=hotspot_df["longitude"].median()),
                ),
                height=620,
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                            bgcolor="rgba(255,255,255,0.85)", font=dict(size=13)),
            )

            st.write(f"Showing **{len(highlight):,}** shops in **{selected_label}**")
            st.plotly_chart(fig_hot, use_container_width=True)

# ── TAB 5: Policy Impact ───────────────────────────────────────────────────
with tab_policy:
    st.subheader("Policy Impact Analysis: Portability Growth")

    if "year" not in unified_filtered.columns:
        st.info("Year column not available in transactions data.")
    else:
        trend = (
            unified_filtered.groupby("year", as_index=False)
            .agg(totalTransactions=("noOfTrans", "sum"), totalOtherShopTrans=("otherShopTrans", "sum"))
        )
        trend["otherShopShare"] = np.where(
            trend["totalTransactions"] > 0,
            trend["totalOtherShopTrans"] / trend["totalTransactions"],
            0,
        )

        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Scatter(
                x=trend["year"],
                y=trend["totalOtherShopTrans"],
                mode="lines+markers",
                name="Other Shop Transactions",
            )
        )
        fig_trend.add_trace(
            go.Scatter(
                x=trend["year"],
                y=trend["totalTransactions"],
                mode="lines+markers",
                name="Total Transactions",
            )
        )
        fig_trend.update_layout(height=450, title="ONORC/Portability Trend by Year")
        st.plotly_chart(fig_trend, use_container_width=True)

        fig_share = px.bar(
            trend,
            x="year",
            y="otherShopShare",
            title="Other Shop Share of Transactions",
            labels={"otherShopShare": "Portability Share"},
        )
        st.plotly_chart(fig_share, use_container_width=True)

        if "month" in unified_filtered.columns:
            monthly = (
                unified_filtered.dropna(subset=["month", "year"])
                .groupby(["year", "month"], as_index=False)["noOfTrans"]
                .sum()
            )
            fig_seasonality = px.line(
                monthly,
                x="month",
                y="noOfTrans",
                color="year",
                markers=True,
                title="Seasonality Check: Monthly Transaction Spikes",
            )
            st.plotly_chart(fig_seasonality, use_container_width=True)

# ── TAB 6: Risk & Logistics ────────────────────────────────────────────────
with tab_risk:
    st.subheader("Fraud Prevention Signals")
    if suspicious_shops is None or suspicious_shops.empty:
        st.info("No suspicious shops detected with current threshold.")
    else:
        cols = ["shopNo", dist_col, "clusterPersona", "kmeans_cluster", "meanTransactionToCardRatio", "clusterZScore"]
        cols = [c for c in cols if c in suspicious_shops.columns]
        st.dataframe(suspicious_shops[cols].head(100), use_container_width=True)

    st.subheader("Logistics Optimization: Portability Hubs")
    if portability_hubs is None or portability_hubs.empty:
        st.info("No portability hubs identified.")
    else:
        cols = ["shopNo", dist_col, "clusterPersona", "kmeans_cluster", "portabilityLoad", "totalOtherShopTrans"]
        cols = [c for c in cols if c in portability_hubs.columns]
        st.dataframe(portability_hubs[cols].head(100), use_container_width=True)
