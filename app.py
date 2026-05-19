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
    run_full_pipeline,
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

    # Run the same full pipeline used by scripts/run_pipeline.py so dashboard
    # results match exported reports and processed CSVs.
    artifacts = run_full_pipeline(DATA_ROOT, n_clusters=5, dbscan_eps=1.5, dbscan_min_samples=5)
    unified = artifacts["unified"]
    shop_features = artifacts["shop_features"]
    profiles = artifacts["cluster_profiles"]
    suspicious = artifacts["suspicious_shops"]
    hubs = artifacts["portability_hubs"]
    k_vals = artifacts["k_values"]
    inertias = artifacts["inertias"]
    sil_scores = artifacts["silhouette_by_k"]
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

# Compute clusterZScore for EVERY shop (not just the flagged ones).
# Formula: how many std-devs away is this shop's transaction-to-card ratio
# compared to other shops in the same K-Means cluster?
if "clusterZScore" not in shop_features.columns:
    _ratio = "meanTransactionToCardRatio"
    if _ratio in shop_features.columns and "kmeans_cluster" in shop_features.columns:
        _grp_mean = shop_features.groupby("kmeans_cluster")[_ratio].transform("mean")
        _grp_std  = shop_features.groupby("kmeans_cluster")[_ratio].transform("std").replace(0, np.nan)
        shop_features["clusterZScore"] = ((shop_features[_ratio] - _grp_mean) / _grp_std).fillna(0)
    else:
        shop_features["clusterZScore"] = 0.0

PERSONA_COLORS = {
    "Stable Rural Shops": "#2ca02c",
    "Active Urban Shops": "#1f77b4",
    "High-Volume Urban Hubs": "#ff7f0e",
    "Volatile Portability Hubs": "#d62728",
    "Anomalous High-Utilization": "#9467bd",
    "Extreme Anomaly": "#e377c2",
    "Extreme Anomaly (Pre-filtered)": "#e377c2",
}
CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Human-readable names for every model feature
FEATURE_LABELS = {
    "meanTransactions":            "Avg Monthly Transactions",
    "medianTransactions":          "Median Monthly Transactions",
    "maxTransactions":             "Peak Monthly Transactions",
    "stdTransactions":             "Transaction Variability",
    "totalOtherShopTrans":         "Total Out-of-Shop Transactions",
    "meanUtilization":             "Utilization Rate (Txn / Cards)",
    "meanPortability":             "Avg Portability Rate",
    "meanTransactionToCardRatio":  "Transactions per Registered Card",
    "totalRice":                   "Total Rice Distributed (kg)",
    "totalWheat":                  "Total Wheat Distributed (kg)",
    "meanRiceWheatRatio":          "Rice-to-Wheat Ratio",
    "totalRcs":                    "Registered Card Count",
    "volatilityCoeff":             "Month-to-Month Volatility",
    "portabilityLoad":             "Portability Load (Out-of-shop share)",
    "seasonalPeakToMean":          "Seasonal Peak vs Average",
    "clusterZScore":               "Anomaly Z-Score (vs cluster peers)",
}


# ── Sidebar filters ─────────────────────────────────────────────────────────
st.sidebar.title("Filters")

districts = sorted(shop_features[dist_col].dropna().unique())

# Initialise session state for district multiselect
if "district_multiselect" not in st.session_state:
    st.session_state["district_multiselect"] = districts

col_dist, col_all, col_clear = st.sidebar.columns([2, 1, 1])
with col_dist:
    st.markdown("**Select District(s)**")
with col_all:
    if st.button("All", key="select_all_districts"):
        st.session_state["district_multiselect"] = districts
with col_clear:
    if st.button("Clear", key="clear_districts"):
        st.session_state["district_multiselect"] = []

selected_districts = st.sidebar.multiselect(
    "Select District(s)",
    districts,
    key="district_multiselect",
    label_visibility="collapsed",
)

years = sorted(unified["year"].dropna().unique())
selected_years = st.sidebar.multiselect("Select Year(s)", years, default=years) if years else []

# Filter transactions by both district and year. Shop-level cluster features are
# built over the full available history, but the year filter still limits maps,
# search, risk tables, and KPIs to shops active in the selected transaction years.
if "year" in unified.columns and selected_years:
    unified_filtered = unified[unified["year"].isin(selected_years)]
else:
    unified_filtered = unified.copy()

if selected_districts and dist_col in unified_filtered.columns:
    unified_filtered = unified_filtered[unified_filtered[dist_col].isin(selected_districts)]
elif not selected_districts:
    unified_filtered = unified_filtered.iloc[0:0]

shop_filtered = shop_features[shop_features[dist_col].isin(selected_districts)].copy()
if not unified_filtered.empty and {"shopNo", "distCode"}.issubset(unified_filtered.columns):
    active_shop_keys = unified_filtered[["shopNo", "distCode"]].drop_duplicates()
    shop_filtered = shop_filtered.merge(
        active_shop_keys,
        on=["shopNo", "distCode"],
        how="inner",
    )
else:
    active_shop_keys = pd.DataFrame(columns=["shopNo", "distCode"])
    shop_filtered = shop_filtered.iloc[0:0]

if suspicious_shops is not None and not suspicious_shops.empty:
    suspicious_filtered = suspicious_shops[suspicious_shops[dist_col].isin(selected_districts)].copy()
    if not active_shop_keys.empty:
        suspicious_filtered = suspicious_filtered.merge(active_shop_keys, on=["shopNo", "distCode"], how="inner")
else:
    suspicious_filtered = suspicious_shops

if portability_hubs is not None and not portability_hubs.empty:
    hubs_filtered = portability_hubs[portability_hubs[dist_col].isin(selected_districts)].copy()
    if not active_shop_keys.empty:
        hubs_filtered = hubs_filtered.merge(active_shop_keys, on=["shopNo", "distCode"], how="inner")
else:
    hubs_filtered = portability_hubs


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
# ── Anomaly helpers ──────────────────────────────────────────────────────────

def explain_anomaly(shop_row, cluster_avg, z_score: float):
    """Return a list of plain-language red-flag strings for a given shop."""
    flags = []

    util = float(shop_row.get("meanUtilization", 0) or 0)
    if util > 1.5:
        flags.append(
            f"🔴 **Extremely high utilization ({util:.2f}×):** "
            f"This shop conducted {util:.1f}× more transactions than its registered card base. "
            "A ratio above 1.0 is impossible under normal operations and strongly suggests "
            "ghost card usage or data manipulation."
        )
    elif util > 1.0:
        flags.append(
            f"🟠 **Over-utilization ({util:.2f}×):** "
            f"Transactions slightly exceed the registered card count — "
            "could indicate cards being served that do not belong to this shop."
        )

    tcr  = float(shop_row.get("meanTransactionToCardRatio", 0) or 0)
    cavg = float(cluster_avg.get("meanTransactionToCardRatio", 1) or 1) if cluster_avg is not None else 1
    if cavg > 0 and tcr > cavg * 2:
        flags.append(
            f"🔴 **Transactions per card: {tcr:.2f}** (cluster peers average {cavg:.2f}). "
            f"This shop has **{tcr / cavg:.1f}× more transactions per card** than similar shops — "
            "a key fraud signal."
        )

    vol      = float(shop_row.get("volatilityCoeff", 0) or 0)
    cavg_vol = float(cluster_avg.get("volatilityCoeff", 0) or 0) if cluster_avg is not None else 0
    if vol > 0.5 and vol > cavg_vol * 1.5:
        flags.append(
            f"🟠 **High month-to-month volatility (coefficient: {vol:.2f}):** "
            "Monthly transaction counts swing wildly compared to peer shops. "
            "Legitimate PDS shops typically show stable, predictable volumes."
        )

    port = float(shop_row.get("portabilityLoad", 0) or 0)
    if port > 0.7:
        flags.append(
            f"🟡 **Very high portability load ({port * 100:.0f}% out-of-shop cards):** "
            "Most transactions here are from card-holders registered at other shops. "
            "While ONORC allows this, such extreme levels can mask diversion."
        )

    if abs(z_score) >= 2.5:
        flags.append(
            f"📊 **Statistical outlier within its cluster (Z-score: {z_score:.2f}):** "
            f"This shop's transaction-to-card ratio is **{abs(z_score):.1f} standard deviations** "
            "away from what peer shops in the same cluster show — that is unusual."
        )

    return flags


def compute_risk_reason(row) -> str:
    """Single-line reason string for the Risk & Logistics table."""
    reasons = []
    util = float(row.get("meanUtilization", 0) or 0)
    if util > 1.5:
        reasons.append(f"Extreme over-utilization ({util:.2f}×)")
    elif util > 1.0:
        reasons.append(f"Over-utilization ({util:.2f}×)")
    vol = float(row.get("volatilityCoeff", 0) or 0)
    if vol > 0.5:
        reasons.append(f"High volatility ({vol:.2f})")
    port = float(row.get("portabilityLoad", 0) or 0)
    if port > 0.7:
        reasons.append(f"Extreme portability load ({port * 100:.0f}%)")
    z = float(row.get("clusterZScore", 0) or 0)
    if abs(z) >= 3:
        reasons.append(f"Extreme outlier (Z={z:.1f})")
    elif abs(z) >= 2.5:
        reasons.append(f"Statistical outlier (Z={z:.1f})")
    return "; ".join(reasons) if reasons else "Elevated transaction-to-card ratio"


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
    with st.expander("ℹ️ What do the colours mean?", expanded=False):
        st.markdown("""
| Colour | Persona | What it means |
|--------|---------|---------------|
| 🟢 Green | **Stable Rural Shops** | Low-volume shops serving a fixed, predictable set of card-holders. Lowest risk. |
| 🔵 Blue | **Active Urban Shops** | Busier-than-average shops (higher number of monthly transactions) that also serve some beneficiaries who are registered at other shops (via ONORC portability). This is normal urban behaviour and not a fraud signal on its own. |
| 🟠 Orange | **High-Volume Urban Hubs** | Shops serving a very large number of beneficiaries, including many card-holders registered at other shops (ONORC). Not a fraud signal — but since stock is allocated based on a shop's *own* registered cards, these shops may receive less grain than they actually need. Monitoring ensures they get adequate supplies and no beneficiary is turned away. |
| 🔴 Red | **Volatile Portability Hubs** | Erratic month-to-month transactions and heavy out-of-shop card usage. Warrants closer audit. |
| 🟣 Purple | **Anomalous High-Utilization** | Transactions exceed registered card base — statistically impossible under clean data. Highest fraud risk. |
| 🩷 Pink | **Extreme Anomaly (Pre-filtered)** | Transaction-to-card ratio so extreme the shop was isolated before clustering. Immediate review recommended. |
        """)
        st.caption(
            "Map persona is risk-aware: shops with clusterZScore >= 2.5 display as "
            "Anomalous High-Utilization, and DBSCAN outliers display as Extreme Anomaly."
        )
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

            map_df["base_persona"] = map_df["persona_label"]
            map_df.loc[
                map_df["persona_label"] == "Extreme Anomaly (Pre-filtered)",
                "persona_label",
            ] = "Extreme Anomaly"
            if "clusterZScore" in map_df.columns:
                high_util_mask = map_df["clusterZScore"] >= 2.5
                map_df.loc[high_util_mask, "persona_label"] = "Anomalous High-Utilization"
            if "dbscan_cluster" in map_df.columns:
                dbscan_mask = map_df["dbscan_cluster"] == -1
                map_df.loc[dbscan_mask, "persona_label"] = "Extreme Anomaly"

            fig_map = px.scatter_mapbox(
                map_df,
                lat="latitude",
                lon="longitude",
                color="persona_label",
                color_discrete_map=PERSONA_COLORS if has_personas else None,
                hover_data={"shopNo": True, dist_col: True, "totalTransactions": ":,.0f",
                            "base_persona": True, "clusterZScore": ":.2f",
                            "dbscan_cluster": True, "latitude": False,
                            "longitude": False, "persona_label": False},
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
            # ── Map result summary ───────────────────────────────────────────
            persona_counts = map_df["persona_label"].value_counts().reset_index()
            persona_counts.columns = ["Persona", "Shops"]
            total_mapped = persona_counts["Shops"].sum()
            with st.expander("📊 What does this map tell us? (Summary)", expanded=False):
                st.markdown(f"**{total_mapped:,} shops** are plotted across the selected districts and years.")
                for _, prow in persona_counts.iterrows():
                    pct = prow["Shops"] / total_mapped * 100 if total_mapped > 0 else 0
                    st.markdown(f"- **{prow['Persona']}**: {prow['Shops']:,} shops ({pct:.1f}%)")
                zscore_count = int((map_df["clusterZScore"] >= 2.5).sum()) if "clusterZScore" in map_df.columns else 0
                dbscan_count = int((map_df["dbscan_cluster"] == -1).sum()) if "dbscan_cluster" in map_df.columns else 0
                st.markdown(f"- **Promoted to Anomalous High-Utilization by z-score**: {zscore_count:,} shops")
                st.markdown(f"- **Promoted to Extreme Anomaly by DBSCAN**: {dbscan_count:,} shops")
                high_risk_personas = {"Anomalous High-Utilization", "Extreme Anomaly", "Extreme Anomaly (Pre-filtered)", "Volatile Portability Hubs"}
                anomalous_count = int(persona_counts[persona_counts["Persona"].isin(high_risk_personas)]["Shops"].sum())
                if anomalous_count > 0:
                    st.warning(
                        f"**{anomalous_count:,} shops** ({anomalous_count / total_mapped * 100:.1f}%) fall into "
                        "high-risk personas (Red, Purple, or Pink). Purple and Pink include shop-level anomaly "
                        "signals from z-score and DBSCAN, not only K-Means cluster averages."
                    )
                else:
                    st.success(
                        "No high-risk persona shops detected in the current filter selection. "
                        "No high positive z-score or DBSCAN outlier shops are present for this filter."
                    )

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

            # ── Z-Score: read directly from shop_features (available for all shops) ──
            z_score = float(shop_row.get("clusterZScore", 0) or 0)

            # ── Show Z-Score + key signals as headline metrics ───────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Anomaly Z-Score",
                f"{z_score:.2f}",
                help="How many standard deviations this shop's transaction-to-card ratio "
                     "is above its cluster average. Above 2.5 = flagged. Above 3 = high risk.",
            )
            m2.metric(
                "Utilization Rate",
                f"{float(shop_row.get('meanUtilization', 0)):.2f}×",
                help="Transactions ÷ Registered Cards. Above 1.0 is suspicious; above 1.5 is a strong fraud signal.",
            )
            m3.metric(
                "Month-to-Month Volatility",
                f"{float(shop_row.get('volatilityCoeff', 0)):.2f}",
                help="How erratically monthly transactions swing (std ÷ mean). Above 0.5 is high.",
            )
            m4.metric(
                "Portability Load",
                f"{float(shop_row.get('portabilityLoad', 0)) * 100:.0f}%",
                help="Share of transactions from card-holders registered at other shops.",
            )

            with st.expander("📖 What do these numbers mean? (click to expand)"):
                st.markdown("""
**Anomaly Z-Score**
: Compares this shop's transaction-to-card ratio against other shops *in the same cluster* (peer shops of similar size and location).
  A score of **0** means the shop is exactly average for its group.
  A score of **+2.5 or above** means it is unusually high compared to peers — flagged for investigation.
  A **negative** score means it is below its cluster's average (but if the whole cluster is anomalous, that still doesn't mean the shop is clean — check Utilization Rate).

---

**Utilization Rate**
: Number of transactions recorded ÷ number of ration cards registered at this shop.
  - **Below 1.0×** → Normal. The shop is serving fewer people than cards on record (some beneficiaries may not have collected that month).
  - **Exactly 1.0×** → Every registered card collected rations — perfect utilization.
  - **Above 1.0×** → More transactions than registered cards. This is physically impossible under honest operations and is a red flag.
  - **Above 1.5×** → Strong fraud signal. Likely ghost cards or fabricated transactions.
  - **Above 2.0×** → Immediate audit recommended.

---

**Month-to-Month Volatility**
: Measures how much a shop's monthly transaction count fluctuates over time (calculated as: standard deviation ÷ average).
  - **0.0 – 0.1** → Very stable. Nearly the same volume every month. ✅
  - **0.1 – 0.3** → Mild, normal seasonal variation. ✅
  - **0.3 – 0.5** → Noticeable swings. Worth monitoring. 🟡
  - **Above 0.5** → Erratic. Large spikes and crashes month to month. 🔴
  
  *A high utilization rate combined with very low volatility (like 0.02) is especially suspicious — it suggests the same inflated number is being entered every month deliberately.*

---

**Portability Load**
: Percentage of transactions at this shop from beneficiaries who are **registered at a different shop** (allowed under the ONORC / One Nation One Ration Card scheme).
  - **0–20%** → Mostly serves its own registered card-holders. Normal for rural shops.
  - **20–60%** → Moderate. Common in urban areas where people move around.
  - **Above 70%** → Very high. The shop is primarily serving outsiders. Not necessarily fraud, but combined with other signals it can mask diversion of grain.
                """)

            flags = explain_anomaly(shop_row, cluster_avg, z_score) if cluster_avg is not None else []

            # Warn if shop is in an anomalous cluster but Z-score looks low
            anomalous_personas = {"Anomalous High-Utilization", "Extreme Anomaly (Pre-filtered)"}
            in_anomalous_cluster = str(persona) in anomalous_personas

            if flags:
                st.error("#### ⚠️ Why is this shop flagged as anomalous?")
                st.markdown(
                    "The system compares each shop's behaviour against other shops in the same "
                    "cluster (shops with similar size and location profile). "
                    "The following signals were detected:"
                )
                for flag in flags:
                    st.markdown(f"- {flag}")
                if in_anomalous_cluster and z_score <= 0:
                    st.warning(
                        "🔍 **Why is the Z-Score negative for a flagged shop?** "
                        "The Z-score measures how unusual this shop is *compared to its own cluster*. "
                        f"This shop is in the **'{persona}'** cluster — meaning *all* shops here already "
                        "have extremely high utilization rates. A negative Z-score just means this shop "
                        "is slightly below the (already extreme) cluster average. "
                        "**The Utilization Rate above is the real fraud signal, not the Z-score.**"
                    )
                st.markdown(
                    "> **What should an investigator do?** Visit the shop, cross-check "
                    "the beneficiary register with actual card holders, and audit "
                    "monthly stock movement records."
                )
            elif in_anomalous_cluster:
                st.warning(
                    f"⚠️ This shop belongs to the **'{persona}'** cluster. "
                    "Even though individual ratio flags weren't triggered, every shop in this cluster "
                    "has an abnormally high utilization rate. Check the Utilization Rate metric above — "
                    "a value above 1.0× is suspicious regardless of Z-score."
                )
            else:
                st.success("✅ No strong anomaly signals detected for this shop. Behaviour is consistent with its cluster peers.")

            st.divider()

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Shop Values**")
                display_vals = shop_row[compare_features].rename(FEATURE_LABELS)
                st.dataframe(display_vals.to_frame("Value"), use_container_width=True)

            with col_b:
                if cluster_avg is not None:
                    st.markdown(f"**Cluster {cluster_id} Average (peer shops)**")
                    avg_vals = cluster_avg[compare_features].rename(FEATURE_LABELS)
                    st.dataframe(avg_vals.to_frame("Cluster Avg"), use_container_width=True)

            if cluster_avg is not None:
                # Split features into large-scale (totals) and small-scale (ratios/coefficients)
                ratio_features = [
                    "meanUtilization", "meanPortability", "meanTransactionToCardRatio",
                    "meanRiceWheatRatio", "volatilityCoeff", "portabilityLoad", "seasonalPeakToMean",
                ]
                # Grain features (kg — scale of 100,000s) must be on their own axis
                grain_features = [f for f in compare_features if f in ("totalRice", "totalWheat")]
                # Count features (transactions, cards — scale of 100s–1,000s)
                count_features = [
                    f for f in compare_features
                    if f not in ratio_features and f not in grain_features
                ]
                small_features = [f for f in compare_features if f in ratio_features]

                count_labels = [FEATURE_LABELS.get(f, f) for f in count_features]
                grain_labels  = [FEATURE_LABELS.get(f, f) for f in grain_features]
                small_labels  = [FEATURE_LABELS.get(f, f) for f in small_features]

                # ── Chart 1: Transaction & Card Counts ───────────────────
                st.markdown("#### Transaction & Card Counts")
                st.caption(
                    "How many transactions and registered cards this shop has vs its cluster average. "
                    "These are raw counts (number of people served per month, total cards, etc.)."
                )
                fig1 = go.Figure()
                fig1.add_trace(go.Bar(
                    name=f"Shop {shop_id}",
                    x=count_labels,
                    y=shop_row[count_features].values.astype(float),
                ))
                fig1.add_trace(go.Bar(
                    name=f"Cluster {cluster_id} Avg",
                    x=count_labels,
                    y=cluster_avg[count_features].values.astype(float),
                ))
                fig1.update_layout(
                    barmode="group", xaxis_tickangle=-35, height=380,
                    yaxis_title="Count",
                )
                st.plotly_chart(fig1, use_container_width=True)
                # ── Chart 1 result summary ──────────────────────────────────
                mean_txn_shop  = float(shop_row.get("meanTransactions", 0) or 0)
                mean_txn_clust = float(cluster_avg.get("meanTransactions", 1) or 1)
                if mean_txn_clust > 0:
                    txn_ratio = mean_txn_shop / mean_txn_clust
                    if txn_ratio > 1.5:
                        txn_verdict = f"⬆️ **{txn_ratio:.1f}×** the cluster average — notably busier than peer shops."
                    elif txn_ratio < 0.6:
                        txn_verdict = f"⬇️ Only **{txn_ratio:.1f}×** the cluster average — significantly quieter than peer shops."
                    else:
                        txn_verdict = f"✅ **Broadly in line** with cluster peers ({txn_ratio:.1f}× the average)."
                    st.info(f"**Result:** Monthly transaction volume is {txn_verdict}")

                # ── Chart 2: Grain Distributed (kg) ──────────────────────
                if grain_features:
                    st.markdown("#### Grain Distributed (kg)")
                    st.caption(
                        "Total kilograms of rice and wheat distributed by this shop over all recorded months. "
                        "These are much larger numbers than transaction counts because each transaction "
                        "distributes several kilograms of grain per family member."
                    )
                    fig_grain = go.Figure()
                    fig_grain.add_trace(go.Bar(
                        name=f"Shop {shop_id}",
                        x=grain_labels,
                        y=shop_row[grain_features].values.astype(float),
                    ))
                    fig_grain.add_trace(go.Bar(
                        name=f"Cluster {cluster_id} Avg",
                        x=grain_labels,
                        y=cluster_avg[grain_features].values.astype(float),
                    ))
                    fig_grain.update_layout(
                        barmode="group", xaxis_tickangle=-35, height=320,
                        yaxis_title="Kilograms (kg)",
                    )
                    st.plotly_chart(fig_grain, use_container_width=True)
                    st.caption(
                        "💡 **Why is rice so much more than wheat?** "
                        "Telangana primarily distributes rice under PDS. Wheat allocation is much smaller "
                        "and not available in all districts."
                    )
                    # ── Grain chart result summary ──────────────────────────
                    if "totalRice" in shop_row.index and "totalRice" in cluster_avg.index:
                        rice_shop  = float(shop_row.get("totalRice", 0) or 0)
                        rice_clust = float(cluster_avg.get("totalRice", 1) or 1)
                        if rice_clust > 0:
                            rice_ratio = rice_shop / rice_clust
                            if rice_ratio > 1.3:
                                grain_verdict = f"⬆️ This shop distributed **{rice_ratio:.1f}×** more rice than cluster peers — either it served more beneficiaries or distribution records are inflated."
                            elif rice_ratio < 0.7:
                                grain_verdict = f"⬇️ Only **{rice_ratio:.1f}×** the cluster's average rice — fewer beneficiaries served or several months with no distribution."
                            else:
                                grain_verdict = f"✅ Rice distributed is **broadly in line** with cluster peers ({rice_ratio:.1f}× cluster average)."
                            st.info(f"**Result:** {grain_verdict}")

                # ── Radar chart ──────────────────────────────────────────
                st.markdown("#### Risk Radar — Shop vs Cluster Average")
                st.caption(
                    "Each spoke is a normalised risk metric. "
                    "A larger red area than blue means this shop behaves unusually."
                )
                radar_features = [
                    "meanUtilization", "volatilityCoeff", "portabilityLoad",
                    "meanTransactionToCardRatio", "seasonalPeakToMean", "clusterZScore",
                ]
                radar_features = [f for f in radar_features if f in shop_row.index]
                radar_labels   = [FEATURE_LABELS.get(f, f) for f in radar_features]

                shop_vals_r = shop_row[radar_features].values.astype(float)
                # Cluster average Z-score is 0 by definition (it is the reference)
                avg_vals_r = np.array([
                    float(cluster_avg[f]) if f in cluster_avg.index else 0.0
                    for f in radar_features
                ])

                # Normalise to 0-1 scale per feature using cluster avg as reference
                max_vals = np.maximum(np.abs(shop_vals_r), np.abs(avg_vals_r))
                max_vals[max_vals == 0] = 1
                shop_norm = shop_vals_r / max_vals
                avg_norm  = avg_vals_r  / max_vals

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=list(shop_norm) + [shop_norm[0]],
                    theta=radar_labels + [radar_labels[0]],
                    fill="toself",
                    name=f"Shop {shop_id}",
                    line_color="#d62728",
                    fillcolor="rgba(214,39,40,0.25)",
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=list(avg_norm) + [avg_norm[0]],
                    theta=radar_labels + [radar_labels[0]],
                    fill="toself",
                    name=f"Cluster {cluster_id} Avg",
                    line_color="#1f77b4",
                    fillcolor="rgba(31,119,180,0.25)",
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    height=420,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.15),
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                # ── Radar chart result summary ──────────────────────────────
                above_spokes = [
                    FEATURE_LABELS.get(f, f)
                    for f, s_n, a_n in zip(radar_features, shop_norm, avg_norm)
                    if s_n > a_n * 1.2 and f != "clusterZScore"
                ]
                if above_spokes:
                    st.warning(
                        f"**Radar Summary:** This shop's red shape extends beyond the blue cluster average on "
                        f"**{len(above_spokes)} metric(s):** {', '.join(above_spokes)}. "
                        "These are the dimensions where it stands out most from peer shops."
                    )
                else:
                    st.success(
                        "**Radar Summary:** This shop's risk profile is within or below its cluster average "
                        "on all measured dimensions — no spokes significantly exceed the blue reference area."
                    )

# ── TAB 3: Cluster Profile Report ───────────────────────────────────────────
with tab_profile:
    st.subheader("K-Means Cluster Profile Report")
    if cluster_profiles is not None:
        numeric_cols = cluster_profiles.select_dtypes(include="number").columns
        fmt = {c: "{:.2f}" for c in numeric_cols}
        st.dataframe(cluster_profiles.style.format(fmt), use_container_width=True)
        # ── Cluster table result summary ─────────────────────────────────
        with st.expander("📊 Key findings from this table (click to expand)", expanded=True):
            findings = []
            if "meanUtilization" in cluster_profiles.columns:
                hi_c = int(cluster_profiles["meanUtilization"].idxmax())
                hi_v = float(cluster_profiles["meanUtilization"].max())
                lo_c = int(cluster_profiles["meanUtilization"].idxmin())
                lo_v = float(cluster_profiles["meanUtilization"].min())
                findings.append(f"- **Cluster {hi_c}** has the highest average utilization rate (**{hi_v:.2f}×**). Shops here conduct more transactions than their registered card base — highest fraud risk.")
                findings.append(f"- **Cluster {lo_c}** has the lowest utilization rate (**{lo_v:.2f}×**) — the most compliant group.")
            if "volatilityCoeff" in cluster_profiles.columns:
                mv_c = int(cluster_profiles["volatilityCoeff"].idxmax())
                mv_v = float(cluster_profiles["volatilityCoeff"].max())
                findings.append(f"- **Cluster {mv_c}** is the most volatile (coefficient: **{mv_v:.2f}**) — monthly transactions fluctuate more than any other cluster.")
            if "portabilityLoad" in cluster_profiles.columns:
                mp_c = int(cluster_profiles["portabilityLoad"].idxmax())
                mp_v = float(cluster_profiles["portabilityLoad"].max())
                findings.append(f"- **Cluster {mp_c}** carries the highest portability load (**{mp_v*100:.0f}%** out-of-shop cards) — heavily reliant on ONORC beneficiaries.")
            for f_line in findings:
                st.markdown(f_line)

        profile_features = [f for f in CLUSTER_FEATURES if f in cluster_profiles.columns]
        idx_col = cluster_profiles.index.name or "kmeans_cluster"

        # Split into grain-volume features (large scale) vs behavioural features (small scale)
        VOLUME_FEATURES = [f for f in ("totalRice", "totalWheat") if f in profile_features]
        other_features   = [f for f in profile_features if f not in VOLUME_FEATURES]

        # Chart 1 — Grain Volumes (kg)
        if VOLUME_FEATURES:
            st.markdown("#### Grain Distributed per Cluster (kg)")
            st.caption(
                "Total kilograms of rice and wheat distributed, averaged per shop in each cluster. "
                "Higher bars = larger shops or clusters serving more beneficiaries."
            )
            fig_vol = px.bar(
                cluster_profiles.reset_index().melt(
                    id_vars=idx_col,
                    value_vars=VOLUME_FEATURES,
                ),
                x=idx_col,
                y="value",
                color="variable",
                color_discrete_map={
                    "totalRice":  "#e07b54",
                    "totalWheat": "#f0c040",
                },
                barmode="group",
                labels={"value": "Total Grain (kg)", idx_col: "Cluster", "variable": "Grain Type"},
                height=400,
            )
            fig_vol.update_layout(legend_title_text="Grain Type")
            st.plotly_chart(fig_vol, use_container_width=True)
            # ── Grain chart result summary ────────────────────────────────
            if "totalRice" in cluster_profiles.columns and "totalWheat" in cluster_profiles.columns:
                top_rice_c = int(cluster_profiles["totalRice"].idxmax())
                top_rice_v = float(cluster_profiles["totalRice"].max())
                wheat_total = float(cluster_profiles["totalWheat"].sum())
                rice_to_wheat = float(cluster_profiles["totalRice"].sum()) / wheat_total if wheat_total > 0 else float("inf")
                st.info(
                    f"**Result:** Cluster {top_rice_c} distributes the most rice on average "
                    f"(**{top_rice_v:,.0f} kg** per shop). "
                    f"Across all clusters combined, rice accounts for roughly **{rice_to_wheat:.0f}×** "
                    "the volume of wheat, consistent with Telangana’s rice-primary PDS allocation policy."
                )

        # Chart 2 — Behavioural & Transaction Metrics
        if other_features:
            st.markdown("#### Behavioural & Transaction Metrics per Cluster")
            st.caption(
                "All non-volume metrics: transaction counts, utilization rate, volatility, portability, etc. "
                "Clusters with high utilization or volatility bars warrant closer scrutiny."
            )
            fig_beh = px.bar(
                cluster_profiles.reset_index().melt(
                    id_vars=idx_col,
                    value_vars=other_features,
                ),
                x=idx_col,
                y="value",
                color="variable",
                barmode="group",
                labels={"value": "Value", idx_col: "Cluster", "variable": "Metric"},
                height=500,
            )
            fig_beh.update_layout(legend_title_text="Metric")
            st.plotly_chart(fig_beh, use_container_width=True)
            # ── Behavioural chart result summary ───────────────────────────
            beh_lines = []
            if "meanUtilization" in cluster_profiles.columns:
                hi_u_c = int(cluster_profiles["meanUtilization"].idxmax())
                hi_u_v = float(cluster_profiles["meanUtilization"].max())
                beh_lines.append(f"**Cluster {hi_u_c}** stands out with the highest utilization rate ({hi_u_v:.2f}×).")
            if "volatilityCoeff" in cluster_profiles.columns:
                hi_vc = int(cluster_profiles["volatilityCoeff"].idxmax())
                hi_vv = float(cluster_profiles["volatilityCoeff"].max())
                beh_lines.append(f"**Cluster {hi_vc}** is the most erratic month-to-month (volatility coefficient: {hi_vv:.2f}).")
            if "portabilityLoad" in cluster_profiles.columns:
                hi_pc = int(cluster_profiles["portabilityLoad"].idxmax())
                hi_pv = float(cluster_profiles["portabilityLoad"].max())
                beh_lines.append(f"**Cluster {hi_pc}** has the heaviest portability load ({hi_pv*100:.0f}% out-of-shop beneficiaries).")
            if beh_lines:
                st.info("**Result:** " + "  \n".join(beh_lines))

    st.subheader("DBSCAN Outlier Summary")
    if "dbscan_cluster" in shop_features.columns:
        outliers = shop_filtered[shop_filtered["dbscan_cluster"] == -1]
        st.write(f"**{len(outliers)}** shops flagged as outliers by DBSCAN")
        if not outliers.empty:
            st.dataframe(
                outliers[["shopNo", dist_col] + CLUSTER_FEATURES].head(50),
                use_container_width=True,
            )
            # ── DBSCAN result summary ────────────────────────────────────
            st.info(
                f"**Result:** DBSCAN identified **{len(outliers)} shops** that do not fit into any cluster — "
                "their behaviour is so unusual that the algorithm could not group them with any peer set. "
                "These shops warrant immediate individual investigation, as their activity cannot be explained "
                "by any normal cluster pattern."
            )

    st.subheader("Clustering Diagnostics")
    st.caption(
        "These charts compare candidate K-Means cluster counts. This dashboard uses "
        "k=5 for business-friendly personas, while the silhouette chart shows whether "
        "a smaller or larger k is mathematically cleaner."
    )
    if diagnostics is not None and not diagnostics.empty:
        left, right = st.columns(2)
        with left:
            fig_elbow = px.line(
                diagnostics,
                x="k",
                y="inertia",
                markers=True,
                title="Elbow Curve (K vs Inertia)",
                labels={
                    "k": "Number of Clusters (k)  →  more groups",
                    "inertia": "Inertia  →  lower = tighter clusters",
                },
            )
            fig_elbow.add_vline(
                x=5, line_dash="dash", line_color="red",
                annotation_text="Dashboard k=5", annotation_position="top right",
            )
            st.plotly_chart(fig_elbow, use_container_width=True)
            st.caption(
                "**X-axis (k):** how many groups we split shops into.  \n"
                "**Y-axis (Inertia):** total spread inside clusters — lower is better.  \n"
                "Use the elbow curve to judge whether adding more clusters gives meaningfully "
                "tighter groups. k=5 is retained here because it gives interpretable shop personas."
            )
        with right:
            fig_sil = px.line(
                diagnostics,
                x="k",
                y="silhouette",
                markers=True,
                title="Silhouette Score by K",
                labels={
                    "k": "Number of Clusters (k)  →  more groups",
                    "silhouette": "Silhouette Score  →  higher = better separation",
                },
            )
            fig_sil.add_vline(
                x=5, line_dash="dash", line_color="red",
                annotation_text="Dashboard k=5", annotation_position="top right",
            )
            st.plotly_chart(fig_sil, use_container_width=True)
            st.caption(
                "**X-axis (k):** number of clusters.  \n"
                "**Y-axis (Silhouette Score, 0–1):** how well each shop fits its own "
                "cluster vs. the nearest other cluster. Higher = cleaner separation.  \n"
                "k=5 is a presentation choice for richer operational profiling, not necessarily the top silhouette score."
            )

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
            # ── Hotspot result summary ───────────────────────────────────
            n_total_mapped = len(hotspot_df)
            pct_highlighted = len(highlight) / n_total_mapped * 100 if n_total_mapped > 0 else 0
            if len(highlight) > 0 and dist_col in highlight.columns:
                top_dist = highlight[dist_col].value_counts().idxmax()
                top_dist_count = int(highlight[dist_col].value_counts().iloc[0])
                st.info(
                    f"**Result:** The **{selected_label}** cluster makes up **{pct_highlighted:.1f}%** "
                    f"of all mapped shops ({len(highlight):,} of {n_total_mapped:,}). "
                    f"The highest concentration is in **{top_dist}** ({top_dist_count:,} shops). "
                    "Geographic clustering of high-risk shops in the same district may indicate "
                    "a systemic issue or shared supply chain manipulation."
                )

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
        # ── ONORC trend result summary ───────────────────────────────
        if len(trend) >= 2:
            first_yr = trend.iloc[0]
            last_yr  = trend.iloc[-1]
            onorc_chg = last_yr["totalOtherShopTrans"] - first_yr["totalOtherShopTrans"]
            if onorc_chg > 0:
                st.info(
                    f"**Result:** ONORC/portability transactions grew from "
                    f"**{first_yr['totalOtherShopTrans']:,.0f}** ({int(first_yr['year'])}) to "
                    f"**{last_yr['totalOtherShopTrans']:,.0f}** ({int(last_yr['year'])}), "
                    f"an increase of **{onorc_chg:,.0f} transactions**. "
                    "This reflects growing adoption of the One Nation One Ration Card scheme, "
                    "allowing beneficiaries to collect grain from any shop across India."
                )
            else:
                st.info(
                    f"**Result:** Portability transactions declined over the observed period "
                    f"({int(first_yr['year'])} → {int(last_yr['year'])}). "
                    "This may reflect improved local access or reduced migration in the selected districts."
                )

        fig_share = px.bar(
            trend,
            x="year",
            y="otherShopShare",
            title="Other Shop Share of Transactions",
            labels={"otherShopShare": "Portability Share"},
        )
        st.plotly_chart(fig_share, use_container_width=True)
        # ── Portability share result summary ───────────────────────────
        avg_share = float(trend["otherShopShare"].mean())
        max_share_row = trend.loc[trend["otherShopShare"].idxmax()]
        st.info(
            f"**Result:** On average, **{avg_share*100:.1f}%** of all transactions were from "
            "out-of-shop beneficiaries (ONORC portability). "
            f"The highest share was in **{int(max_share_row['year'])}** "
            f"({max_share_row['otherShopShare']*100:.1f}%). "
            "A rising share indicates the ONORC policy is working — more beneficiaries are "
            "collecting from shops outside their registered home shop."
        )

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
            # ── Seasonality result summary ────────────────────────────────
            peak_row = monthly.loc[monthly["noOfTrans"].idxmax()]
            low_row  = monthly.loc[monthly["noOfTrans"].idxmin()]
            st.info(
                f"**Result:** The highest transaction volume across all years occurs in "
                f"**Month {int(peak_row['month'])}** ({peak_row['noOfTrans']:,.0f} transactions), "
                f"and the lowest in **Month {int(low_row['month'])}** ({low_row['noOfTrans']:,.0f}). "
                "Seasonal peaks around harvest or festive months are expected for PDS. "
                "However, a very sharp spike at a *single shop* out of its normal cycle is a fraud signal."
            )

# ── TAB 6: Risk & Logistics ────────────────────────────────────────────────
with tab_risk:
    st.subheader("Fraud Prevention Signals")
    st.markdown(
        "Shops below were flagged because their **Transactions per Registered Card** ratio "
        "is significantly higher than other shops in the same cluster. "
        "The **Reason for Flagging** column explains in plain language what was unusual."
    )
    if suspicious_filtered is None or suspicious_filtered.empty:
        st.info("No suspicious shops detected with current threshold.")
    else:
        display_susp = suspicious_filtered.copy()
        display_susp["Reason for Flagging"] = display_susp.apply(compute_risk_reason, axis=1)
        cols = [
            "shopNo", dist_col, "clusterPersona", "kmeans_cluster",
            "meanTransactionToCardRatio", "meanUtilization",
            "volatilityCoeff", "portabilityLoad", "clusterZScore",
            "Reason for Flagging",
        ]
        cols = [c for c in cols if c in display_susp.columns]
        rename_map = {k: FEATURE_LABELS.get(k, k) for k in cols}
        rename_map.update({
            "shopNo": "Shop No",
            dist_col: "District",
            "clusterPersona": "Cluster Persona",
            "kmeans_cluster": "Cluster ID",
            "Reason for Flagging": "Reason for Flagging",
        })
        st.dataframe(
            display_susp[cols].rename(columns=rename_map).head(100),
            use_container_width=True,
        )
        st.caption(
            "**How to read this table:** Z-Score shows how many standard deviations "
            "the shop's transaction-to-card ratio is above its cluster average. "
            "A score above 2.5 is flagged. Higher = more unusual."
        )
        # ── Fraud table result summary ──────────────────────────────────
        n_flagged = len(suspicious_filtered)
        n_total_shops = shop_filtered["shopNo"].nunique()
        n_extreme = int((suspicious_filtered["meanUtilization"] > 1.5).sum()) if "meanUtilization" in suspicious_filtered.columns else 0
        flag_pct = n_flagged / n_total_shops * 100 if n_total_shops > 0 else 0
        st.warning(
            f"**Result:** **{n_flagged:,} shops** ({flag_pct:.1f}% of all shops) were flagged as suspicious. "
            + (f"Of these, **{n_extreme}** have an extreme utilization rate above 1.5× — "
               "these are the highest-priority cases for immediate field audit."
               if n_extreme > 0 else
               "Use the Shop Search tab to investigate individual shops in detail.")
        )

    st.subheader("Logistics Optimization: Portability Hubs")
    if hubs_filtered is None or hubs_filtered.empty:
        st.info("No portability hubs identified.")
    else:
        cols = ["shopNo", dist_col, "clusterPersona", "kmeans_cluster", "portabilityLoad", "totalOtherShopTrans"]
        cols = [c for c in cols if c in hubs_filtered.columns]
        st.dataframe(hubs_filtered[cols].head(100), use_container_width=True)
        # ── Portability hubs result summary ─────────────────────────────
        n_hubs = len(hubs_filtered)
        if "portabilityLoad" in hubs_filtered.columns:
            avg_hub_load = float(hubs_filtered["portabilityLoad"].mean())
            top_hub_dist = hubs_filtered[dist_col].value_counts().idxmax() if dist_col in hubs_filtered.columns else "N/A"
            st.info(
                f"**Result:** **{n_hubs:,} shops** qualify as portability hubs, each serving a high share of "
                f"out-of-shop card-holders (average load: **{avg_hub_load*100:.0f}%**). "
                f"The most hub-dense district is **{top_hub_dist}**. "
                "These shops are critical logistics nodes — stock allocation must account for their "
                "higher-than-registered demand to prevent beneficiaries being turned away."
            )
        else:
            st.info(
                f"**Result:** **{n_hubs:,} shops** qualify as portability hubs. "
                "These are key logistics nodes that need higher stock allocation to serve "
                "beneficiaries arriving from other shops."
            )
