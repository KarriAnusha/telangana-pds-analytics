"""Data pipeline and clustering helpers for Telangana PDS analytics."""

import os
import glob
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


TRANSACTION_RENAMES = {
    "nooftrans": "noOfTrans",
    "othershoptransactions": "otherShopTrans",
    "othershoptrans": "otherShopTrans",
    "othershoptranscnt": "otherShopTrans",
    "riceqtykg": "riceQty",
    "wheatqtykg": "wheatQty",
    "wheat": "wheatQty",
}

CARD_RENAMES = {
    "totalrcs": "totalRcs",
    "totalrationcards": "totalRcs",
}

LOCATION_RENAMES = {
    "lat": "latitude",
    "lng": "longitude",
    "long": "longitude",
    "fpsstatus": "shopStatus",
}

COMMON_RENAMES = {
    "shopno": "shopNo",
    "distcode": "distCode",
    "distname": "distName",
    "district": "distName",
}

DATE_COLUMN_CANDIDATES = [
    "txnDate",
    "transactionDate",
    "date",
    "monthYear",
    "yearMonth",
]

DATA_FOLDER_ALIASES = {
    "transactions": ["transactions", "transactions data", "transaction data"],
    "card_status": ["card_status", "card status", "card status data"],
    "fps_locations": ["fps_locations", "fps locations", "location data", "locations"],
}


def _normalize_column_names(df: pd.DataFrame, renames: Dict[str, str]) -> pd.DataFrame:
    """Normalize naming style and apply known column aliases."""
    out = df.copy()
    normalized = {
        c: c.strip().replace(" ", "").replace("_", "") if isinstance(c, str) else c
        for c in out.columns
    }
    out = out.rename(columns=normalized)

    lookup = {c.lower(): c for c in out.columns if isinstance(c, str)}
    mapped = {}
    for old_norm, new_name in {**COMMON_RENAMES, **renames}.items():
        if old_norm in lookup:
            mapped[lookup[old_norm]] = new_name
    return out.rename(columns=mapped)


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _ensure_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Infer year/month columns from available date columns when possible."""
    out = df.copy()

    date_col = next((c for c in DATE_COLUMN_CANDIDATES if c in out.columns), None)
    if date_col is not None:
        parsed = pd.to_datetime(out[date_col], errors="coerce")
        out["year"] = out.get("year", parsed.dt.year)
        out["month"] = out.get("month", parsed.dt.month)

    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    if "month" in out.columns:
        out["month"] = pd.to_numeric(out["month"], errors="coerce").astype("Int64")

    return out


def _normalize_name(text: str) -> str:
    return text.strip().replace("_", " ").replace("-", " ").lower()


def _resolve_data_folders(data_root: str) -> Dict[str, str]:
    """Resolve dataset folders even when names contain spaces or title case."""
    existing_dirs = [
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ]
    norm_map = {_normalize_name(d): d for d in existing_dirs}

    resolved = {}
    for dataset_name, aliases in DATA_FOLDER_ALIASES.items():
        found = None
        for alias in aliases:
            norm_alias = _normalize_name(alias)
            if norm_alias in norm_map:
                found = norm_map[norm_alias]
                break
        if found is None:
            raise FileNotFoundError(
                f"Could not find folder for '{dataset_name}' under {data_root}. "
                f"Looked for aliases: {aliases}"
            )
        resolved[dataset_name] = os.path.join(data_root, found)
    return resolved


def _harmonize_transaction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw portal transaction columns to model-ready canonical columns."""
    out = df.copy()

    rice_parts = [c for c in ["riceAfsc", "riceFsc", "riceAap"] if c in out.columns]
    if "riceQty" not in out.columns:
        if rice_parts:
            out[rice_parts] = out[rice_parts].apply(pd.to_numeric, errors="coerce").fillna(0)
            out["riceQty"] = out[rice_parts].sum(axis=1)
        else:
            out["riceQty"] = 0

    if "wheatQty" not in out.columns:
        out["wheatQty"] = pd.to_numeric(out.get("wheat", 0), errors="coerce").fillna(0)

    if "otherShopTrans" not in out.columns:
        out["otherShopTrans"] = pd.to_numeric(out.get("otherShopTransCnt", 0), errors="coerce").fillna(0)

    return out


# ---------------------------------------------------------------------------
# 1. Data Acquisition & Consolidation
# ---------------------------------------------------------------------------

def load_and_combine_csvs(folder_path: str) -> pd.DataFrame:
    """Read all CSV files in *folder_path* and concatenate into one DataFrame."""
    all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    dfs = [pd.read_csv(f) for f in all_files]
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def load_all_datasets(data_root: str = "data"):
    """Load transactions, card-status, and FPS-location CSVs and return
    three DataFrames."""
    folder_map = _resolve_data_folders(data_root)

    transactions = load_and_combine_csvs(folder_map["transactions"])
    card_status = load_and_combine_csvs(folder_map["card_status"])
    fps_locations = load_and_combine_csvs(folder_map["fps_locations"])

    transactions = _normalize_column_names(transactions, TRANSACTION_RENAMES)
    card_status = _normalize_column_names(card_status, CARD_RENAMES)
    fps_locations = _normalize_column_names(fps_locations, LOCATION_RENAMES)
    transactions = _harmonize_transaction_columns(transactions)

    transactions = _coerce_numeric(
        transactions,
        ["shopNo", "distCode", "noOfTrans", "otherShopTrans", "riceQty", "wheatQty", "year", "month"],
    )
    card_status = _coerce_numeric(card_status, ["shopNo", "distCode", "totalRcs"])
    fps_locations = _coerce_numeric(fps_locations, ["shopNo", "distCode", "latitude", "longitude"])

    transactions = _ensure_time_columns(transactions)

    return transactions, card_status, fps_locations


def create_unified_dataset(transactions: pd.DataFrame,
                           card_status: pd.DataFrame,
                           fps_locations: pd.DataFrame) -> pd.DataFrame:
    """Triple-Join on shopNo and distCode to create a single unified dataset."""
    for key in ["shopNo", "distCode"]:
        if key not in transactions.columns:
            raise KeyError(f"Missing required key in transactions: {key}")
        if key not in card_status.columns:
            raise KeyError(f"Missing required key in card_status: {key}")
        if key not in fps_locations.columns:
            raise KeyError(f"Missing required key in fps_locations: {key}")

    unified = transactions.merge(
        card_status,
        on=["shopNo", "distCode"],
        how="left",
        suffixes=("", "_card"),
    )
    unified = unified.merge(
        fps_locations,
        on=["shopNo", "distCode"],
        how="left",
        suffixes=("", "_loc"),
    )
    return unified


# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features required for clustering."""
    out = df.copy()

    required_numeric = ["noOfTrans", "otherShopTrans", "riceQty", "wheatQty", "totalRcs"]
    out = _coerce_numeric(out, required_numeric)
    for col in required_numeric:
        if col not in out.columns:
            out[col] = 0

    # Utilization ratio: transactions executed per eligible card base.
    out["utilizationRatio"] = np.where(out["totalRcs"] > 0, out["noOfTrans"] / out["totalRcs"], 0)

    # Commodity intensity feature: rice-to-wheat mix.
    wheat_safe = out["wheatQty"].replace(0, np.nan)
    out["riceWheatRatio"] = out["riceQty"] / wheat_safe
    out["riceWheatRatio"] = out["riceWheatRatio"].fillna(0)

    # Portability ratio: dependency on non-home shop transactions.
    out["portabilityRatio"] = np.where(
        out["noOfTrans"] > 0,
        out["otherShopTrans"] / out["noOfTrans"],
        0,
    )

    total_grain = out["riceQty"] + out["wheatQty"]
    out["riceShare"] = np.where(total_grain > 0, out["riceQty"] / total_grain, 0)
    out["wheatShare"] = np.where(total_grain > 0, out["wheatQty"] / total_grain, 0)

    # Fraud risk proxy requested in business use case section.
    out["transactionToCardRatio"] = np.where(
        out["totalRcs"] > 0,
        out["noOfTrans"] / out["totalRcs"],
        0,
    )

    num_cols = out.select_dtypes(include="number").columns
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return out


def compute_shop_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly data to shop-level and compute volatility features."""
    group_cols = ["shopNo", "distCode"]
    agg = df.groupby(group_cols).agg(
        totalTransactions=("noOfTrans", "sum"),
        meanTransactions=("noOfTrans", "mean"),
        medianTransactions=("noOfTrans", "median"),
        maxTransactions=("noOfTrans", "max"),
        stdTransactions=("noOfTrans", "std"),
        totalOtherShopTrans=("otherShopTrans", "sum"),
        meanUtilization=("utilizationRatio", "mean"),
        meanPortability=("portabilityRatio", "mean"),
        totalRice=("riceQty", "sum"),
        totalWheat=("wheatQty", "sum"),
        meanRiceWheatRatio=("riceWheatRatio", "mean"),
        meanTransactionToCardRatio=("transactionToCardRatio", "mean"),
        totalRcs=("totalRcs", "mean"),
    ).reset_index()

    agg["stdTransactions"] = agg["stdTransactions"].fillna(0)

    # Volatility coefficient captures temporal variability by shop.
    agg["volatilityCoeff"] = np.where(
        agg["meanTransactions"] > 0,
        agg["stdTransactions"] / agg["meanTransactions"],
        0,
    )

    agg["portabilityLoad"] = np.where(
        agg["totalTransactions"] > 0,
        agg["totalOtherShopTrans"] / agg["totalTransactions"],
        0,
    )

    if "year" in df.columns and "month" in df.columns:
        per_month = (
            df.groupby(["shopNo", "distCode", "year", "month"], dropna=False)["noOfTrans"]
            .sum()
            .reset_index()
        )
        season = per_month.groupby(["shopNo", "distCode"]).agg(
            seasonalPeakTransactions=("noOfTrans", "max"),
            seasonalMeanTransactions=("noOfTrans", "mean"),
            seasonalStdTransactions=("noOfTrans", "std"),
        ).reset_index()
        season["seasonalStdTransactions"] = season["seasonalStdTransactions"].fillna(0)
        season["seasonalPeakToMean"] = np.where(
            season["seasonalMeanTransactions"] > 0,
            season["seasonalPeakTransactions"] / season["seasonalMeanTransactions"],
            0,
        )
        agg = agg.merge(season, on=["shopNo", "distCode"], how="left")

    num_cols_agg = agg.select_dtypes(include="number").columns
    agg[num_cols_agg] = agg[num_cols_agg].replace([np.inf, -np.inf], np.nan).fillna(0)

    return agg


def attach_location_info(shop_features: pd.DataFrame,
                         fps_locations: pd.DataFrame) -> pd.DataFrame:
    """Merge latitude/longitude and other location fields back onto shop-level
    features for geospatial visualization."""
    loc_cols = ["shopNo", "distCode"]
    for c in ["latitude", "longitude", "lat", "lng", "shopName", "distName",
              "shopStatus", "mandalName"]:
        if c in fps_locations.columns:
            loc_cols.append(c)
    loc_dedup = fps_locations[loc_cols].drop_duplicates(subset=["shopNo", "distCode"])
    merged = shop_features.merge(loc_dedup, on=["shopNo", "distCode"], how="left")

    # Normalize lat/lng column names
    if "lat" in merged.columns and "latitude" not in merged.columns:
        merged.rename(columns={"lat": "latitude"}, inplace=True)
    if "lng" in merged.columns and "longitude" not in merged.columns:
        merged.rename(columns={"lng": "longitude"}, inplace=True)

    return merged


# ---------------------------------------------------------------------------
# 3. Clustering Pipeline
# ---------------------------------------------------------------------------

CLUSTER_FEATURES = [
    "meanTransactions",
    "medianTransactions",
    "maxTransactions",
    "stdTransactions",
    "totalOtherShopTrans",
    "meanUtilization",
    "meanPortability",
    "meanTransactionToCardRatio",
    "totalRice",
    "totalWheat",
    "meanRiceWheatRatio",
    "totalRcs",
    "volatilityCoeff",
    "portabilityLoad",
    "seasonalPeakToMean",
]


def scale_features(df: pd.DataFrame, features: list = None):
    """StandardScaler on selected features. Returns scaled array & scaler."""
    if features is None:
        features = CLUSTER_FEATURES
    features = [f for f in features if f in df.columns]
    if not features:
        raise ValueError("No clustering features available after preprocessing.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features].fillna(0))
    return X_scaled, scaler, features


def run_pca(X_scaled, n_components: int = 3):
    """Reduce dimensions with PCA."""
    n_components = min(n_components, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca


def find_optimal_k(X_scaled, k_range=range(2, 11)):
    """Return inertia and silhouette scores for the elbow / silhouette method."""
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        if len(np.unique(labels)) > 1:
            sil_scores.append(silhouette_score(X_scaled, labels))
        else:
            sil_scores.append(np.nan)
    return list(k_range), inertias, sil_scores


def run_kmeans(X_scaled, n_clusters: int = 5):
    """Fit K-Means and return labels + model."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km


def run_dbscan(X_scaled, eps: float = 1.5, min_samples: int = 5):
    """Fit DBSCAN and return labels + model.  Label -1 = noise/outlier."""
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    return labels, db


def build_cluster_profiles(shop_df: pd.DataFrame,
                           cluster_col: str = "kmeans_cluster",
                           features: list = None) -> pd.DataFrame:
    """Mean of all features per cluster — the Cluster Profile Report."""
    if features is None:
        features = CLUSTER_FEATURES
    features = [f for f in features if f in shop_df.columns]
    profile = shop_df.groupby(cluster_col)[features].mean()
    profile["shop_count"] = shop_df.groupby(cluster_col)[cluster_col].count()
    return profile


def compute_cluster_purity(
    shop_df: pd.DataFrame,
    cluster_col: str = "kmeans_cluster",
    label_col: str = "districtType",
) -> Optional[float]:
    """Compute cluster purity if a known class label column exists."""
    if cluster_col not in shop_df.columns or label_col not in shop_df.columns:
        return None
    tmp = shop_df[[cluster_col, label_col]].dropna()
    if tmp.empty:
        return None
    counts = tmp.groupby([cluster_col, label_col]).size().reset_index(name="n")
    best_per_cluster = counts.groupby(cluster_col)["n"].max().sum()
    return float(best_per_cluster / len(tmp))


def flag_suspicious_shops(
    shop_df: pd.DataFrame,
    ratio_col: str = "meanTransactionToCardRatio",
    cluster_col: str = "kmeans_cluster",
    z_threshold: float = 2.5,
) -> pd.DataFrame:
    """Flag shops deviating strongly from peers inside the same cluster."""
    if ratio_col not in shop_df.columns or cluster_col not in shop_df.columns:
        return pd.DataFrame(columns=list(shop_df.columns) + ["clusterZScore"])

    out = shop_df.copy()
    grp_mean = out.groupby(cluster_col)[ratio_col].transform("mean")
    grp_std = out.groupby(cluster_col)[ratio_col].transform("std").replace(0, np.nan)
    out["clusterZScore"] = ((out[ratio_col] - grp_mean) / grp_std).fillna(0)
    flagged = out[out["clusterZScore"].abs() >= z_threshold].copy()
    return flagged.sort_values("clusterZScore", ascending=False)


def identify_portability_hubs(
    shop_df: pd.DataFrame,
    portability_col: str = "portabilityLoad",
    quantile: float = 0.9,
) -> pd.DataFrame:
    """Identify top portability load shops useful for logistics planning."""
    if portability_col not in shop_df.columns:
        return pd.DataFrame(columns=shop_df.columns)
    cutoff = shop_df[portability_col].quantile(quantile)
    return shop_df[shop_df[portability_col] >= cutoff].sort_values(
        portability_col, ascending=False
    )


def pre_filter_extreme_outliers(
    shop_df: pd.DataFrame,
    ratio_col: str = "meanTransactionToCardRatio",
    threshold: float = 50.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split off shops with impossibly high utilization before clustering."""
    if ratio_col not in shop_df.columns:
        return shop_df.copy(), pd.DataFrame(columns=shop_df.columns)
    mask = shop_df[ratio_col] < threshold
    return shop_df[mask].copy().reset_index(drop=True), shop_df[~mask].copy()


def assign_cluster_personas(
    profile_df: pd.DataFrame,
    cluster_col: str = "kmeans_cluster",
) -> Dict[int, str]:
    """Dynamically assign readable persona names to clusters based on
    their mean feature values.  Works regardless of cluster-ID ordering."""
    personas = {}
    for cid in profile_df.index:
        row = profile_df.loc[cid]
        util = row.get("meanUtilization", 0)
        port = row.get("portabilityLoad", row.get("meanPortability", 0))
        vol = row.get("volatilityCoeff", 0)
        trans = row.get("meanTransactions", 0)
        count = row.get("shop_count", 0)

        if util > 2:
            label = "Anomalous High-Utilization"
        elif vol > 0.25 and port > 0.4:
            label = "Volatile Portability Hubs"
        elif trans > 900 and port > 0.6:
            label = "High-Volume Urban Hubs"
        elif port > 0.4:
            label = "Active Urban Shops"
        else:
            label = "Stable Rural Shops"
        personas[cid] = label
    return personas


# ---------------------------------------------------------------------------
# 4. Full Pipeline Helper
# ---------------------------------------------------------------------------

def run_full_pipeline(data_root: str = "data", n_clusters: int = 5,
                      dbscan_eps: float = 1.5, dbscan_min_samples: int = 5):
    """Execute the entire pipeline end-to-end and return key artefacts."""

    # Load
    transactions, card_status, fps_locations = load_all_datasets(data_root)

    # Unify
    unified = create_unified_dataset(transactions, card_status, fps_locations)

    # Feature engineering on unified (row-level)
    unified = engineer_features(unified)

    # Shop-level aggregation
    shop_features = compute_shop_level_features(unified)
    shop_features = attach_location_info(shop_features, fps_locations)

    # Pre-filter extreme outliers before distance-based clustering
    shop_features, pre_outliers = pre_filter_extreme_outliers(shop_features)

    # Scale
    X_scaled, scaler, active_features = scale_features(shop_features)

    # PCA
    X_pca, pca = run_pca(X_scaled, n_components=3)
    shop_features["pca1"] = X_pca[:, 0]
    shop_features["pca2"] = X_pca[:, 1]
    shop_features["pca3"] = X_pca[:, 2] if X_pca.shape[1] >= 3 else 0

    # K-Means
    km_labels, km_model = run_kmeans(X_scaled, n_clusters=n_clusters)
    shop_features["kmeans_cluster"] = km_labels

    # DBSCAN
    db_labels, db_model = run_dbscan(X_scaled, eps=dbscan_eps,
                                      min_samples=dbscan_min_samples)
    shop_features["dbscan_cluster"] = db_labels

    # Cluster profiles & persona labels
    cluster_profiles = build_cluster_profiles(shop_features, features=active_features)
    personas = assign_cluster_personas(cluster_profiles)
    shop_features["clusterPersona"] = shop_features["kmeans_cluster"].map(personas)
    cluster_profiles["persona"] = cluster_profiles.index.map(personas)

    # Tag pre-filtered extreme outliers
    if not pre_outliers.empty:
        pre_outliers["kmeans_cluster"] = -99
        pre_outliers["dbscan_cluster"] = -1
        pre_outliers["clusterPersona"] = "Extreme Anomaly (Pre-filtered)"
        for col in ["pca1", "pca2", "pca3"]:
            if col not in pre_outliers.columns:
                pre_outliers[col] = 0
        shop_features = pd.concat([shop_features, pre_outliers], ignore_index=True)

    # Business-use-case artefacts
    suspicious_shops = flag_suspicious_shops(shop_features)
    portability_hubs = identify_portability_hubs(shop_features)

    # Validation curves
    k_values, inertias, sil_scores = find_optimal_k(X_scaled)

    # Silhouette
    sil = silhouette_score(X_scaled, km_labels) if len(np.unique(km_labels)) > 1 else np.nan

    return {
        "unified": unified,
        "shop_features": shop_features,
        "cluster_profiles": cluster_profiles,
        "personas": personas,
        "scaler": scaler,
        "pca": pca,
        "kmeans_model": km_model,
        "dbscan_model": db_model,
        "silhouette_score": sil,
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_by_k": sil_scores,
        "active_features": active_features,
        "suspicious_shops": suspicious_shops,
        "portability_hubs": portability_hubs,
        "X_scaled": X_scaled,
        "X_pca": X_pca,
        "transactions": transactions,
        "card_status": card_status,
        "fps_locations": fps_locations,
    }
