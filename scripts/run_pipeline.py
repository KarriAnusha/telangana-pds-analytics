"""Run full Telangana PDS analytics pipeline and export project deliverables."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_processing import compute_cluster_purity, run_full_pipeline


DATA_ROOT = ROOT / "data"
PROCESSED_DIR = DATA_ROOT / "processed"
REPORT_PATH = ROOT / "reports" / "cluster_profile_report.md"


def _best_k_row(diagnostics: pd.DataFrame) -> pd.Series | None:
    if diagnostics.empty:
        return None
    valid = diagnostics.dropna(subset=["silhouette"])
    if valid.empty:
        return None
    return valid.sort_values("silhouette", ascending=False).iloc[0]


def _to_markdown_table(df: pd.DataFrame, index: bool = True) -> str:
    """Render a small DataFrame as Markdown without optional tabulate dependency."""
    out = df.copy()
    if index:
        index_name = out.index.name or ""
        out = out.reset_index().rename(columns={out.index.name or "index": index_name})
    out = out.fillna("")
    headers = [str(c) for c in out.columns]
    rows = out.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_report(artifacts: dict) -> None:
    shop_features = artifacts["shop_features"]
    profiles = artifacts["cluster_profiles"]
    suspicious = artifacts["suspicious_shops"]
    hubs = artifacts["portability_hubs"]

    diagnostics = pd.DataFrame(
        {
            "k": artifacts["k_values"],
            "inertia": artifacts["inertias"],
            "silhouette": artifacts["silhouette_by_k"],
        }
    )
    best_k = _best_k_row(diagnostics)

    purity = compute_cluster_purity(shop_features, label_col="districtType")

    lines = []
    lines.append("# Telangana PDS Cluster Profile Report")
    lines.append("")
    lines.append("## Run Summary")
    lines.append(f"- Shops analyzed: {shop_features['shopNo'].nunique():,}")
    lines.append(f"- K-Means clusters: {shop_features['kmeans_cluster'].nunique()}")
    lines.append(f"- DBSCAN outliers: {(shop_features['dbscan_cluster'] == -1).sum():,}")
    lines.append(f"- Silhouette score (selected K): {artifacts['silhouette_score']:.4f}")
    if best_k is not None:
        lines.append(f"- Best K by silhouette curve: {int(best_k['k'])} ({best_k['silhouette']:.4f})")
        if int(best_k["k"]) != artifacts["kmeans_model"].n_clusters:
            lines.append(
                "- Note: K=5 is retained for operationally useful shop personas, "
                "even though the silhouette curve prefers a different K."
            )
    if purity is not None:
        lines.append(f"- Cluster purity vs districtType: {purity:.4f}")
    else:
        lines.append("- Cluster purity: not computed (districtType column missing)")

    lines.append("")
    lines.append("## Cluster Profiles")
    lines.append("")
    lines.append(_to_markdown_table(profiles.round(4)))

    lines.append("")
    lines.append("## Top Suspicious Shops (Fraud Risk Proxy)")
    lines.append("")
    if suspicious.empty:
        lines.append("No suspicious shops were flagged with current threshold.")
    else:
        cols = [
            c
            for c in ["shopNo", "distCode", "distName", "kmeans_cluster", "meanTransactionToCardRatio", "clusterZScore"]
            if c in suspicious.columns
        ]
        lines.append(_to_markdown_table(suspicious[cols].head(20).round(4), index=False))

    lines.append("")
    lines.append("## Portability Hubs (Logistics Priority)")
    lines.append("")
    if hubs.empty:
        lines.append("No portability hubs were identified.")
    else:
        cols = [
            c
            for c in ["shopNo", "distCode", "distName", "kmeans_cluster", "portabilityLoad", "totalOtherShopTrans"]
            if c in hubs.columns
        ]
        lines.append(_to_markdown_table(hubs[cols].head(20).round(4), index=False))

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    artifacts = run_full_pipeline(
        data_root=str(DATA_ROOT),
        n_clusters=5,
        dbscan_eps=1.5,
        dbscan_min_samples=5,
    )

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
    unified_export = artifacts["unified"][[
        c for c in unified_export_cols if c in artifacts["unified"].columns
    ]].copy()

    artifacts["unified"] = unified_export
    artifacts["unified"].to_csv(PROCESSED_DIR / "unified_dataset.csv", index=False)
    artifacts["shop_features"].to_csv(PROCESSED_DIR / "shop_features_clustered.csv", index=False)
    artifacts["cluster_profiles"].to_csv(PROCESSED_DIR / "cluster_profiles.csv")
    artifacts["suspicious_shops"].to_csv(PROCESSED_DIR / "suspicious_shops.csv", index=False)
    artifacts["portability_hubs"].to_csv(PROCESSED_DIR / "portability_hubs.csv", index=False)

    diagnostics = pd.DataFrame(
        {
            "k": artifacts["k_values"],
            "inertia": artifacts["inertias"],
            "silhouette": artifacts["silhouette_by_k"],
        }
    )
    diagnostics.to_csv(PROCESSED_DIR / "cluster_diagnostics.csv", index=False)

    write_report(artifacts)

    print("Pipeline completed.")
    print(f"Processed files written to: {PROCESSED_DIR}")
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
