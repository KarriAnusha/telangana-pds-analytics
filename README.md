# Telangana PDS Analytics

Multi-Dimensional Shop Performance Clustering and Anomaly Profiling for the Telangana State Public Distribution System (PDS).

## Project Structure

```
telangana-pds-analytics/
├── data/
│   ├── transactions/          # Place transaction CSV files here (2023-2025)
│   ├── card_status/           # Place card status CSV files here
│   ├── fps_locations/         # Place FPS location CSV files here
│   └── processed/             # Auto-generated after running the notebook
├── notebooks/
│   └── eda_and_modeling.ipynb # Comprehensive EDA and Model Development
├── scripts/
│   └── run_pipeline.py        # Reproducible pipeline execution + report generation
├── reports/
│   └── cluster_profile_report.md # Auto-generated final report
├── app.py                     # Streamlit interactive dashboard
├── data_processing.py         # Data pipeline and clustering module
├── requirements.txt           # Python dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Data Preparation

Download CSV files from the [Telangana Open Data Portal](https://data.telangana.gov.in/) and place them in:

| Dataset | Folder |
|---------|--------|
| Transactions (monthly volumes, portability) | `data/transactions/` |
| Card Status (entitlement, ration card counts) | `data/card_status/` |
| FPS Locations (coordinates, shop status) | `data/fps_locations/` |

**Expected key columns across datasets:**
- `shopNo`, `distCode` — used as join keys
- `noOfTrans`, `otherShopTrans`, `riceQty`, `wheatQty` — in transactions
- `totalRcs` — in card status
- `latitude`/`lat`, `longitude`/`lng` — in FPS locations

## Running the Notebook

```bash
cd notebooks
jupyter notebook eda_and_modeling.ipynb
```

Run all cells in order. The notebook will:
1. Load and consolidate raw CSVs
2. Perform detailed EDA (trends, correlations, distributions, seasonality)
3. Engineer features (utilization ratio, commodity intensity, volatility)
4. Run PCA, K-Means, and DBSCAN clustering
5. Save processed data to `data/processed/` for the dashboard

## Running the Pipeline Script (Recommended)

```bash
python scripts/run_pipeline.py
```

This command creates all deliverables used by the dashboard and presentation:
- `data/processed/unified_dataset.csv`
- `data/processed/shop_features_clustered.csv`
- `data/processed/cluster_profiles.csv`
- `data/processed/suspicious_shops.csv`
- `data/processed/portability_hubs.csv`
- `data/processed/cluster_diagnostics.csv`
- `reports/cluster_profile_report.md`

## Running the Dashboard

```bash
streamlit run app.py
```

The dashboard provides:
- **Geospatial Map** — Shops color-coded by K-Means Cluster ID
- **Shop Search** — Input a shopNo to compare its performance against its cluster average
- **Cluster Profiles** — Feature means per cluster and DBSCAN outlier summary
- **Hotspot Map** — Highlight concentrations of specific cluster types
- **Policy Impact** — Year-wise portability trend and monthly seasonality analysis
- **Risk & Logistics** — Suspicious shop flags and high portability hub identification

## Technical Details

- **Scaling:** StandardScaler applied before all distance-based models
- **PCA:** Reduces features to 3 components for visualization
- **K-Means:** Default 5 clusters; validated with Elbow Curve and Silhouette Score
- **DBSCAN:** Identifies noise/outlier shops (label = -1)
- **Caching:** `st.cache_data` used for efficient large dataset loading in Streamlit

## Evaluation Metrics Implemented

- **Silhouette Score** for selected K-Means model
- **Elbow + Silhouette curve** exported to `cluster_diagnostics.csv`
- **Cluster Purity helper** available in `data_processing.py` against labels like `districtType` (if present)

## Notes on Data Variants

The pipeline normalizes common column naming differences (for example `shopno` -> `shopNo`, `lat` -> `latitude`) so CSVs from different years can be combined more reliably.
