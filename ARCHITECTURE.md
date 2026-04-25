# Architecture

## Mental Model

We are building a function that answers:

> "Given what satellites and weather stations observed about corn fields in Iowa as of August 1, 2025 — how many bushels per acre will Iowa produce at end of season?"

We answer this by teaching a model on 20 years of historical data (2005–2024), then pointing it at 2025.

---

## Pipeline Overview

```
RAW DATA SOURCES
      │
      ▼
┌─────────────────────────────────────┐
│  01_quickstats.ipynb                │
│  USDA NASS yield data (2005–2024)   │
│  → Y variable (what we predict)     │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  02_weather.ipynb                   │
│  NOAA temperature + precipitation   │
│  per state per month (2005–2024)    │
│  → X features                       │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  03_satellite.ipynb                 │
│  HLS / Landsat / Sentinel-2         │
│  NDVI per state per forecast date   │
│  (AWS S3 — pending access)          │
│  → X features                       │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  04_merge_features.ipynb            │
│  Combine all sources into one flat  │
│  training CSV:                      │
│  Year | State | NDVI | Temp |       │
│  Precip | ... | Yield (Y)           │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  05_model.ipynb                     │
│  Random Forest Regressor            │
│  Train: 2005–2020 (80%)             │
│  Test:  2021–2024 (20%)             │
│  Predict: 2025                      │
│  Output: yield + cone of            │
│  uncertainty per state per date     │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  06_viz.ipynb                       │
│  Charts, maps, uncertainty bands    │
│  → presentation assets              │
└─────────────────────────────────────┘
```

---

## Training Data Schema

Each row in the final training CSV represents one state in one year:

| Column | Source | Description |
|---|---|---|
| `year` | QuickStats | 2005–2024 |
| `state` | QuickStats | IA, CO, WI, MO, NE |
| `yield_bu_acre` | QuickStats | **Y variable** — what we predict |
| `ndvi_aug1` | HLS/Landsat | Vegetation index as of Aug 1 |
| `ndvi_sep1` | HLS/Landsat | Vegetation index as of Sep 1 |
| `ndvi_oct1` | HLS/Landsat | Vegetation index as of Oct 1 |
| `temp_jun_aug` | NOAA | Avg temperature June–August |
| `precip_jun_aug` | NOAA | Total precipitation June–August |
| `temp_may` | NOAA | Planting season temperature |
| `precip_may` | NOAA | Planting season precipitation |

---

## Model

**Random Forest Regressor** (scikit-learn)

- Train on 2005–2020
- Validate on 2021–2024
- Predict 2025
- Four separate models, one per forecast date (Aug 1, Sep 1, Oct 1, End of Season)
- Cone of uncertainty via bootstrap confidence intervals

**Analog year identification:** find historical years with the most similar NDVI + weather profile to 2025. Use their actual yields to bound the uncertainty cone.

---

## Data Sources

| Dataset | Used For | Status |
|---|---|---|
| USDA QuickStats | Y variable (yield labels) | Loaded |
| NOAA Climate Data | Weather features | Pending |
| HLS Satellite (NASA) | NDVI features | Pending AWS |
| Cropland Data Layer | Corn field masking | Pending |
| NAIP Imagery | Corn field masking | Pending |
| Prithvi Foundation Model | Optional richer features | Stretch goal |

---

## Fallback Plan (No AWS)

QuickStats + NOAA weather alone produces a legitimate yield model. NDVI from satellite is the differentiator — if AWS access is restored with time remaining, satellite features get merged in via `03_satellite.ipynb`. The pipeline is designed so this is a drop-in addition, not a rebuild.
