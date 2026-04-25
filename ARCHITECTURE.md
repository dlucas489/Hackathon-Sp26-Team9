# Architecture

## Mental Model

> "Given what satellites and weather stations observed about corn fields in Iowa, Colorado,
> Wisconsin, Missouri, and Nebraska as of a given date in 2025 — how many bushels per acre
> will each state produce at end of season?"

Answered by training on 20 years of historical data (2005–2024), then predicting 2025.

---

## MVP Definition

Guaranteed deliverables within the hackathon window:

1. End-to-end pipeline executes without errors (notebooks 01 → 06)
2. 2025 yield predictions for all 5 states at all 4 forecast dates
3. Bootstrap confidence interval (cone of uncertainty) per state per date
4. Analog year identification per state per date
5. Presentation-ready charts

**Beyond MVP (priority order):**
1. Prithvi replacing Random Forest in `05_model.ipynb`
2. NAIP / CDL corn field masking

---

## Pipeline

```
RAW DATA SOURCES
      │
      ▼
┌─────────────────────────────────────┐
│  01_quickstats.ipynb                │
│  USDA NASS yield data (2005–2024)   │
│  → data/processed/quickstats_yield  │
│  STATUS: ✅ Complete                │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  02_weather.ipynb                   │
│  NOAA GSOM — monthly TAVG + PRCP    │
│  5 states × May–Oct × 2005–2025     │
│  → data/processed/weather_features  │
│  STATUS: ✅ Complete                │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  03_satellite_fixed.ipynb (SageMaker)│
│  NASA HLS S30 — NDVI per state      │
│  CDL corn-field masking applied     │
│  per forecast date, 2015–2024       │
│  → data/raw/ndvi_by_state_date.csv  │
│  STATUS: ✅ Complete (CSV committed) │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  04_merge_features.ipynb            │
│  Joins QuickStats + NOAA + NDVI     │
│  on year + state                    │
│  → data/processed/training_features │
│  STATUS: ✅ Complete                │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  05_model_production.ipynb          │
│  Random Forest Regressor (v1.4)     │
│  Prithvi-100M (under consideration) │
│  → outputs/predictions.05_model_production.csv │
│  STATUS: ✅ Complete                │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  06_viz.ipynb (notebooks/analysis/) │
│  Yield trajectories, uncertainty    │
│  cones, analog year overlays        │
│  → outputs/*.png                    │
│  STATUS: ✅ Complete                │
└─────────────────────────────────────┘
```

---

## Model Architecture

### Option A — Random Forest Regressor (MVP baseline)

Four `RandomForestRegressor` instances (scikit-learn), one per forecast date.
Feature sets are strictly bounded by data available at each date — no future leakage.

| Model | Features |
|---|---|
| `aug1` | Weather May–Jul + ndvi_aug1 |
| `sep1` | Weather May–Aug + ndvi_aug1, ndvi_sep1 |
| `oct1` | Weather May–Sep + ndvi_aug1–oct1 |
| `final` | Weather May–Oct + ndvi_aug1–final |

Train: 2005–2023 | Validate: 2024 | Predict: 2025

Uncertainty: 500-iteration bootstrap (5th–95th percentile CI).
Analog years: top 3 historical years by Euclidean distance on normalized features.

### Model Selection: Production Choice (v1.4)

**Selected:** Random Forest Regressor with extended training window and per-state bias correction.

**Rationale:** See [MODEL_SELECTION.md](MODEL_SELECTION.md) and `outputs/full_diff.csv` for full comparison.

Key improvements over baseline (v1.0):
- Extended training: 2005–2020 → 2005–2023
- Per-state detrending: Removes long-term trend before training
- Per-state bias correction: Validation errors subtracted from predictions

Result: Validation accuracy and uncertainty calibration significantly improved vs. earlier versions.
See `outputs/full_diff.csv` for validation results per model per date.

### Validation Strategy

- **Train:** 2005–2023 (19 years of historical data)
- **Test:** 2024 only (held-out validation set; 5 state-years)
- **Predict:** 2025 (5 state-years; ground truth not yet available)

Metrics computed per forecast date (aug1, sep1, oct1, final).

**No data leakage:** Each forecast date uses only features available at that date.
- aug1: Weather May–Jul + ndvi_aug1
- sep1: Weather May–Aug + ndvi_aug1, ndvi_sep1
- oct1: Weather May–Sep + ndvi_aug1–oct1
- final: Weather May–Oct + ndvi_aug1–final

See `outputs/full_diff.csv` for validation results per model per date.

### Option B — Prithvi-100M (primary model per prompt, under consideration)

The hackathon prompt specifies Prithvi — NASA/IBM's open-source geospatial foundation model
(`nasa-ibm/prithvi-100m`, available on Hugging Face) — as the intended model backbone.

Prithvi replaces raw NDVI scalars with rich temporal-spectral embeddings extracted from HLS
tile stacks. A lightweight regression head (MLP or RF) trains on top of the frozen encoder.
The pipeline architecture does not change — only `05_model.ipynb` is modified. Output schema
(predictions.csv) is identical to Option A. Runs on SageMaker GPU (ml.g4dn.xlarge or larger).

Contact: Kevin (NASA) in the hackathon room for HLS tile access and Prithvi setup questions.

> **Decision:** Random Forest (Option A) was selected as the production model (v1.4).
> See [MODEL_SELECTION.md](MODEL_SELECTION.md) for full rationale and `outputs/full_diff.csv` for metrics.

---

## Training Data Schema

| Column | Source | Coverage |
|---|---|---|
| `year` | QuickStats | 2005–2024 |
| `state` | QuickStats | IA, CO, WI, MO, NE |
| `yield_bu_acre` | QuickStats | **Y variable** |
| `ndvi_aug1/sep1/oct1/final` | HLS S30 (2015–2024) + MODIS MOD13Q1 (2005–2014) | 2005–2024 |
| `tavg_may` – `tavg_oct` | NOAA GSOM | 2005–2025 |
| `prcp_may` – `prcp_oct` | NOAA GSOM | 2005–2025 |

MODIS MOD13Q1 resolution is 250m vs. HLS S30 at 30m. Both are valid vegetation indices for
state-level modeling — the resolution difference is immaterial when averaging across an entire
state bounding box.

---

## Data Sources

| Dataset | Purpose | Status |
|---|---|---|
| USDA QuickStats | Y variable (yield labels) | ✅ Complete |
| NOAA GSOM API | Weather features | ✅ Complete |
| NASA HLS S30 | NDVI 2015–2024 (CDL-masked) | ✅ Complete (CSV committed) |
| Prithvi-100M | Geospatial embeddings (Option B) | Under consideration |
| Cropland Data Layer | Corn field masking | ✅ Integrated in 03_satellite_fixed |
| NAIP Imagery | Field boundary data | Not integrated |

---

## AWS / SageMaker

SageMaker is a managed cloud Jupyter environment with GPU access and pre-attached IAM
credentials. It is used exclusively for `03_satellite.ipynb` because that notebook requires
NASA Earthdata credentials (provisioned via IAM + `.netrc`) and benefits from cloud proximity
to NASA S3 endpoints.

All other notebooks run locally in the `geospatial-python-crash-course` conda environment.

**Satellite data workflow:**
1. Teammate runs `03_satellite_fixed.ipynb` on SageMaker → produces `ndvi_by_state_date.csv`
2. Teammate commits the CSV directly to the repo (**done** — CSV is in `data/raw/`)
3. `03_satellite_fixed.ipynb` lives in the repo for documentation and reproducibility only
4. No other team member needs SageMaker access to proceed
