# CSU Hackathon 2026 — Geospatial AI Crop Yield Forecasting

**Team:** Dev Lucas, Juan, Haven, Sophia

---

## Challenge

Predict corn grain yield (bushels/acre) for Iowa, Colorado, Wisconsin, Missouri, and Nebraska
at four forecast dates during the 2025 growing season, with a cone of uncertainty per forecast.
Ground truth: 2025 actuals already published by USDA NASS.

| Forecast Date | Crop Stage |
|---|---|
| August 1 | Early grain fill |
| September 1 | Late grain fill / dough stage |
| October 1 | Maturity / early harvest |
| End of Season | Post-harvest reconciliation |

---

## Deliverables

1. Working model pipeline with documented code
2. 2025 yield forecasts for all 5 states at all 4 forecast dates
3. Cone of uncertainty per state per date
4. 5–7 minute presentation (methodology, pipeline, model, results)

---

## Stack

- Python 3.9 — `conda activate geospatial-python-crash-course`
- pandas, numpy, scikit-learn, matplotlib, geopandas, rasterio
- earthaccess, rioxarray, pystac-client (satellite notebooks)
- Prithvi-100M via Hugging Face: `nasa-ibm/prithvi-100m` (model Phase 2, SageMaker GPU)

---

## Quickstart

```bash
git clone <repo-url>
cd top_CSUHackathon26
conda activate geospatial-python-crash-course
cp .env.example .env   # add your NOAA_API_KEY
jupyter notebook
```

---

## Notebook Execution Order

| Notebook | Output | Status | Notes |
|---|---|---|---|
| `01_quickstats` | `data/processed/quickstats_yield.csv` | ✅ Complete | |
| `02_weather` | `data/processed/weather_features.csv` | ✅ Complete | Requires `NOAA_API_KEY` in `.env` |
| `03_satellite_fixed` | `data/raw/ndvi_by_state_date.csv` | ✅ Complete | **SageMaker only** — CSV committed to repo, do not re-run locally |
| `04_merge_features` | `data/processed/training_features.csv` | ✅ Complete | |
| `05_model_production` | `outputs/predictions.05_model_production.csv` | ✅ Complete | Random Forest v1.4 — production model |
| `outputs/` | `full_diff.csv`, `comparison_summary.csv` | ✅ Complete | Run `notebooks/analysis/07_compare.ipynb` to regenerate |

> `03_satellite_fixed.ipynb` runs on SageMaker and requires NASA Earthdata credentials via IAM.
> It includes CDL corn-field masking (reprojected to HLS pixel grid) and a checkpoint/resume loop.
> The output CSV (`ndvi_by_state_date.csv`) is committed to the repo — do not re-run locally.
> All other notebooks run in the local conda environment.

---

## Model Selection

Model 1.4 (Random Forest + extended training window + per-state bias correction) was selected
after comparing four model versions.

See [MODEL_SELECTION.md](MODEL_SELECTION.md) for the full evaluation methodology and rationale.

For detailed metrics comparison, see `outputs/full_diff.csv`.

---

## Quick Results Summary

**2025 Forecasts (`05_model_production.ipynb`)**

For full accuracy, calibration, and error metrics, see:
- `outputs/full_diff.csv` — Model version comparison
- `outputs/USDA_TRUTH.csv` — Validation ground truth (if available)

Key finding: Model 1.4 selected over 4 experimental versions.
See [MODEL_SELECTION.md](MODEL_SELECTION.md) for methodology and reasoning.

---

## Contacts in the Room

| Person | Role |
|---|---|
| Andrew Dau | USDA NASS — crop yield track prompt owner |
| Kevin (NASA) | HLS data and Prithvi setup questions |
| AWS Solutions Architects | SageMaker environment, GPU instances |
| David Bartels | USDA — fruit fly track |
| Esri reps | Visualization and GIS |

**Go ask them questions. The winners always do.**
