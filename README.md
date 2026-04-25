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

| Notebook | Output | Notes |
|---|---|---|
| `01_quickstats` | `data/processed/quickstats_yield.csv` | |
| `02_weather` | `data/processed/weather_features.csv` | Requires `NOAA_API_KEY` in `.env` |
| `03_satellite` | `data/raw/ndvi_by_state_date.csv` | **SageMaker only** — commit CSV to repo, do not re-run locally |
| `03b_modis_backfill` | `data/raw/ndvi_combined.csv` | Merges MODIS (2005–2014) + HLS (2015–2024); runs locally |
| `04_merge_features` | `data/processed/training_features.csv` | |
| `05_model` | `outputs/predictions.csv` | Phase 1: RF baseline → Phase 2: Prithvi |
| `06_viz` | `outputs/*.png` | |

> `03_satellite.ipynb` runs on SageMaker and requires NASA Earthdata credentials via IAM.
> Once complete, the teammate running it commits `ndvi_by_state_date.csv` directly to the repo.
> All other notebooks run in the local conda environment.

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
