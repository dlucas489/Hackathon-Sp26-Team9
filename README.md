# CSU Hackathon 2026 — Geospatial AI Crop Yield Forecasting

## Team
- Dev Lucas
- Juan
- Haven
- Sophia

## The Challenge
Build a machine learning pipeline that predicts **corn grain yield (bushels/acre)** for five states — Iowa, Colorado, Wisconsin, Missouri, and Nebraska — at four points during the 2025 growing season:

| Forecast Date | Crop Stage |
|---|---|
| August 1 | Early grain fill |
| September 1 | Late grain fill / dough stage |
| October 1 | Maturity / early harvest |
| End of Season | Post-harvest reconciliation |

Each forecast must include a **cone of uncertainty** (confidence interval) derived from analog historical years.

The 2025 actuals are already published by USDA NASS. Our predictions will be ground-truthed against them.

## Deliverables
1. Working model pipeline with documented code
2. Yield forecast outputs for all 5 states at all 4 time points for the 2025 season
3. Cone of uncertainty for each state and forecast date
4. 5–7 minute presentation covering methodology, pipeline, model architecture, and results

## Stack
- Python 3.9 (conda env: `geospatial-python-crash-course`)
- pandas, numpy, scikit-learn, matplotlib, geopandas, rasterio
- Jupyter Notebooks
- Git for version control

## Quickstart
```bash
git clone <repo-url>
cd top_CSUHackathon26
conda activate geospatial-python-crash-course
jupyter notebook
```

Run notebooks in order: `01` → `02` → `03` → `04` → `05`
