# Roadmap

## Timeline: ~22 Hours (Friday 7pm → Saturday 5pm)

---

## ✅ DONE
- Repo initialized
- Conda environment set up (`geospatial-python-crash-course`)
- All packages installed and verified
- Crash course notebooks cloned from `col.st/kj5oh`
- **`01_quickstats.ipynb` complete** — QuickStats CSV filtered and saved (22,832 rows → `data/processed/quickstats_yield.csv`)
- **`02_weather.ipynb` complete** — NOAA GSOM monthly TAVG + PRCP pulled for all 5 states, 2005–2024 (`data/processed/weather_features.csv`)
- **`03_satellite_fixed.ipynb` complete** — HLS S30 NDVI computed with CDL corn-field masking on SageMaker; `ndvi_by_state_date.csv` committed to repo

---

## 🔴 PHASE 1 — Data (Hours 0–3, ~7pm–10pm Friday) — ✅ COMPLETE

**Goal: Every dataset loaded and readable. No modeling yet.**

### ✅ 1A — QuickStats filter (`notebooks/01_quickstats.ipynb`)
Filtered raw CSV to exactly what we need:
- Program = SURVEY
- Data Item = CORN, GRAIN - YIELD, MEASURED IN BU / ACRE
- Geo Level = STATE
- States = Iowa, Colorado, Wisconsin, Missouri, Nebraska
- Years = 2005–2024

Output: `data/processed/quickstats_yield.csv`

### ✅ 1B — NOAA Weather (`notebooks/02_weather.ipynb`)
Monthly temperature and precipitation for each of the 5 states, 2005–2024.
API: `https://www.ncei.noaa.gov/cdo-web/api/v2/`
Output: `data/processed/weather_features.csv`

### ✅ 1C — Satellite / HLS (`notebooks/03_satellite_fixed.ipynb`)
- HLS tiles pulled for each state around Aug 1, Sep 1, Oct 1 per year (2015–2024)
- CDL corn-field mask applied (reprojected to HLS pixel grid via `reproject_match`)
- NDVI averaged per state per forecast date
- Output: `data/raw/ndvi_by_state_date.csv` (committed to repo)

---

## ✅ PHASE 2 — Feature Engineering (Hours 3–6, ~10pm–1am Friday) — ✅ COMPLETE

**Goal: One flat training CSV combining all data sources.**

### `notebooks/04_merge_features.ipynb` — complete
- Join QuickStats yield + NOAA weather on `year` + `state`
- Join NDVI from `data/raw/ndvi_by_state_date.csv`
- Output schema: `year | state | ndvi_aug1 | ndvi_sep1 | ndvi_oct1 | ndvi_final | tavg_may … tavg_oct | prcp_may … prcp_oct | yield_bu_acre`
- Save to `data/processed/training_features.csv`

---

## ✅ PHASE 3 — Model Selection & Training — COMPLETE

**Goal: Predictions for all 5 states at all 4 forecast dates.**

### `notebooks/05_model_production.ipynb`

**Selected:** Option A — Random Forest Regressor (v1.4)

Implementation:
- Four RF regressors (one per forecast date: aug1, sep1, oct1, final)
- Train: 2005–2023 | Validate: 2024 | Predict: 2025
- Bootstrap confidence intervals: 500 iterations, 5th–95th percentile
- Per-state analog year identification (top 3 by normalized feature distance)
- Per-state detrending + per-state bias correction

Results: See `outputs/full_diff.csv` for accuracy, error, and calibration metrics.

Key insight: Model 1.4 was selected because earlier versions (1.0–1.2) provided
overconfident uncertainty estimates, while version 1.5 overfitted.
See [MODEL_SELECTION.md](MODEL_SELECTION.md) for full comparison.

Previous model iterations archived to `notebooks/archive/`.

---

## ✅ PHASE 4 — Visualization (Hours 12–18, ~7am–1pm Saturday) — COMPLETE

**Goal: Charts and maps ready for the presentation.**

### `notebooks/analysis/06_viz.ipynb`
- Time series: yield forecast trajectory per state across 4 dates
- Cone of uncertainty: shaded bands around point estimates
- Bar chart: predicted vs. historical yield by state
- Map: choropleth of predicted 2025 yield across 5 states

---

## ✅ PHASE 5 — Presentation (Hours 18–22, ~1pm–5pm Saturday) — COMPLETE

**Goal: 5–7 minute pitch ready by 5pm.**

### Slide structure
1. Problem & why it matters (30 sec)
2. Our pipeline — what data, how it flows (1 min)
3. Model architecture & validation results (1 min)
4. 2025 yield predictions + uncertainty cones (2 min)
5. What we'd do with more time / satellite data (30 sec)
6. Impact statement (30 sec)

**Key message for judges:** frame every result in terms of real-world impact. "Our August 1 model predicts Iowa at X bu/acre ± Y — that's actionable 2 months before harvest."

---

## ✅ PHASE 6 — Documentation & Repository Reorganization (2.5h before presentation)

- ✅ Organized notebooks: production vs. archive vs. analysis
- ✅ Updated README, ARCHITECTURE, ROADMAP, MODEL_SELECTION for clarity
- ✅ Populated requirements.txt with all dependencies
- ✅ Updated .env.example with credential placeholders
- ✅ Reorganized directory structure for discoverability

All metrics referenced from CSV files (full_diff.csv, comparison_summary.csv, etc.).

---

## Stretch Goals (only if ahead of schedule)
- County-level predictions (more granular than state)
- Interactive map visualization

---

## Contacts in the Room
- **David Bartels** — USDA, prompt owner for fruit fly track
- **Andrew Dau** — USDA NASS, prompt owner for crop yield track (our prompt)
- **Kevin (NASA)** — HLS data and Prithvi questions
- **AWS Solutions Architects** — environment issues, S3 access
- **Esri reps** — visualization and GIS questions

**Don't be shy. Go ask them questions. The winners always do.**
