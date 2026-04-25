# Roadmap

## Timeline: ~22 Hours (Friday 7pm → Saturday 5pm)

---

## ✅ DONE
- Repo initialized
- Conda environment set up (`geospatial-python-crash-course`)
- All packages installed and verified
- QuickStats CSV downloaded and loaded (22,832 rows)
- Crash course notebooks cloned from `col.st/kj5oh`

---

## 🔴 PHASE 1 — Data (Hours 0–3, ~7pm–10pm Friday)

**Goal: Every dataset loaded and readable. No modeling yet.**

### 1A — Fix QuickStats filter (`notebooks/01_quickstats.ipynb`)
The raw CSV contains mixed data. Filter to exactly what we need:
- Program = SURVEY
- Data Item = CORN, GRAIN - YIELD, MEASURED IN BU / ACRE
- Geo Level = STATE
- States = Iowa, Colorado, Wisconsin, Missouri, Nebraska
- Years = 2005–2024

Save clean output to `data/processed/quickstats_yield.csv`

### 1B — Pull NOAA Weather (`notebooks/02_weather.ipynb`)
Download monthly temperature and precipitation for each of the 5 states, 2005–2024.
API: `https://www.ncei.noaa.gov/cdo-web/api/v2/`
Free API key: `https://www.ncdc.noaa.gov/cdo-web/token`
Save to `data/raw/noaa_weather.csv`

### 1C — Satellite / HLS (`notebooks/03_satellite.ipynb`)
**Blocked on AWS access.** When unblocked:
- Pull HLS tiles for each state around Aug 1, Sep 1, Oct 1 per year
- Compute NDVI per tile
- Average NDVI per state per date
- Save to `data/raw/ndvi_by_state_date.csv`

**If AWS stays blocked:** skip and proceed with QuickStats + NOAA only.

---

## 🟡 PHASE 2 — Feature Engineering (Hours 3–6, ~10pm–1am Friday)

**Goal: One flat training CSV combining all data sources.**

### `notebooks/04_merge_features.ipynb`
- Join QuickStats yield + NOAA weather on `year` + `state`
- Join NDVI if available
- Output schema: `Year | State | ndvi_aug1 | temp_jun_aug | precip_jun_aug | ... | yield_bu_acre`
- Save to `data/processed/training_features.csv`

---

## 🟡 PHASE 3 — Model (Hours 6–12, ~1am–7am Saturday)

**Goal: Predictions for all 5 states at all 4 forecast dates.**

### `notebooks/05_model.ipynb`
- Train Random Forest Regressor on 2005–2020
- Validate on 2021–2024, report RMSE
- Predict 2025 yield for each state at each forecast date
- Generate cone of uncertainty via bootstrap (500 iterations)
- Analog year identification: find top 3 historical years most similar to 2025 by feature distance
- Save predictions to `outputs/predictions.csv`

**Hyperparameter tuning:** adjust `n_estimators`, `max_depth`, `min_samples_leaf` until validation RMSE is minimized.

---

## 🟢 PHASE 4 — Visualization (Hours 12–18, ~7am–1pm Saturday)

**Goal: Charts and maps ready for the presentation.**

### `notebooks/06_viz.ipynb`
- Time series: yield forecast trajectory per state across 4 dates
- Cone of uncertainty: shaded bands around point estimates
- Bar chart: predicted vs. historical yield by state
- Map: choropleth of predicted 2025 yield across 5 states

---

## 🟢 PHASE 5 — Presentation (Hours 18–22, ~1pm–5pm Saturday)

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

## Stretch Goals (only if ahead of schedule)
- Integrate Prithvi foundation model embeddings as additional features
- County-level predictions (more granular than state)
- Interactive map visualization
- Integrate HLS satellite NDVI once AWS access is restored

---

## Contacts in the Room
- **David Bartels** — USDA, prompt owner for fruit fly track
- **Andrew Dau** — USDA NASS, prompt owner for crop yield track (our prompt)
- **Kevin (NASA)** — HLS data and Prithvi questions
- **AWS Solutions Architects** — environment issues, S3 access
- **Esri reps** — visualization and GIS questions

**Don't be shy. Go ask them questions. The winners always do.**
