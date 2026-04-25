# Roadmap

## Timeline: ~17 Hours Remaining

---

## ✅ DONE

- Repo initialized, conda environment verified
- QuickStats CSV downloaded and loaded (22,832 rows)
- `01_quickstats.ipynb` — filtered, cleaned, output saved
- `02_weather.ipynb` — NOAA GSOM pull complete
- `03_satellite.ipynb` — Iowa NDVI complete, Colorado in progress (running on SageMaker)
- Crash course notebooks cloned

---

## 🔴 PHASE 1 — Data Completion (In Progress)

**Goal: All CSVs committed to repo. No modeling until this is done.**

### 1A — Finish satellite pull (`03_satellite.ipynb` — teammate on SageMaker)
- Remaining states: Wisconsin, Missouri, Nebraska
- Once all states complete: commit `data/raw/ndvi_by_state_date.csv` to repo
- Do not block other work on this — phases 1B and 2 can proceed in parallel

### 1B — MODIS backfill (`03b_modis_backfill.ipynb` — run locally)
- Pulls MODIS MOD13Q1 pre-computed NDVI for 2005–2014 (pre-HLS coverage)
- Concatenates with HLS output → `data/raw/ndvi_combined.csv`
- Same `earthaccess` auth as satellite notebook; no GPU required
- Can begin immediately — does not depend on `03_satellite.ipynb` completing first
  (notebook handles concatenation once both inputs are present)

---

## 🟡 PHASE 2 — Feature Engineering

**Goal: One flat training CSV combining all sources.**

### `04_merge_features.ipynb`
- Join QuickStats yield + NOAA weather on `year` + `state`
- Join `ndvi_combined.csv` on `year` + `state`
- Output schema: `year | state | ndvi_aug1 | ndvi_sep1 | ndvi_oct1 | ndvi_final | tavg_may | prcp_may | ... | yield_bu_acre`
- Save to `data/processed/training_features.csv`
- Hard error if NDVI data is missing — do not proceed without satellite features

---

## 🟡 PHASE 3 — Model

**Goal: Real predictions for all 5 states at all 4 forecast dates.**

### `05_model.ipynb` — Phase 1 (Random Forest baseline)
- Four `RandomForestRegressor` instances, one per forecast date
- Feature sets bounded by data available at each forecast date — no future leakage
- Train: 2005–2020 | Validate: 2021–2024 | Predict: 2025
- Bootstrap uncertainty: 500 iterations, 5th–95th percentile CI
- Analog year identification: top 3 historical years by Euclidean distance on normalized features
- Output: `outputs/predictions.csv` — columns: `state`, `forecast_date`, `predicted_yield`, `ci_lower`, `ci_upper`, `analog_years`, `val_rmse`

**Hyperparameter tuning:** adjust `n_estimators`, `max_depth`, `min_samples_leaf` to minimize validation RMSE.

### `05_model.ipynb` — Phase 2 (Prithvi upgrade)
Prithvi (`nasa-ibm/prithvi-100m`) is the primary model specified by the prompt. Implement
after RF baseline is producing valid predictions. Runs on SageMaker GPU.

- Load Prithvi encoder from Hugging Face
- Extract temporal-spectral embeddings from HLS tile stacks per state per forecast date
- Concatenate embeddings with weather features
- Train lightweight regression head on combined feature matrix
- Output schema identical to Phase 1 — no downstream changes

Contact Kevin (NASA) for Prithvi access and HLS tile format questions.

---

## 🟢 PHASE 4 — Visualization

**Goal: Presentation-ready charts from `outputs/predictions.csv`.**

### `06_viz.ipynb`
- Time series: yield forecast trajectory per state across 4 forecast dates
- Uncertainty cone: shaded 90% CI band around point estimates
- Bar chart: 2025 predicted vs. historical average yield by state
- Analog year overlay: highlight the top 3 analog years on historical trend lines
- Choropleth map: predicted 2025 yield across 5 states (geopandas)

---

## 🟢 PHASE 5 — Presentation

**Goal: 5–7 minute pitch ready by deadline.**

### Slide structure
1. Problem and why it matters (30 sec)
2. Pipeline — data sources, how they flow (1 min)
3. Model architecture and validation RMSE (1 min)
4. 2025 predictions and uncertainty cones per state (2 min)
5. Prithvi integration — status and next steps (30 sec)
6. Impact statement (30 sec)

**Key message:** frame every result in terms of real-world impact.
*"Our August 1 model predicts Iowa at X bu/acre ± Y — actionable 2 months before harvest."*

---

## Beyond MVP (only if ahead of schedule)

- CDL / NAIP corn field masking for cleaner NDVI signal
- County-level predictions (more granular than state)
- Interactive choropleth map

---

## Contacts in the Room

| Person | Role |
|---|---|
| Andrew Dau | USDA NASS — our prompt owner |
| Kevin (NASA) | HLS data and Prithvi setup |
| AWS Solutions Architects | SageMaker GPU instances |
| Esri reps | Visualization and GIS |
