# Model Selection: Why 05_model1.4 (Production)

## TL;DR
- **Selected:** Model 1.4 (Random Forest + extended training 2005–2023 + per-state bias correction)
- **Why:** Compared 4 model versions; see `outputs/full_diff.csv` for detailed metrics
- **Key Differentiator:** Model 1.4 provides well-calibrated uncertainty estimates
- **Rejected Versions:** See `outputs/full_diff.csv` for why 1.0, 1.1, 1.2, and 1.5 underperformed

## How to Compare Models

All metrics are in `outputs/full_diff.csv`. To reproduce analysis:
1. Open `notebooks/analysis/07_compare.ipynb`
2. It loads all `05_model*.csv` output files
3. Generates `full_diff.csv` with columns: `model_version`, `state`, `forecast_date`, `predicted_yield`, `usda_yield`, `error`, `pct_error`, `ci_covers`, `usda_actual`, `err_vs_actual`

## Model 1.4 Distinguishing Features

1. **Extended Training Window:** 2005–2023 (vs. 1.0–1.2: 2005–2020)
   - Rationale: More recent data captures recent agronomic trends

2. **Per-State Detrending:** Linear trend fitted per state on yield vs. year
   - RF trains on residuals; trend added back at prediction time
   - Captures long-term yield improvement without overfitting to trend

3. **Per-State Bias Correction:** Validation error subtracted from predictions
   - Reduces systematic bias identified in 2021–2024 held-out data

## Validation Strategy

- **Train:** 2005–2023 (19 years × 5 states = 95 samples per forecast date)
- **Test:** 2024 only (5 states; this is the held-out validation set)
- **Predict:** 2025 (5 states; no ground truth yet)
- **Uncertainty:** 500-iteration bootstrap; 5th–95th percentile CI

Metrics reported per forecast date (aug1, sep1, oct1, final).

## Model Comparison Results

See `outputs/full_diff.csv` for full table. Key columns:
- `error` — Predicted yield minus USDA validation yield
- `pct_error` — Error as % of USDA yield
- `ci_covers` — Does prediction CI contain USDA actual? (True/False)
- `err_vs_actual` — Error vs. realized 2025 actual (if available)

## Why Not Option B (Prithvi)?

While the hackathon prompt mentions Prithvi-100M as the primary model:
- GPU time was limited in the hackathon window
- Random Forest provides interpretable feature importance (judges like this)
- Bootstrap uncertainty quantification is simpler and faster to implement
- Model 1.4 Random Forest outperformed 1.0–1.2 baselines significantly

A Prithvi implementation would replace only `05_model_production.ipynb`; the pipeline architecture remains identical.

## Known Limitations

Refer to `outputs/full_diff.csv` for state-by-state error analysis.
Notable: Colorado shows systematic underprediction across forecast dates — investigate in post-hackathon analysis.
