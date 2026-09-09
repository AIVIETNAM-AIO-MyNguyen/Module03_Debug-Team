# Context-Aware Dual-Regime Stacking Ensemble with Physics-Informed Feature Engineering for Hourly Load Forecasting

An advanced extension of the **IISE PG&E Energy Analytics Challenge 2025** baseline, introducing physics-informed feature engineering (HDD/CDD, Fourier harmonics), dual-regime error diversity analysis, and a Context-Aware Stacking (CAS) ensemble architecture.

---

## 📌 Key Research Results (Year 3 Test Set / 2022)

| Model Configuration | Model Family / Architecture | RMSE (MW) ↓ | MAPE ↓ | $R^2$ ↑ | Improvement vs. Paper |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Original Paper Baseline** | 24-Hourly XGBoost (Lag1) | 170.84 | 5.38% | 0.8746 | *Baseline* |
| **Feature-Enhanced Baseline** | 24-Hourly XGBoost (+Fourier +HDD/CDD) | 160.80 | 5.14% | 0.8889 | **−10.04 MW** |
| **Standalone ElasticNet** | Global Linear Regression | 207.46 | 6.92% | 0.8150 | +36.62 MW |
| **Standalone XGBoost** | Global Gradient Boosted Trees | 165.85 | 5.45% | 0.8818 | −4.99 MW |
| **Proposed Context-Aware Stacking (CAS)** | **Dual-Regime [ElasticNet + XGBoost] + Meta-Learner** | **141.81** | **4.75%** | **0.9136** | **−29.03 MW (−17.0%)** |

---

## 🏗️ Methodological Framework & Architecture

```
┌────────────────────────────────────────────────────────┐
│                   EXOGENOUS FEATURES                   │
│   [PCA_Temp, PCA_GHI, Lags, HDD, CDD, Fourier, Hour]   │
└───────────┬────────────────────────────────┬───────────┘
            │                                │
      ┌─────▼─────┐                    ┌─────▼─────┐
      │ ElasticNet│                    │  XGBoost  │
      │ (Linear)  │                    │(Non-Linear│
      └─────┬─────┘                    └─────┬─────┘
            │                                │
            ▼ ŷ_linear                       ▼ ŷ_tree
      ┌────────────────────────────────────────────┐
      │          CONTEXT-AWARE META-LEARNER        │
      │                   (LightGBM)               │
      │                                            │
      │   Inputs: [ŷ_linear, ŷ_tree]               │
      │         + [Hour, HDD, CDD, Fourier, ...]   │
      │                                            │
      │   → Night (00-06): trusts ElasticNet more  │
      │   → Heatwave (CDD>3): trusts XGBoost more  │
      └─────────────────────┬──────────────────────┘
                            │
                            ▼
                    Final Load Forecast
```

### Core Innovations:
1. **Physics-Informed Exogenous Features:**
   - **HDD & CDD ($18^\circ\text{C}$ base):** Captures non-linear HVAC activation thresholds.
   - **Fourier Harmonics:** $\sin/\cos(2\pi \cdot \text{DOY}/365.25)$ for continuous annual seasonality.
2. **Dual-Regime Discovery:**
   - **Nocturnal Baseload (00:00–06:00):** ElasticNet strictly outperforms tree models (131.28 MW vs. 140.60 MW).
   - **Summer Cooling Peaks (CDD > 3°C):** XGBoost dominates linear models (241.83 MW vs. 313.25 MW).
3. **Context-Aware Meta-Learning:** The LightGBM meta-learner dynamically routes trust based on real-time meteorological and temporal context ($r=0.6110$ residual diversity).

---

## 📂 Modular Codebase Organization

```
d:/AIO2026/M03/
├── replication/                  # [ISOLATED] ORIGINAL PAPER REPLICATION PACKAGE
│   ├── config.py                 # Baseline paths and settings
│   ├── data_loader.py            # Raw data loading & calendar dummies
│   ├── feature_engineering.py    # PCA & Lag/Lead features
│   ├── model_hourly_xgboost.py   # 24-Hourly XGBoost forecaster
│   ├── evaluate.py               # Protocol validation metrics
│   ├── visualize.py              # Figures 1-7 reproduction
│   └── run_replication.py        # Single execution command for baseline replication
│
├── src/                          # [EXTENDED] MODULAR RESEARCH PACKAGE
│   ├── config.py                 # Paths, constants, and feature definitions
│   ├── data_loader.py            # Ingestion & calendar features
│   ├── feature_engineering.py    # Degree Days (HDD/CDD), Fourier, PCA, Lags
│   ├── models/                   # Core predictive models
│   │   ├── hourly_xgboost.py     # 24-Hourly XGBoost forecaster
│   │   ├── single_xgboost.py     # Monolithic global XGBoost forecaster
│   │   └── stacking.py           # Context-Aware Stacking (CAS) forecaster
│   ├── evaluation/               # Evaluation protocols & metrics
│   │   └── evaluate.py           # Y1->Y2, Y2->Y1, 5-Fold TS-CV, Year 3 test
│   ├── explainability/           # Explainability algorithms
│   │   └── shap_analysis.py      # 24-Hour SHAP heatmap generator
│   └── visualization/            # Figure plotting modules
│       └── visualize.py          # Figures 6, 7, 8, 9 generators
│
├── run_extended_experiments.py   # MASTER PIPELINE (Executes Stages 1 - 4)
├── README.md                     # Comprehensive documentation
├── Train.xlsx / Test.xlsx        # Raw PG&E dataset
│
└── output/                       # Output artifacts (Predictions, figures, CSVs)
    ├── predictions_xgboost.xlsx  # Final Year 3 predictions (Excel)
    ├── predictions_xgboost.csv   # Final Year 3 predictions (CSV)
    ├── results_master_paper_comparison.csv
    ├── results_residual_correlation.csv
    ├── results_conditional_regime_errors.csv
    └── figures/                  # Publication-ready figures
```

---

## 🚀 How to Run

### 1. Run Master Research Pipeline (Extended Model, Stacking, Figures)
```bash
python run_extended_experiments.py
```

### 2. Replicate Original Paper Baseline Only
```bash
python replication/run_replication.py
```