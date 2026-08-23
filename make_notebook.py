"""
Generate Jupyter Notebook replicate_paper.ipynb.
"""

import json

notebook = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# Replication: IISE PG&E Energy Analytics Challenge 2025\n',
                '## *Hourly-Binned Regression Models Beat Transformers in Load Forecasting*\n',
                '**Authors:** Millend Roy, Vladimir Pyltsov, Yinbo Hu (Columbia University)\n\n',
                '**Replication Scope:** 24 Hourly-Binned XGBoost Regression Models with PCA Exogenous Feature Engineering and Lag/Lead Experiments.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'import os\n',
                'import numpy as np\n',
                'import pandas as pd\n',
                'import matplotlib.pyplot as plt\n',
                'import xgboost as xgb\n',
                'import optuna\n',
                '\n',
                'from config import TEMP_COLS, GHI_COLS, TARGET_COL, OUTPUT_DIR, FIGURES_DIR\n',
                'from data_loader import load_and_preprocess_data\n',
                'from feature_engineering import (\n',
                '    compute_vif, PCAExogenousTransformer,\n',
                '    add_all_lag_lead_configurations, get_feature_list_by_config\n',
                ')\n',
                'from model_hourly_xgboost import HourlyXGBoostForecaster\n',
                'from evaluate import (\n',
                '    compute_metrics, evaluate_year1_to_year2,\n',
                '    evaluate_year2_to_year1, evaluate_both_years_cv\n',
                ')\n',
                'from visualize import (\n',
                '    plot_fig1_fig2_load_vs_temp, plot_fig3_pca_variance,\n',
                '    plot_fig4_weekdays_vs_weekends, plot_fig5_xgboost_test_cases,\n',
                '    plot_fig6_final_forecast, plot_fig7_monthly_forecast\n',
                ')\n',
                '\n',
                'print("Environment initialized successfully.")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '### 1. Load Data & Calendar Feature Reconstruction\n',
                '- Year 1: 2020 (Leap year starting on Wednesday Jan 1)\n',
                '- Year 2: 2021\n',
                '- Year 3: 2022 (Test Set)\n',
                '- Adds 12 monthly dummies, weekend indicator, and US federal holiday indicator.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'train_df, test_df = load_and_preprocess_data(use_cache=True)\n',
                'print(f"Train shape: {train_df.shape}")\n',
                'print(f"Test shape : {test_df.shape}")\n',
                'train_df.head(3)'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '### 2. Descriptive Statistics (Table 1 in Paper)'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'cols = [\'Load\'] + TEMP_COLS + GHI_COLS\n',
                'stats_records = []\n',
                'for yr in [1, 2]:\n',
                '    sub = train_df[train_df[\'Year\'] == yr]\n',
                '    for name, func in [(\'Mean (mu)\', \'mean\'), (\'Std (sigma)\', \'std\'), (\'Median\', \'median\'), (\'Min\', \'min\'), (\'Max\', \'max\'), (\'Skew (gamma1)\', \'skew\'), (\'Kurt (gamma2)\', \'kurt\')]:\n',
                '        row = {\'Metric\': name, \'Year\': f\'Year {yr}\'}\n',
                '        for c in cols:\n',
                '            val = getattr(sub[c], func)()\n',
                '            row[c] = round(float(val), 2)\n',
                '        stats_records.append(row)\n',
                'pd.DataFrame(stats_records)'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '### 3. PCA Dimensionality Reduction & VIF Multicollinearity Analysis\n',
                '- Temperature (5 sites) -> `PCA_Temp` (>97% variance)\n',
                '- GHI (5 sites) -> `PCA_GHI` (>99% variance)\n',
                '- Replicates Table 2 & Table 3.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'pca_trans = PCAExogenousTransformer()\n',
                'train_df = pca_trans.fit_transform(train_df)\n',
                'test_df = pca_trans.transform(test_df)\n',
                'print("Explained Variance:", pca_trans.get_explained_variance())\n',
                '\n',
                'print("=== Table 2: VIF Before PCA (Temperature) ===")\n',
                'display(compute_vif(train_df, TEMP_COLS + [TARGET_COL]))\n',
                '\n',
                'print("=== Table 2: VIF Before PCA (GHI) ===")\n',
                'display(compute_vif(train_df, GHI_COLS + [TARGET_COL]))\n',
                '\n',
                'print("=== Table 3: VIF After PCA ===")\n',
                'display(compute_vif(train_df, [\'PCA_Temp\', \'PCA_GHI\', TARGET_COL]))'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '### 4. Exploratory Visualizations (Figures 1, 2, 3, 4)'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'plot_fig1_fig2_load_vs_temp(train_df, FIGURES_DIR)\n',
                'plot_fig3_pca_variance(train_df, TEMP_COLS, GHI_COLS, FIGURES_DIR)\n',
                'plot_fig4_weekdays_vs_weekends(train_df, FIGURES_DIR)'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '### 5. Exogenous Lag & Lead Feature Engineering (Table 5)'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'train_df = add_all_lag_lead_configurations(train_df)\n',
                'test_df = add_all_lag_lead_configurations(test_df)\n',
                'print("Augmented Features Count:", len(train_df.columns))\n',
                'train_df[[\'Date\', \'Hour\', \'PCA_Temp\', \'PCA_Temp_Lag1\', \'PCA_GHI\', \'PCA_GHI_Lag1\']].head(3)'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '### 6. XGBoost 24-Hourly Model Experiments (Table 4 & Table 5)'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'from run_experiments import run_lag_lead_experiments\n',
                'table5_df = run_lag_lead_experiments(train_df, n_tune_trials=10)\n',
                'table5_df'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '### 7. Figure 5: XGBoost Model Test Cases'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'final_features = get_feature_list_by_config(\'Lag1\')\n',
                '_, y2_preds = evaluate_year1_to_year2(train_df, final_features, tune_trials=10)\n',
                '_, y1_preds = evaluate_year2_to_year1(train_df, final_features, tune_trials=10)\n',
                '_, cv_preds, _ = evaluate_both_years_cv(train_df, final_features, n_splits=5, tune_trials=0)\n',
                '\n',
                'y1_df = train_df[train_df[\'Year\'] == 1]\n',
                'y2_df = train_df[train_df[\'Year\'] == 2]\n',
                'df_sorted = train_df.sort_values([\'Year\', \'Month\', \'Day\', \'Hour\']).reset_index(drop=True)\n',
                'n_cv = len(cv_preds)\n',
                'cv_trues = df_sorted.iloc[-n_cv:][\'Load\'].values\n',
                '\n',
                'plot_fig5_xgboost_test_cases(y1_df, y2_df, y2_preds, y1_preds, cv_trues, cv_preds, FIGURES_DIR)'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '### 8. Train Final Model on Full 2-Year Dataset & Forecast Year 3 (Figures 6 & 7)'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'from run_experiments import train_final_model_and_forecast\n',
                'y3_preds, forecaster = train_final_model_and_forecast(train_df, test_df, final_features, n_tune_trials=15)\n',
                'plot_fig6_final_forecast(test_df, y3_preds, FIGURES_DIR)\n',
                'plot_fig7_monthly_forecast(test_df, y3_preds, FIGURES_DIR)'
            ]
        }
    ],
    'metadata': {
        'language_info': {
            'name': 'python'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 2
}

with open('replicate_paper.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("replicate_paper.ipynb successfully generated!")
