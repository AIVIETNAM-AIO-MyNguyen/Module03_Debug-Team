"""
Standalone execution script to replicate the Original Paper results and figures:
- "IISE PG&E Energy Analytics Challenge 2025: Hourly-Binned Regression Models Beat Transformers in Load Forecasting"
"""

import os
import sys
import pandas as pd

# Add parent directory to sys.path if running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from replication.config import (
    TEMP_COLS, GHI_COLS, TARGET_COL, OUTPUT_DIR, FIGURES_DIR
)
from replication.data_loader import load_and_preprocess_data
from replication.feature_engineering import (
    PCAExogenousTransformer,
    add_all_lag_lead_configurations,
    get_feature_list_by_config
)
from replication.model_hourly_xgboost import HourlyXGBoostForecaster
from replication.evaluate import (
    compute_metrics,
    evaluate_year1_to_year2,
    evaluate_year2_to_year1,
    evaluate_both_years_cv
)
from replication.visualize import (
    plot_fig1_fig2_load_vs_temp,
    plot_fig3_pca_variance,
    plot_fig4_weekdays_vs_weekends,
    plot_fig6_final_forecast,
    plot_fig7_monthly_forecast
)


def main():
    print("=" * 75)
    print(" REPLICATING ORIGINAL PAPER BASELINE (24-HOURLY XGBOOST)")
    print("=" * 75)

    # 1. Load Data
    train_df, test_df = load_and_preprocess_data(use_cache=True)

    # 2. EDA Figures
    print("\nGenerating EDA Figures (Figure 1, 2, 3, 4)...")
    plot_fig1_fig2_load_vs_temp(train_df, FIGURES_DIR)
    plot_fig3_pca_variance(train_df, TEMP_COLS, GHI_COLS, FIGURES_DIR)
    plot_fig4_weekdays_vs_weekends(train_df, FIGURES_DIR)

    # 3. PCA
    pca_trans = PCAExogenousTransformer()
    train_df = pca_trans.fit_transform(train_df)
    test_df = pca_trans.transform(test_df)

    # 4. Lags & Leads
    train_df = add_all_lag_lead_configurations(train_df)
    test_df = add_all_lag_lead_configurations(test_df)

    # 5. Table 5 Replication: Evaluate Configurations
    configs = ['Baseline', 'Lag1', 'Lag2', 'Lead1']
    records = []

    print("\nEvaluating Original Paper Configurations...")
    for cfg in configs:
        feats = get_feature_list_by_config(cfg)
        m_y1_y2, _ = evaluate_year1_to_year2(train_df, feats, tune_trials=0)
        m_y2_y1, _ = evaluate_year2_to_year1(train_df, feats, tune_trials=0)
        m_cv, _, _ = evaluate_both_years_cv(train_df, feats, n_splits=5, tune_trials=0)

        records.append({
            'Config': cfg,
            'Y1_Y2_RMSE': round(m_y1_y2['RMSE'], 2),
            'Y1_Y2_MAPE': round(m_y1_y2['MAPE'], 2),
            'Y2_Y1_RMSE': round(m_y2_y1['RMSE'], 2),
            'Y2_Y1_MAPE': round(m_y2_y1['MAPE'], 2),
            'Both_CV_RMSE': round(m_cv['RMSE'], 2),
            'Both_CV_MAPE': round(m_cv['MAPE'], 2)
        })

    df_table5 = pd.DataFrame(records)
    print("\n" + "=" * 75)
    print(" REPLICATED TABLE 5 (XGBOOST CONFIGURATIONS)")
    print("=" * 75)
    print(df_table5.to_string(index=False))
    df_table5.to_csv(os.path.join(OUTPUT_DIR, 'table_5_replicated.csv'), index=False)

    # 6. Final Champion Model (Lag1) on Year 3 Test Set
    print("\nTraining Final Paper Champion Model (Lag1) on Full 2-Year Dataset...")
    lag1_feats = get_feature_list_by_config('Lag1')
    forecaster = HourlyXGBoostForecaster()
    forecaster.fit(train_df, lag1_feats)
    y3_preds = forecaster.predict(test_df, lag1_feats)

    m_y3 = compute_metrics(test_df[TARGET_COL].values, y3_preds)
    print(f"\nYear 3 (Test Set) Evaluation Metrics:\n  R2: {m_y3['R2']:.4f}\n  RMSE: {m_y3['RMSE']:.2f} MW\n  MAPE: {m_y3['MAPE']:.2f}%")

    plot_fig6_final_forecast(test_df, y3_preds, FIGURES_DIR)
    plot_fig7_monthly_forecast(test_df, y3_preds, FIGURES_DIR)
    print(f"\nReplication completed. Figures and tables saved in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
