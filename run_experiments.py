"""
Master Replication Script:
Executes the full pipeline to reproduce the original paper's:
- Table 1: Descriptive Statistics
- Table 2 & 3: VIF results before and after PCA
- Table 4: Model comparison results (XGBoost)
- Table 5: XGBoost Improvement Results across Lag/Lead feature configurations
- Final Model Year 3 (Test Set) Forecast & export
- Figures 1, 2, 3, 4, 5, 6, and 7
"""

import os
import sys
import time
import pandas as pd
import numpy as np

from config import (
    TEMP_COLS, GHI_COLS, TARGET_COL, OUTPUT_DIR, FIGURES_DIR,
    TEST_EXCEL
)
from data_loader import load_and_preprocess_data
from feature_engineering import (
    compute_vif, PCAExogenousTransformer,
    add_all_lag_lead_configurations, get_feature_list_by_config
)
from model_hourly_xgboost import HourlyXGBoostForecaster
from evaluate import (
    compute_metrics, evaluate_year1_to_year2,
    evaluate_year2_to_year1, evaluate_both_years_cv
)
from visualize import (
    plot_fig1_fig2_load_vs_temp, plot_fig3_pca_variance,
    plot_fig4_weekdays_vs_weekends, plot_fig5_xgboost_test_cases,
    plot_fig6_final_forecast, plot_fig7_monthly_forecast
)


def print_header(title: str):
    print("\n" + "=" * 75)
    print(f" {title.upper()}")
    print("=" * 75)


def run_descriptive_stats(train_df: pd.DataFrame):
    """
    Computes Table 1: Descriptive Statistics of the Dataset.
    """
    print_header("Table 1: Descriptive Statistics of the Dataset")

    cols_to_stat = ['Load'] + TEMP_COLS + GHI_COLS
    records = []

    for yr in [1, 2]:
        sub = train_df[train_df['Year'] == yr]
        mean_row = {'Metric': 'Mean (mu)', 'Year': f'Year {yr}'}
        std_row = {'Metric': 'Std (sigma)', 'Year': f'Year {yr}'}
        med_row = {'Metric': 'Median', 'Year': f'Year {yr}'}
        min_row = {'Metric': 'Min', 'Year': f'Year {yr}'}
        max_row = {'Metric': 'Max', 'Year': f'Year {yr}'}
        skew_row = {'Metric': 'Skew (gamma1)', 'Year': f'Year {yr}'}
        kurt_row = {'Metric': 'Kurt (gamma2)', 'Year': f'Year {yr}'}

        for c in cols_to_stat:
            s = sub[c]
            mean_row[c] = round(float(s.mean()), 2)
            std_row[c] = round(float(s.std()), 2)
            med_row[c] = round(float(s.median()), 2)
            min_row[c] = round(float(s.min()), 2)
            max_row[c] = round(float(s.max()), 2)
            skew_row[c] = round(float(s.skew()), 2)
            kurt_row[c] = round(float(s.kurt()), 2)

        records.extend([mean_row, std_row, med_row, min_row, max_row, skew_row, kurt_row])

    stats_df = pd.DataFrame(records)
    print(stats_df[['Metric', 'Year', 'Load', 'Site-1 Temp', 'Site-5 Temp', 'Site-1 GHI', 'Site-5 GHI']].to_string(index=False))
    return stats_df


def run_vif_analysis(train_df: pd.DataFrame):
    """
    Computes Table 2 (VIF before PCA) and Table 3 (VIF after PCA).
    """
    print_header("Table 2: Variance Inflation Factor (VIF) Results Before PCA")
    vif_temp = compute_vif(train_df, TEMP_COLS + [TARGET_COL])
    vif_ghi = compute_vif(train_df, GHI_COLS + [TARGET_COL])

    print("\n--- Temperature Features VIF ---")
    print(vif_temp.to_string(index=False))

    print("\n--- GHI Features VIF ---")
    print(vif_ghi.to_string(index=False))

    print_header("Table 3: Variance Inflation Factor (VIF) Results After PCA")
    vif_pca = compute_vif(train_df, ['PCA_Temp', 'PCA_GHI', TARGET_COL])
    print(vif_pca.to_string(index=False))

    vif_temp.to_csv(os.path.join(OUTPUT_DIR, 'table2_vif_temp_before_pca.csv'), index=False)
    vif_ghi.to_csv(os.path.join(OUTPUT_DIR, 'table2_vif_ghi_before_pca.csv'), index=False)
    vif_pca.to_csv(os.path.join(OUTPUT_DIR, 'table3_vif_after_pca.csv'), index=False)


def run_lag_lead_experiments(train_df: pd.DataFrame, n_tune_trials: int = 10, use_cache: bool = True):
    """
    Computes Table 5: XGBoost Improvement Results across all Lag/Lead configurations.
    """
    print_header("Table 5: XGBoost Improvement Results (Lag and Lead Experiments)")
    csv_path = os.path.join(OUTPUT_DIR, 'results_table5_xgboost_improvements.csv')

    if use_cache and os.path.exists(csv_path):
        print(f"Loading existing Table 5 results from: {csv_path}")
        df_table5 = pd.read_csv(csv_path)
        print(df_table5.to_string(index=False))
        return df_table5

    configs = [
        'Baseline', 'Lag1', 'Lag2', 'Lead1', 'Lag5',
        'Lag1+Lead1', 'Lag3', 'Lag2+Lead2', 'Lead2',
        'Lag3+Lead2', 'Lag5+Lead2'
    ]

    table5_results = []

    for cfg in configs:
        feat_cols = get_feature_list_by_config(cfg)
        print(f"\nEvaluating configuration: [{cfg}] (Total Features: {len(feat_cols)})...")

        # Test Case 1: Year 1 -> Year 2
        m_y1_y2, _ = evaluate_year1_to_year2(train_df, feat_cols, tune_trials=n_tune_trials if cfg in ['Baseline', 'Lag1'] else 0)

        # Test Case 2: Year 2 -> Year 1
        m_y2_y1, _ = evaluate_year2_to_year1(train_df, feat_cols, tune_trials=n_tune_trials if cfg in ['Baseline', 'Lag1'] else 0)

        # Test Case 3: Both Years CV
        m_cv, _, _ = evaluate_both_years_cv(train_df, feat_cols, n_splits=5, tune_trials=0)

        print(f"  Year 1 -> Year 2: R2={m_y1_y2['R2']:.2f}, RMSE={m_y1_y2['RMSE']:.2f}, MAPE={m_y1_y2['MAPE']:.2f}%, sMAPE={m_y1_y2['sMAPE']:.2f}%")
        print(f"  Year 2 -> Year 1: R2={m_y2_y1['R2']:.2f}, RMSE={m_y2_y1['RMSE']:.2f}, MAPE={m_y2_y1['MAPE']:.2f}%, sMAPE={m_y2_y1['sMAPE']:.2f}%")
        print(f"  Both Years (CV) : R2={m_cv['R2']:.2f}, RMSE={m_cv['RMSE']:.2f}, MAPE={m_cv['MAPE']:.2f}%, sMAPE={m_cv['sMAPE']:.2f}%")

        table5_results.append({
            'Model_Config': cfg,
            'Y1_Y2_R2': round(m_y1_y2['R2'], 2),
            'Y1_Y2_RMSE': round(m_y1_y2['RMSE'], 2),
            'Y1_Y2_MAPE': round(m_y1_y2['MAPE'], 2),
            'Y1_Y2_sMAPE': round(m_y1_y2['sMAPE'], 2),
            'Y2_Y1_R2': round(m_y2_y1['R2'], 2),
            'Y2_Y1_RMSE': round(m_y2_y1['RMSE'], 2),
            'Y2_Y1_MAPE': round(m_y2_y1['MAPE'], 2),
            'Y2_Y1_sMAPE': round(m_y2_y1['sMAPE'], 2),
            'Both_R2': round(m_cv['R2'], 2),
            'Both_RMSE': round(m_cv['RMSE'], 2),
            'Both_MAPE': round(m_cv['MAPE'], 2),
            'Both_sMAPE': round(m_cv['sMAPE'], 2),
        })

    df_table5 = pd.DataFrame(table5_results)
    df_table5.to_csv(csv_path, index=False)
    print(f"\nTable 5 results saved to: {csv_path}")

    return df_table5



def train_final_model_and_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    n_tune_trials: int = 20
) -> tuple[np.ndarray, HourlyXGBoostForecaster]:
    """
    Trains the final Lag1 XGBoost hourly forecaster on the full 2-year training dataset
    and generates Year 3 predictions for the test dataset.
    """
    print_header("Training Final XGBoost Model (Lag1 Configuration) on Full 2-Year Dataset")

    forecaster = HourlyXGBoostForecaster()
    forecaster.tune_all_hours(train_df, feature_cols, target_col=TARGET_COL, n_trials=n_tune_trials, verbose=True)
    forecaster.fit(train_df, feature_cols, target_col=TARGET_COL)

    print("\nGenerating Year 3 (Test Set) load predictions...")
    y3_preds = forecaster.predict(test_df, feature_cols)

    # Save predictions to output directory
    pred_df = test_df[['Year', 'Real_Year', 'Month', 'Day', 'Hour']].copy()
    pred_df['Predicted_Load'] = y3_preds

    if TARGET_COL in test_df.columns:
        pred_df['Actual_Load'] = test_df[TARGET_COL]
        # Evaluate metrics if ground truth exists
        test_metrics = compute_metrics(test_df[TARGET_COL].values, y3_preds)
        print_header("Year 3 (Test Dataset) Evaluation Metrics")
        print(f"  R2   : {test_metrics['R2']:.4f}")
        print(f"  RMSE : {test_metrics['RMSE']:.2f}")
        print(f"  MAPE : {test_metrics['MAPE']:.2f}%")
        print(f"  sMAPE: {test_metrics['sMAPE']:.2f}%")

    excel_path = os.path.join(OUTPUT_DIR, 'predictions_xgboost.xlsx')
    csv_path = os.path.join(OUTPUT_DIR, 'predictions_xgboost.csv')
    pred_df.to_excel(excel_path, index=False)
    pred_df.to_csv(csv_path, index=False)
    print(f"\nFinal predictions successfully exported to:\n  - {excel_path}\n  - {csv_path}")

    # Feature importance
    feat_imp = forecaster.get_feature_importances(feature_cols)
    print("\nTop 10 Important Features across 24 Hourly Models:")
    print(feat_imp.head(10).to_string(index=False))

    return y3_preds, forecaster


def main():
    start_time = time.time()
    print_header("Replicating: Hourly-Binned Regression Models Beat Transformers in Load Forecasting")

    # 1. Load data
    print("Step 1: Loading raw datasets and constructing calendar features...")
    train_df, test_df = load_and_preprocess_data(use_cache=True)
    print(f"Train dataset: {len(train_df)} hourly records ({train_df['Year'].nunique()} years)")
    print(f"Test dataset : {len(test_df)} hourly records ({test_df['Year'].nunique()} year)")

    # 2. Descriptive statistics (Table 1)
    run_descriptive_stats(train_df)

    # 3. PCA Feature Engineering
    print("\nStep 2: Fitting PCA on Temperature and GHI exogenous variables...")
    pca_trans = PCAExogenousTransformer()
    train_df = pca_trans.fit_transform(train_df)
    test_df = pca_trans.transform(test_df)
    var_exp = pca_trans.get_explained_variance()
    print(f"  PCA Temperature Explained Variance Ratio: {var_exp['PCA_Temp_Ratio']:.4f} (~{var_exp['PCA_Temp_Ratio']*100:.1f}%)")
    print(f"  PCA GHI Explained Variance Ratio        : {var_exp['PCA_GHI_Ratio']:.4f} (~{var_exp['PCA_GHI_Ratio']*100:.1f}%)")

    # 4. Compute VIF (Table 2 & Table 3)
    run_vif_analysis(train_df)

    # 5. Add all lag/lead configurations
    print("\nStep 3: Generating Exogenous Lag and Lead Features...")
    train_df = add_all_lag_lead_configurations(train_df)
    test_df = add_all_lag_lead_configurations(test_df)

    # 6. Generate Figures 1, 2, 3, 4
    print("\nStep 4: Generating Exploratory Analysis Figures (Fig 1, 2, 3, 4)...")
    plot_fig1_fig2_load_vs_temp(train_df, FIGURES_DIR)
    plot_fig3_pca_variance(train_df, TEMP_COLS, GHI_COLS, FIGURES_DIR)
    plot_fig4_weekdays_vs_weekends(train_df, FIGURES_DIR)

    # 7. Run Lag/Lead Experiments (Table 5)
    print("\nStep 5: Running Systematic Lag/Lead Experiments (Table 5)...")
    run_lag_lead_experiments(train_df, n_tune_trials=10)

    # 8. Detailed Test Cases Evaluation for Figure 5 (Lag1 model)
    print("\nStep 6: Running Detailed Test Cases for Figure 5 (Lag1 XGBoost)...")
    final_features = get_feature_list_by_config('Lag1')
    _, y2_preds = evaluate_year1_to_year2(train_df, final_features, tune_trials=15)
    _, y1_preds = evaluate_year2_to_year1(train_df, final_features, tune_trials=15)
    _, cv_preds, _ = evaluate_both_years_cv(train_df, final_features, n_splits=5, tune_trials=0)

    y1_df = train_df[train_df['Year'] == 1]
    y2_df = train_df[train_df['Year'] == 2]
    # For CV actuals:
    df_sorted = train_df.sort_values(['Year', 'Month', 'Day', 'Hour']).reset_index(drop=True)
    n_cv = len(cv_preds)
    cv_trues = df_sorted.iloc[-n_cv:]['Load'].values

    plot_fig5_xgboost_test_cases(y1_df, y2_df, y2_preds, y1_preds, cv_trues, cv_preds, FIGURES_DIR)

    # Save Table 4 comparison for XGBoost
    m_y1_y2 = compute_metrics(y2_df['Load'].values, y2_preds)
    m_y2_y1 = compute_metrics(y1_df['Load'].values, y1_preds)
    m_cv = compute_metrics(cv_trues, cv_preds)
    table4_df = pd.DataFrame([
        {'Model': 'XGBoost', 'Test_Case': 'Year 1 -> Year 2', 'R2': round(m_y1_y2['R2'], 2), 'RMSE': round(m_y1_y2['RMSE'], 1), 'MAPE': round(m_y1_y2['MAPE'], 1), 'sMAPE': round(m_y1_y2['sMAPE'], 1)},
        {'Model': 'XGBoost', 'Test_Case': 'Year 2 -> Year 1', 'R2': round(m_y2_y1['R2'], 2), 'RMSE': round(m_y2_y1['RMSE'], 1), 'MAPE': round(m_y2_y1['MAPE'], 1), 'sMAPE': round(m_y2_y1['sMAPE'], 1)},
        {'Model': 'XGBoost', 'Test_Case': 'Both years (CV)', 'R2': round(m_cv['R2'], 2), 'RMSE': round(m_cv['RMSE'], 1), 'MAPE': round(m_cv['MAPE'], 1), 'sMAPE': round(m_cv['sMAPE'], 1)},
    ])
    table4_df.to_csv(os.path.join(OUTPUT_DIR, 'results_table4_xgboost.csv'), index=False)
    print("\nTable 4 (XGBoost Metrics):")
    print(table4_df.to_string(index=False))

    # 9. Train Final Model & Forecast Year 3 (Test Set)
    print("\nStep 7: Training Final Model on 2-Year Dataset & Forecasting Year 3...")
    y3_preds, forecaster = train_final_model_and_forecast(train_df, test_df, final_features, n_tune_trials=20)

    # 10. Generate Figures 6 & 7 (Final Forecasts)
    print("\nStep 8: Generating Final Forecast Visualizations (Fig 6, 7)...")
    plot_fig6_final_forecast(test_df, y3_preds, FIGURES_DIR)
    plot_fig7_monthly_forecast(test_df, y3_preds, FIGURES_DIR)

    elapsed = time.time() - start_time
    print_header(f"Replication Complete in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)!")


if __name__ == '__main__':
    main()
