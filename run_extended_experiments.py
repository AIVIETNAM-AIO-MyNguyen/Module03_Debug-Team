"""
Master Research Pipeline for:
"Context-Aware Dual-Regime Stacking Ensemble with Physics-Informed Feature Engineering for Hourly Load Forecasting"

Executes all empirical research stages:
1. Physics-Informed Feature Engineering (HDD/CDD, Fourier harmonics).
2. Systematic Model Comparison (Original Paper Features vs. Our Added Features vs. Stacking).
3. Error Diversity (Residual Correlation Matrix, Figure 9).
4. Conditional Error Decomposition across Regimes (Figure 8).
5. 24-Hour Diurnal SHAP Explainability Heatmap (Figure 10).
6. Final Predictions Export (Excel/CSV) and Forecast Curves (Figures 6 & 7).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
import xgboost as xgb

from src.config import (
    TEMP_COLS, GHI_COLS, TARGET_COL, OUTPUT_DIR, FIGURES_DIR
)
from src.data_loader import load_and_preprocess_data
from src.feature_engineering import (
    PCAExogenousTransformer,
    add_all_lag_lead_configurations,
    get_original_paper_features,
    get_extended_features,
    add_fourier_features,
    add_hdd_cdd_features
)
from src.models import (
    HourlyXGBoostForecaster,
    SingleXGBoostForecaster,
    ContextAwareStackingForecaster
)
from src.evaluation import (
    compute_metrics,
    evaluate_year1_to_year2,
    evaluate_year2_to_year1,
    evaluate_year3_test
)
from src.visualization import (
    setup_style,
    plot_final_forecast,
    plot_monthly_forecast,
    plot_conditional_regime_errors,
    plot_residual_correlation_matrix
)
from src.explainability import generate_shap_heatmap


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)


def stage1_feature_engineering(train_df: pd.DataFrame, test_df: pd.DataFrame):
    print_header("Stage 1: Physics-Informed Feature Engineering & Preprocessing")

    print("Fitting PCA on 5 Temperature and 5 GHI sites...")
    pca_trans = PCAExogenousTransformer()
    train_df = pca_trans.fit_transform(train_df)
    test_df = pca_trans.transform(test_df)

    print("Adding original paper exogenous lag and lead configurations...")
    train_df = add_all_lag_lead_configurations(train_df)
    test_df = add_all_lag_lead_configurations(test_df)

    print("Adding our continuous yearly & daily Fourier harmonics...")
    train_df = add_fourier_features(train_df)
    test_df = add_fourier_features(test_df)

    print("Adding our Heating Degree Days (HDD) & Cooling Degree Days (CDD) at 18°C base...")
    train_df = add_hdd_cdd_features(train_df, base_temp=18.0)
    test_df = add_hdd_cdd_features(test_df, base_temp=18.0)

    print(f"Engineered Train Shape: {train_df.shape}, Test Shape: {test_df.shape}")
    return train_df, test_df


def stage2_error_diversity_and_regimes(train_df: pd.DataFrame, test_df: pd.DataFrame):
    print_header("Stage 2: Error Diversity & Conditional Regime Analysis (Motivating CAS)")

    feature_cols = get_extended_features('Lag1', include_fourier=True, include_degree_days=True)
    all_features = ['Hour'] + feature_cols

    X_train = train_df[all_features].values
    y_train = train_df[TARGET_COL].values
    X_test = test_df[all_features].values
    y_test = test_df[TARGET_COL].values

    # Fit standalone base models
    print("Fitting standalone base models: ElasticNet (Linear) and XGBoost (Tree)...")
    enet = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000).fit(X_train, y_train)
    xgb_m = xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1).fit(X_train, y_train)
    
    # Fit Proposed 2-model CAS
    print("Fitting Proposed Context-Aware Stacking (CAS) to synthesize both regimes...")
    cas_model = ContextAwareStackingForecaster(random_state=42)
    cas_model.fit(train_df, feature_cols, target_col=TARGET_COL)

    pred_enet = enet.predict(X_test)
    pred_xgb = xgb_m.predict(X_test)
    pred_cas = cas_model.predict(test_df, feature_cols)

    # 1. Residual correlation
    res_df = pd.DataFrame({
        'ElasticNet': y_test - pred_enet,
        'XGBoost': y_test - pred_xgb
    })
    corr_matrix = res_df.corr(method='pearson')
    corr_csv = os.path.join(OUTPUT_DIR, 'results_residual_correlation.csv')
    corr_matrix.round(4).to_csv(corr_csv)
    plot_residual_correlation_matrix(corr_matrix, FIGURES_DIR)

    # 2. Conditional Regime Breakdown
    test_analysis = test_df.copy()
    test_analysis['Actual_Load'] = y_test
    test_analysis['Pred_ElasticNet'] = pred_enet
    test_analysis['Pred_XGBoost'] = pred_xgb
    test_analysis['Pred_CAS'] = pred_cas

    regimes = {
        'High CDD (Summer Peaks, CDD > 3°C)': test_analysis['CDD'] > 3.0,
        'High HDD (Winter Heating, HDD > 5°C)': test_analysis['HDD'] > 5.0,
        'Mild / Neutral Weather (HDD<=1 & CDD<=1)': (test_analysis['HDD'] <= 1.0) & (test_analysis['CDD'] <= 1.0),
        'Peak Operating Hours (15:00 - 19:00)': test_analysis['Hour'].isin([15, 16, 17, 18, 19]),
        'Off-Peak Night Hours (00:00 - 06:00)': test_analysis['Hour'].isin([1, 2, 3, 4, 5, 6]),
        'Mid-Day Solar Hours (10:00 - 14:00)': test_analysis['Hour'].isin([10, 11, 12, 13, 14]),
        'Weekdays (Mon - Fri)': test_analysis['Is_Weekend'] == 0,
        'Weekends & Holidays': (test_analysis['Is_Weekend'] == 1) | (test_analysis['Is_Holiday'] == 1),
    }

    reg_records = []
    for r_name, mask in regimes.items():
        sub = test_analysis[mask]
        y_sub = sub['Actual_Load'].values
        m_e = compute_metrics(y_sub, sub['Pred_ElasticNet'].values)
        m_x = compute_metrics(y_sub, sub['Pred_XGBoost'].values)
        m_c = compute_metrics(y_sub, sub['Pred_CAS'].values)
        best_base = 'ElasticNet' if m_e['RMSE'] < m_x['RMSE'] else 'XGBoost'

        reg_records.append({
            'Regime': r_name,
            'Sample_Count': len(sub),
            'ElasticNet_RMSE': round(m_e['RMSE'], 2),
            'XGBoost_RMSE': round(m_x['RMSE'], 2),
            'Best_Base_Model': best_base,
            'CAS_Ensemble_RMSE': round(m_c['RMSE'], 2),
            'Ensemble_Gain_MW': round(min(m_e['RMSE'], m_x['RMSE']) - m_c['RMSE'], 2)
        })

    df_reg = pd.DataFrame(reg_records)
    reg_csv = os.path.join(OUTPUT_DIR, 'results_conditional_regime_errors.csv')
    df_reg.to_csv(reg_csv, index=False)
    plot_conditional_regime_errors(df_reg, FIGURES_DIR)

    print("Diversity & Regime Analysis successfully completed and figures saved.")
    return pred_cas


def stage3_systematic_model_comparison(train_df: pd.DataFrame, test_df: pd.DataFrame):
    print_header("Stage 3: Master Model Architecture & Feature Ablation Benchmark")

    # Distinct Feature Sets: Original Paper vs. Extended
    orig_feats_lag1 = get_original_paper_features('Lag1')                                    # 18 features
    feats_with_fourier = get_extended_features('Lag1', include_fourier=True, include_degree_days=False) # 22 features
    feats_full = get_extended_features('Lag1', include_fourier=True, include_degree_days=True)          # 24 features

    experiments = [
        {'name': '1. Original Paper Baseline (24-XGBoost, Original Features)', 'model_type': 'hourly', 'features': orig_feats_lag1, 'feat_count': len(orig_feats_lag1)},
        {'name': '2. 24-XGBoost + Fourier (Our Added Seasonal Features)',      'model_type': 'hourly', 'features': feats_with_fourier, 'feat_count': len(feats_with_fourier)},
        {'name': '3. 24-XGBoost + Fourier + HDD/CDD (Our Full Feature Set)',   'model_type': 'hourly', 'features': feats_full, 'feat_count': len(feats_full)},
        {'name': '4. Single-XGBoost (Original Features + Hour)',               'model_type': 'single', 'features': ['Hour'] + orig_feats_lag1, 'feat_count': len(orig_feats_lag1) + 1},
        {'name': '5. Proposed Context-Aware Stacking (CAS, Our Full Set)',      'model_type': 'cas',    'features': feats_full, 'feat_count': len(feats_full)},
    ]

    results = []
    for exp in experiments:
        name = exp['name']
        m_type = exp['model_type']
        feats = exp['features']
        f_count = exp['feat_count']
        print(f"\nEvaluating [{name}] (Feature Count: {f_count})...")

        m_y1_y2, _ = evaluate_year1_to_year2(train_df, feats, tune_trials=0, model_type=m_type)
        m_y2_y1, _ = evaluate_year2_to_year1(train_df, feats, tune_trials=0, model_type=m_type)
        m_y3, _, _ = evaluate_year3_test(train_df, test_df, feats, tune_trials=0, model_type=m_type)

        print(f"  Y1->Y2: RMSE={m_y1_y2['RMSE']:.2f}, MAE={m_y1_y2['MAE']:.2f}, MAPE={m_y1_y2['MAPE']:.2f}% | Year 3: R2={m_y3['R2']:.4f}, RMSE={m_y3['RMSE']:.2f}, MAE={m_y3['MAE']:.2f}, MAPE={m_y3['MAPE']:.2f}%")

        results.append({
            'Model_Configuration': name,
            'Feature_Count': f_count,
            'Y1_Y2_R2': round(m_y1_y2['R2'], 4),
            'Y1_Y2_RMSE': round(m_y1_y2['RMSE'], 2),
            'Y1_Y2_MAE': round(m_y1_y2['MAE'], 2),
            'Y1_Y2_MAPE': round(m_y1_y2['MAPE'], 2),
            'Y2_Y1_R2': round(m_y2_y1['R2'], 4),
            'Y2_Y1_RMSE': round(m_y2_y1['RMSE'], 2),
            'Y2_Y1_MAE': round(m_y2_y1['MAE'], 2),
            'Y2_Y1_MAPE': round(m_y2_y1['MAPE'], 2),
            'Year3_R2': round(m_y3['R2'], 4),
            'Year3_RMSE': round(m_y3['RMSE'], 2),
            'Year3_MAE': round(m_y3['MAE'], 2),
            'Year3_MAPE': round(m_y3['MAPE'], 2),
            'Year3_sMAPE': round(m_y3['sMAPE'], 2),
        })

    df_results = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, 'results_master_paper_comparison.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\nMaster comparison saved to: {csv_path}")

    print("\n" + "=" * 95)
    print(" 1. PRIMARY RESEARCH BENCHMARK: UNSEEN TEST SET (YEAR 3)")
    print("=" * 95)
    print(df_results[['Model_Configuration', 'Feature_Count', 'Year3_R2', 'Year3_RMSE', 'Year3_MAE', 'Year3_MAPE', 'Year3_sMAPE']].to_string(index=False))

    print("\n" + "=" * 95)
    print(" 2. CROSS-YEAR GENERALIZATION (YEAR 1 <-> YEAR 2)")
    print("=" * 95)
    print(df_results[['Model_Configuration', 'Y1_Y2_R2', 'Y1_Y2_RMSE', 'Y1_Y2_MAE', 'Y1_Y2_MAPE', 'Y2_Y1_R2', 'Y2_Y1_RMSE', 'Y2_Y1_MAE', 'Y2_Y1_MAPE']].to_string(index=False))
    return df_results


def stage4_shap_and_final_forecasts(train_df: pd.DataFrame, test_df: pd.DataFrame, y3_preds: np.ndarray):
    print_header("Stage 4: SHAP Explainability & Final Forecast Generation")

    lag1_full = get_extended_features('Lag1', include_fourier=True, include_degree_days=True)

    # 1. SHAP Heatmap
    hourly_forecaster = HourlyXGBoostForecaster()
    hourly_forecaster.fit(train_df, lag1_full)
    generate_shap_heatmap(hourly_forecaster, train_df, lag1_full, FIGURES_DIR)

    # 2. Export predictions
    pred_df = test_df[['Year', 'Real_Year', 'Month', 'Day', 'Hour']].copy()
    pred_df['Predicted_Load'] = y3_preds
    if TARGET_COL in test_df.columns:
        pred_df['Actual_Load'] = test_df[TARGET_COL]

    pred_df.to_excel(os.path.join(OUTPUT_DIR, 'predictions_xgboost.xlsx'), index=False)
    pred_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions_xgboost.csv'), index=False)

    # 3. Forecast figures
    plot_final_forecast(test_df, y3_preds, FIGURES_DIR)
    plot_monthly_forecast(test_df, y3_preds, FIGURES_DIR)
    print("Final predictions and all publication figures successfully generated.")


def main():
    start_time = time.time()
    print_header("Starting End-to-End Load Forecasting Research Pipeline")

    raw_train, raw_test = load_and_preprocess_data(use_cache=True)
    train_df, test_df = stage1_feature_engineering(raw_train, raw_test)
    y3_preds_cas = stage2_error_diversity_and_regimes(train_df, test_df)
    stage3_systematic_model_comparison(train_df, test_df)
    stage4_shap_and_final_forecasts(train_df, test_df, y3_preds_cas)

    elapsed = time.time() - start_time
    print_header(f"End-to-End Pipeline Completed Successfully in {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)!")


if __name__ == '__main__':
    main()
