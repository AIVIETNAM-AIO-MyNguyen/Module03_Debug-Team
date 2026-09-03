"""
Evaluation metrics and test case validation for baseline replication.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from .model_hourly_xgboost import HourlyXGBoostForecaster


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = float(np.mean(np.abs(y_true - y_pred) / denominator) * 100.0)
    return {'R2': r2, 'RMSE': rmse, 'MAPE': mape, 'sMAPE': smape}


def evaluate_year1_to_year2(df: pd.DataFrame, feature_cols: list[str], tune_trials: int = 0):
    y1_df = df[df['Year'] == 1].copy()
    y2_df = df[df['Year'] == 2].copy()
    forecaster = HourlyXGBoostForecaster()
    if tune_trials > 0:
        forecaster.tune_all_hours(y1_df, feature_cols, n_trials=tune_trials, verbose=False)
    forecaster.fit(y1_df, feature_cols)
    preds = forecaster.predict(y2_df, feature_cols)
    return compute_metrics(y2_df['Load'].values, preds), preds


def evaluate_year2_to_year1(df: pd.DataFrame, feature_cols: list[str], tune_trials: int = 0):
    y1_df = df[df['Year'] == 1].copy()
    y2_df = df[df['Year'] == 2].copy()
    forecaster = HourlyXGBoostForecaster()
    if tune_trials > 0:
        forecaster.tune_all_hours(y2_df, feature_cols, n_trials=tune_trials, verbose=False)
    forecaster.fit(y2_df, feature_cols)
    preds = forecaster.predict(y1_df, feature_cols)
    return compute_metrics(y1_df['Load'].values, preds), preds


def evaluate_both_years_cv(df: pd.DataFrame, feature_cols: list[str], n_splits: int = 5, tune_trials: int = 0):
    df_sorted = df.sort_values(['Year', 'Month', 'Day', 'Hour']).reset_index(drop=True)
    n_total = len(df_sorted)
    step = n_total // (n_splits + 1)
    all_trues, all_preds, fold_preds = [], [], []

    for fold in range(1, n_splits + 1):
        train_end = fold * step
        test_end = (fold + 1) * step if fold < n_splits else n_total
        tr_fold = df_sorted.iloc[:train_end]
        te_fold = df_sorted.iloc[train_end:test_end]

        forecaster = HourlyXGBoostForecaster()
        if tune_trials > 0:
            forecaster.tune_all_hours(tr_fold, feature_cols, n_trials=tune_trials, verbose=False)
        forecaster.fit(tr_fold, feature_cols)
        p = forecaster.predict(te_fold, feature_cols)
        all_trues.extend(te_fold['Load'].values)
        all_preds.extend(p)
        fold_preds.append(p)

    return compute_metrics(np.array(all_trues), np.array(all_preds)), np.array(all_preds), fold_preds
