"""
Evaluation metrics and experimental test case validation module.
Formulas matching Section 4.3 (Equations 4, 5, 6, 7).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from model_hourly_xgboost import HourlyXGBoostForecaster


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute R^2, RMSE, MAPE (%), and sMAPE (%) matching formulas (4)-(7).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # R^2
    r2 = float(r2_score(y_true, y_pred))

    # RMSE
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # MAPE
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)

    # sMAPE
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = float(np.mean(np.abs(y_true - y_pred) / denominator) * 100.0)

    return {
        'R2': r2,
        'RMSE': rmse,
        'MAPE': mape,
        'sMAPE': smape
    }


def evaluate_year1_to_year2(
    df: pd.DataFrame,
    feature_cols: list[str],
    tune_trials: int = 0
) -> tuple[dict[str, float], np.ndarray]:
    """
    Test Case 1: Train on Year 1 (2020), predict and evaluate on Year 2 (2021).
    """
    y1_df = df[df['Year'] == 1].copy()
    y2_df = df[df['Year'] == 2].copy()

    forecaster = HourlyXGBoostForecaster()
    if tune_trials > 0:
        forecaster.tune_all_hours(y1_df, feature_cols, n_trials=tune_trials, verbose=False)

    forecaster.fit(y1_df, feature_cols)
    y2_preds = forecaster.predict(y2_df, feature_cols)
    metrics = compute_metrics(y2_df['Load'].values, y2_preds)

    return metrics, y2_preds


def evaluate_year2_to_year1(
    df: pd.DataFrame,
    feature_cols: list[str],
    tune_trials: int = 0
) -> tuple[dict[str, float], np.ndarray]:
    """
    Test Case 2: Train on Year 2 (2021), predict and evaluate on Year 1 (2020).
    """
    y1_df = df[df['Year'] == 1].copy()
    y2_df = df[df['Year'] == 2].copy()

    forecaster = HourlyXGBoostForecaster()
    if tune_trials > 0:
        forecaster.tune_all_hours(y2_df, feature_cols, n_trials=tune_trials, verbose=False)

    forecaster.fit(y2_df, feature_cols)
    y1_preds = forecaster.predict(y1_df, feature_cols)
    metrics = compute_metrics(y1_df['Load'].values, y1_preds)

    return metrics, y1_preds


def evaluate_both_years_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
    tune_trials: int = 0
) -> tuple[dict[str, float], np.ndarray, list[np.ndarray]]:
    """
    Test Case 3: 5-Fold expanding window Time-Series Cross Validation across entire training dataset.
    Folds:
      Fold 1: Train 0..1/6, Test 1/6..2/6
      Fold 2: Train 0..2/6, Test 2/6..3/6
      ...
      Fold 5: Train 0..5/6, Test 5/6..6/6
    """
    df_sorted = df.sort_values(['Year', 'Month', 'Day', 'Hour']).reset_index(drop=True)
    n_total = len(df_sorted)
    step = n_total // (n_splits + 1)

    all_test_trues = []
    all_test_preds = []
    fold_preds = []

    for fold in range(1, n_splits + 1):
        train_end = fold * step
        test_end = (fold + 1) * step if fold < n_splits else n_total

        train_fold = df_sorted.iloc[:train_end]
        test_fold = df_sorted.iloc[train_end:test_end]

        forecaster = HourlyXGBoostForecaster()
        if tune_trials > 0:
            forecaster.tune_all_hours(train_fold, feature_cols, n_trials=tune_trials, verbose=False)

        forecaster.fit(train_fold, feature_cols)
        preds_fold = forecaster.predict(test_fold, feature_cols)

        all_test_trues.extend(test_fold['Load'].values)
        all_test_preds.extend(preds_fold)
        fold_preds.append(preds_fold)

    overall_metrics = compute_metrics(np.array(all_test_trues), np.array(all_test_preds))
    return overall_metrics, np.array(all_test_preds), fold_preds
