"""
24 Hourly-Binned XGBoost Regression Model Architecture.
Decomposes 24-hour load forecasting into 24 independent hourly models,
optimizes hyperparameters with Optuna, and reconstructs daily/annual load sequences.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold

optuna.logging.set_verbosity(optuna.logging.WARNING)


class HourlyXGBoostForecaster:
    """
    Forecasting framework using 24 independent hourly XGBoost models.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: dict[int, xgb.XGBRegressor] = {}
        self.best_params: dict[int, dict] = {}
        self.is_fitted = False

    def tune_hour(
        self,
        hour_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = 'Load',
        n_trials: int = 15,
        cv_splits: int = 3
    ) -> dict:
        """
        Optimize hyperparameters for a single hourly model using Optuna.
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 350),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
                'random_state': self.random_state,
                'n_jobs': -1
            }

            kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
            cv_rmses = []

            for tr_idx, val_idx in kf.split(hour_df):
                tr_data = hour_df.iloc[tr_idx]
                val_data = hour_df.iloc[val_idx]

                model = xgb.XGBRegressor(**params)
                model.fit(tr_data[feature_cols], tr_data[target_col])
                preds = model.predict(val_data[feature_cols])

                rmse = np.sqrt(np.mean((val_data[target_col].values - preds) ** 2))
                cv_rmses.append(rmse)

            return np.mean(cv_rmses)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best = study.best_params
        best['random_state'] = self.random_state
        best['n_jobs'] = -1
        return best

    def tune_all_hours(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = 'Load',
        n_trials: int = 15,
        cv_splits: int = 3,
        verbose: bool = True
    ) -> dict[int, dict]:
        """
        Optimize hyperparameters across all 24 hourly models.
        """
        if verbose:
            print(f"Starting Optuna hyperparameter optimization across 24 hourly models (n_trials={n_trials})...")

        for h in range(1, 25):
            hour_data = train_df[train_df['Hour'] == h]
            best_h_params = self.tune_hour(
                hour_data,
                feature_cols,
                target_col=target_col,
                n_trials=n_trials,
                cv_splits=cv_splits
            )
            self.best_params[h] = best_h_params
            if verbose and (h % 6 == 0 or h == 24 or h == 1):
                print(f"  Hour {h:2d}/24 optimized: depth={best_h_params['max_depth']}, n_est={best_h_params['n_estimators']}, lr={best_h_params['learning_rate']:.4f}")

        return self.best_params

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = 'Load',
        params_dict: dict[int, dict] = None
    ):
        """
        Fit all 24 independent hourly XGBoost models.
        """
        self.models.clear()

        for h in range(1, 25):
            hour_data = train_df[train_df['Hour'] == h]
            X_h = hour_data[feature_cols]
            y_h = hour_data[target_col]

            if params_dict and h in params_dict:
                h_params = params_dict[h]
            elif h in self.best_params:
                h_params = self.best_params[h]
            else:
                # Robust default parameters if tuning was not executed
                h_params = {
                    'n_estimators': 150,
                    'max_depth': 4,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'random_state': self.random_state,
                    'n_jobs': -1
                }

            model = xgb.XGBRegressor(**h_params)
            model.fit(X_h, y_h)
            self.models[h] = model

        self.is_fitted = True
        return self

    def predict(self, test_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        """
        Generate predictions for test dataset by dispatching to the corresponding hourly model.
        Returns a 1D numpy array aligned with test_df row order.
        """
        if not self.is_fitted:
            raise RuntimeError("Forecaster must be fitted before calling predict().")

        preds = np.zeros(len(test_df), dtype=float)

        for h in range(1, 25):
            mask = (test_df['Hour'] == h).values
            if not np.any(mask):
                continue
            X_h = test_df.loc[mask, feature_cols]
            preds[mask] = self.models[h].predict(X_h)

        return preds

    def get_feature_importances(self, feature_cols: list[str]) -> pd.DataFrame:
        """
        Returns average feature importance across all 24 hourly models.
        """
        records = []
        for h, model in self.models.items():
            imp = model.feature_importances_
            for feat, val in zip(feature_cols, imp):
                records.append({'Hour': h, 'Feature': feat, 'Importance': val})
        df_imp = pd.DataFrame(records)
        avg_imp = df_imp.groupby('Feature')['Importance'].mean().reset_index().sort_values('Importance', ascending=False)
        return avg_imp
