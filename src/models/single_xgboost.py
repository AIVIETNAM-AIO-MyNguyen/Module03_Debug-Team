"""
Single (Monolithic) XGBoost Regression Model Architecture.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold

optuna.logging.set_verbosity(optuna.logging.WARNING)


class SingleXGBoostForecaster:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.best_params = {}
        self.is_fitted = False

    def tune(self, train_df: pd.DataFrame, feature_cols: list[str], target_col: str = 'Load', n_trials: int = 15, cv_splits: int = 3, verbose: bool = True):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
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
            for tr_idx, val_idx in kf.split(train_df):
                tr_data = train_df.iloc[tr_idx]
                val_data = train_df.iloc[val_idx]
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
        self.best_params = best
        return self.best_params

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str], target_col: str = 'Load', params: dict = None):
        X = train_df[feature_cols]
        y = train_df[target_col]
        model_params = params if params is not None else (self.best_params if self.best_params else {
            'n_estimators': 250, 'max_depth': 6, 'learning_rate': 0.05,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1,
            'reg_lambda': 1.0, 'random_state': self.random_state, 'n_jobs': -1
        })
        self.model = xgb.XGBRegressor(**model_params)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, test_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Forecaster must be fitted before predict().")
        return self.model.predict(test_df[feature_cols])

    def get_feature_importances(self, feature_cols: list[str]) -> pd.DataFrame:
        imp = self.model.feature_importances_
        df_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': imp})
        return df_imp.sort_values('Importance', ascending=False).reset_index(drop=True)
