"""
Context-Aware Stacking (CAS) Ensemble Forecaster.
Combines ElasticNet (Linear Baseload) + XGBoost (Non-linear Thermal Extremes)
with an Optuna-tunable LightGBM Meta-Learner conditioned on physical/temporal context.
"""

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgb

optuna.logging.set_verbosity(optuna.logging.WARNING)


class ContextAwareStackingForecaster:
    """
    Dual-Regime Context-Aware Stacking Forecaster (CAS).
    Combines ElasticNet (Linear) and XGBoost (Non-Linear) base learners,
    routed dynamically via a LightGBM meta-learner conditioned on exogenous context.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.enet_model = None
        self.xgb_model = None
        self.meta_model = None
        self.meta_params = {}
        self.is_fitted = False

    def _init_base_models(self):
        self.enet_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state, max_iter=2000)
        self.xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=self.random_state, n_jobs=-1)

    def tune_meta(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = 'Load',
        n_trials: int = 15,
        n_splits: int = 3,
        verbose: bool = True
    ) -> dict:
        """
        Tunes the LightGBM Meta-Learner hyperparameters using Optuna over out-of-fold predictions.
        """
        if verbose:
            print(f"Optimizing Meta-Learner hyperparameters with Optuna (n_trials={n_trials})...")

        all_features = ['Hour'] + [c for c in feature_cols if c != 'Hour']
        X = train_df[all_features].values
        y = train_df[target_col].values

        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof_preds = np.zeros((len(y), 2))

        # Generate base OOF predictions
        for tr_idx, val_idx in tscv.split(X):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va = X[val_idx]

            m_enet = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state, max_iter=2000).fit(X_tr, y_tr)
            m_xgb = xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=self.random_state, n_jobs=-1).fit(X_tr, y_tr)

            oof_preds[val_idx, 0] = m_enet.predict(X_va)
            oof_preds[val_idx, 1] = m_xgb.predict(X_va)

        valid_idx = np.arange(len(y))[tscv.split(X).__next__()[1][0]:]
        meta_X = np.hstack([oof_preds[valid_idx], X[valid_idx]])
        meta_y = y[valid_idx]

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 250),
                'max_depth': trial.suggest_int('max_depth', 2, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 7, 31),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbose': -1
            }

            kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
            cv_rmses = []
            for tr_k, val_k in kf.split(meta_X):
                m = lgb.LGBMRegressor(**params)
                m.fit(meta_X[tr_k], meta_y[tr_k])
                p = m.predict(meta_X[val_k])
                cv_rmses.append(np.sqrt(np.mean((meta_y[val_k] - p) ** 2)))

            return np.mean(cv_rmses)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best = study.best_params
        best['random_state'] = self.random_state
        best['n_jobs'] = -1
        best['verbose'] = -1
        self.meta_params = best

        if verbose:
            print(f"  Meta-Learner optimized: depth={best['max_depth']}, n_est={best['n_estimators']}, lr={best['learning_rate']:.4f}")

        return self.meta_params

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str], target_col: str = 'Load', n_tune_trials: int = 0):
        """
        Fits base learners and trains the context-aware meta-learner.
        """
        all_features = ['Hour'] + [c for c in feature_cols if c != 'Hour']
        X = train_df[all_features].values
        y = train_df[target_col].values

        if n_tune_trials > 0:
            self.tune_meta(train_df, feature_cols, target_col=target_col, n_trials=n_tune_trials, verbose=True)

        tscv = TimeSeriesSplit(n_splits=3)
        oof_preds = np.zeros((len(y), 2))

        # Generate base OOF predictions
        for tr_idx, val_idx in tscv.split(X):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va = X[val_idx]

            m_enet = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state, max_iter=2000).fit(X_tr, y_tr)
            m_xgb = xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=self.random_state, n_jobs=-1).fit(X_tr, y_tr)

            oof_preds[val_idx, 0] = m_enet.predict(X_va)
            oof_preds[val_idx, 1] = m_xgb.predict(X_va)

        valid_idx = np.arange(len(y))[tscv.split(X).__next__()[1][0]:]
        meta_X = np.hstack([oof_preds[valid_idx], X[valid_idx]])
        meta_y = y[valid_idx]

        # Train meta-learner with tuned or robust default hyperparameters
        meta_hyperparams = self.meta_params if self.meta_params else {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.05,
            'num_leaves': 15,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1
        }

        self.meta_model = lgb.LGBMRegressor(**meta_hyperparams)
        self.meta_model.fit(meta_X, meta_y)

        # Refit base models on full training dataset
        self._init_base_models()
        self.enet_model.fit(X, y)
        self.xgb_model.fit(X, y)

        self.is_fitted = True
        return self

    def predict(self, test_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        """
        Generates predictions from base models and routes through the context meta-learner.
        """
        if not self.is_fitted:
            raise RuntimeError("Forecaster must be fitted before predict().")

        all_features = ['Hour'] + [c for c in feature_cols if c != 'Hour']
        X = test_df[all_features].values

        p_enet = self.enet_model.predict(X).reshape(-1, 1)
        p_xgb = self.xgb_model.predict(X).reshape(-1, 1)

        meta_X = np.hstack([p_enet, p_xgb, X])
        return self.meta_model.predict(meta_X)

    def get_feature_importances(self, feature_cols: list[str]) -> pd.DataFrame:
        all_features = ['Hour'] + [c for c in feature_cols if c != 'Hour']
        meta_feature_names = ['Pred_ElasticNet', 'Pred_XGBoost'] + all_features
        importances = self.meta_model.feature_importances_
        df_imp = pd.DataFrame({
            'Feature': meta_feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).reset_index(drop=True)
        return df_imp
