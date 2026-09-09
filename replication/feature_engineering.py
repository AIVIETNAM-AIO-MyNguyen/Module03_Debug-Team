"""
Original Paper Feature Engineering (PCA on Temperature & GHI, Lag/Lead features).
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from .config import TEMP_COLS, GHI_COLS, CALENDAR_COLS


class PCAExogenousTransformer:
    def __init__(self):
        self.pca_temp = PCA(n_components=1)
        self.pca_ghi = PCA(n_components=1)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        self.pca_temp.fit(df[TEMP_COLS])
        self.pca_ghi.fit(df[GHI_COLS])
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("PCA transformer must be fitted before calling transform().")
        df = df.copy()
        df['PCA_Temp'] = self.pca_temp.transform(df[TEMP_COLS])
        df['PCA_GHI'] = self.pca_ghi.transform(df[GHI_COLS])
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def get_explained_variance(self) -> dict[str, float]:
        if not self.is_fitted:
            raise RuntimeError("Transformer must be fitted first.")
        return {
            'PCA_Temp': float(self.pca_temp.explained_variance_ratio_[0]),
            'PCA_GHI': float(self.pca_ghi.explained_variance_ratio_[0])
        }


def compute_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Computes Variance Inflation Factor (VIF) replicating Table 2 & Table 3 of Roy et al. (2025)
    using statsmodels variance_inflation_factor with unstandardized features.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    sub = df[features].dropna()
    vifs = [variance_inflation_factor(sub.values, i, standardize=False) for i in range(sub.shape[1])]
    return pd.DataFrame({'Feature': features, 'VIF': vifs})


def add_lag_lead_features(df: pd.DataFrame, lags: list[int] = None, leads: list[int] = None) -> pd.DataFrame:
    df = df.copy()
    if lags is None: lags = []
    if leads is None: leads = []

    for lag in lags:
        df[f'PCA_Temp_Lag{lag}'] = df['PCA_Temp'].shift(lag).bfill()
        df[f'PCA_GHI_Lag{lag}'] = df['PCA_GHI'].shift(lag).bfill()

    for lead in leads:
        df[f'PCA_Temp_Lead{lead}'] = df['PCA_Temp'].shift(-lead).ffill()
        df[f'PCA_GHI_Lead{lead}'] = df['PCA_GHI'].shift(-lead).ffill()

    return df


def add_all_lag_lead_configurations(df: pd.DataFrame) -> pd.DataFrame:
    return add_lag_lead_features(df, lags=[1, 2, 3, 5], leads=[1, 2])


def get_feature_list_by_config(config_name: str) -> list[str]:
    base_features = ['PCA_Temp', 'PCA_GHI'] + CALENDAR_COLS
    config_map = {
        'Baseline': [],
        'Lag1': ['PCA_Temp_Lag1', 'PCA_GHI_Lag1'],
        'Lag2': ['PCA_Temp_Lag2', 'PCA_GHI_Lag2'],
        'Lead1': ['PCA_Temp_Lead1', 'PCA_GHI_Lead1'],
        'Lag5': ['PCA_Temp_Lag5', 'PCA_GHI_Lag5'],
        'Lag1+Lead1': ['PCA_Temp_Lag1', 'PCA_GHI_Lag1', 'PCA_Temp_Lead1', 'PCA_GHI_Lead1'],
        'Lag3': ['PCA_Temp_Lag3', 'PCA_GHI_Lag3'],
        'Lag2+Lead2': ['PCA_Temp_Lag2', 'PCA_GHI_Lag2', 'PCA_Temp_Lead2', 'PCA_GHI_Lead2'],
        'Lead2': ['PCA_Temp_Lead2', 'PCA_GHI_Lead2'],
        'Lag3+Lead2': ['PCA_Temp_Lag3', 'PCA_GHI_Lag3', 'PCA_Temp_Lead2', 'PCA_GHI_Lead2'],
        'Lag5+Lead2': ['PCA_Temp_Lag5', 'PCA_GHI_Lag5', 'PCA_Temp_Lead2', 'PCA_GHI_Lead2'],
    }
    if config_name not in config_map:
        raise ValueError(f"Unknown config name: {config_name}")
    return base_features + config_map[config_name]
