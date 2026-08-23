"""
Feature Engineering module:
- Principal Component Analysis (PCA) on Temperature and GHI
- Variance Inflation Factor (VIF) computation
- Lag and Lead feature generation for exogenous variables
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from config import TEMP_COLS, GHI_COLS, CALENDAR_COLS, TARGET_COL


def compute_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for each feature.
    VIF_i = 1 / (1 - R_i^2), where R_i^2 is from regressing feature i on all other features.
    """
    vif_records = []
    for feat in features:
        other_feats = [f for f in features if f != feat]
        X = df[other_feats].values
        y = df[feat].values

        reg = LinearRegression()
        reg.fit(X, y)
        r2 = reg.score(X, y)

        vif_val = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
        vif_records.append({'Feature': feat, 'VIF': vif_val})

    return pd.DataFrame(vif_records)


class PCAExogenousTransformer:
    """
    Fits separate PCA models on Temperature (5 sites) and GHI (5 sites).
    Replaces 10 correlated site features with 2 principal components:
    - PCA_Temp (explains ~97-99% variance)
    - PCA_GHI (explains >99% variance)
    """
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

    def get_explained_variance(self) -> dict:
        return {
            'PCA_Temp_Ratio': float(self.pca_temp.explained_variance_ratio_[0]),
            'PCA_GHI_Ratio': float(self.pca_ghi.explained_variance_ratio_[0])
        }


def add_lag_lead_features(df: pd.DataFrame, lags: list[int] = None, leads: list[int] = None) -> pd.DataFrame:
    """
    Add lagging and leading features for PCA_Temp and PCA_GHI.
    lags: list of integers, e.g. [1, 2, 3, 5]
    leads: list of integers, e.g. [1, 2]
    """
    df = df.copy()
    if lags is None:
        lags = []
    if leads is None:
        leads = []

    for lag in lags:
        df[f'PCA_Temp_Lag{lag}'] = df['PCA_Temp'].shift(lag).bfill()
        df[f'PCA_GHI_Lag{lag}'] = df['PCA_GHI'].shift(lag).bfill()

    for lead in leads:
        df[f'PCA_Temp_Lead{lead}'] = df['PCA_Temp'].shift(-lead).ffill()
        df[f'PCA_GHI_Lead{lead}'] = df['PCA_GHI'].shift(-lead).ffill()

    return df


def add_all_lag_lead_configurations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all lag and lead features used across the paper's experiments:
    Lags: 1, 2, 3, 5
    Leads: 1, 2
    """
    return add_lag_lead_features(df, lags=[1, 2, 3, 5], leads=[1, 2])


def get_feature_list_by_config(config_name: str) -> list[str]:
    """
    Returns feature column names for a given experiment configuration:
    - 'Baseline': ['PCA_Temp', 'PCA_GHI'] + Calendar
    - 'Lag1': Baseline + ['PCA_Temp_Lag1', 'PCA_GHI_Lag1']  (Paper's Final Model)
    - 'Lag2': Baseline + ['PCA_Temp_Lag2', 'PCA_GHI_Lag2']
    - 'Lead1': Baseline + ['PCA_Temp_Lead1', 'PCA_GHI_Lead1']
    - 'Lag5': Baseline + ['PCA_Temp_Lag5', 'PCA_GHI_Lag5']
    - 'Lag1+Lead1': Baseline + Lag1 + Lead1
    - 'Lag3': Baseline + ['PCA_Temp_Lag3', 'PCA_GHI_Lag3']
    - 'Lag2+Lead2': Baseline + Lag2 + Lead2
    - 'Lead2': Baseline + ['PCA_Temp_Lead2', 'PCA_GHI_Lead2']
    - 'Lag3+Lead2': Baseline + Lag3 + Lead2
    - 'Lag5+Lead2': Baseline + Lag5 + Lead2
    """
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
        raise ValueError(f"Unknown config name: {config_name}. Available: {list(config_map.keys())}")

    return base_features + config_map[config_name]
