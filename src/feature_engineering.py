"""
Feature Engineering Module:
Explicitly separates:
1. Original Paper Features (PCA Temperature/GHI, Calendar Dummies, Exogenous Lags/Leads).
2. Our Added Domain Features (Daily/Annual Fourier Harmonics, Heating & Cooling Degree Days).
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from .config import TEMP_COLS, GHI_COLS, CALENDAR_COLS


# =========================================================================
# 1. Feature Set Definitions (Original vs. Our Added Features)
# =========================================================================

# (A) Original Paper Feature Mappings (16 - 20 features)
ORIGINAL_PAPER_CONFIG_MAP = {
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

# (B) Our Added Features
OUR_ADDED_FOURIER_FEATURES = ['Sin_Hour', 'Cos_Hour', 'Sin_Year', 'Cos_Year']
OUR_ADDED_DEGREE_DAY_FEATURES = ['HDD', 'CDD']
OUR_ALL_ADDED_FEATURES = OUR_ADDED_FOURIER_FEATURES + OUR_ADDED_DEGREE_DAY_FEATURES


def get_original_paper_features(config_name: str = 'Lag1') -> list[str]:
    """
    Returns ONLY the feature columns from the original paper (PCA + Calendar + Lags/Leads).
    Total count: 18 features for 'Lag1'.
    """
    if config_name not in ORIGINAL_PAPER_CONFIG_MAP:
        raise ValueError(f"Unknown original paper config: {config_name}. Available: {list(ORIGINAL_PAPER_CONFIG_MAP.keys())}")
    base_features = ['PCA_Temp', 'PCA_GHI'] + CALENDAR_COLS
    return base_features + ORIGINAL_PAPER_CONFIG_MAP[config_name]


def get_feature_list_by_config(config_name: str) -> list[str]:
    """
    Alias for get_original_paper_features for backward compatibility.
    """
    return get_original_paper_features(config_name)


def get_extended_features(
    base_config: str = 'Lag1',
    include_fourier: bool = True,
    include_degree_days: bool = True
) -> list[str]:
    """
    Returns the combined feature list: Original Paper Features + Our Added Features.
    - Base Lag1 alone: 18 features
    - + Fourier: 22 features
    - + Fourier + HDD/CDD: 24 features
    """
    features = get_original_paper_features(base_config)
    if include_fourier:
        features = features + OUR_ADDED_FOURIER_FEATURES
    if include_degree_days:
        features = features + OUR_ADDED_DEGREE_DAY_FEATURES
    return features


# =========================================================================
# 2. Transformers and Feature Generators
# =========================================================================

class PCAExogenousTransformer:
    """
    Fits and extracts the 1st principal component of 5 Temperature sites and 5 GHI sites.
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


def add_lag_lead_features(df: pd.DataFrame, lags: list[int] = None, leads: list[int] = None) -> pd.DataFrame:
    """
    Generates temporal lag and lead shifts on PCA components.
    """
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


def add_fourier_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Our Added Feature]: Continuous diurnal (daily) and annual (yearly) Fourier harmonics.
    """
    df = df.copy()
    # Diurnal (Daily) Seasonality: Period = 24 hours
    df['Sin_Hour'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
    df['Cos_Hour'] = np.cos(2 * np.pi * df['Hour'] / 24.0)

    # Annual (Yearly) Seasonality: Period = 365.25 days
    if 'Real_Year' in df.columns:
        dates = pd.to_datetime(df[['Real_Year', 'Month', 'Day']].rename(
            columns={'Real_Year': 'year', 'Month': 'month', 'Day': 'day'}
        ))
        day_of_year = dates.dt.dayofyear
        df['Sin_Year'] = np.sin(2 * np.pi * day_of_year / 365.25)
        df['Cos_Year'] = np.cos(2 * np.pi * day_of_year / 365.25)

    return df


def add_hdd_cdd_features(df: pd.DataFrame, base_temp: float = 18.0) -> pd.DataFrame:
    """
    [Our Added Feature]: Heating Degree Days (HDD) and Cooling Degree Days (CDD).
    Captures the physical non-linear HVAC activation thresholds at base 18°C.
    """
    df = df.copy()
    avg_temp = df[TEMP_COLS].mean(axis=1)
    df['HDD'] = np.maximum(base_temp - avg_temp, 0)
    df['CDD'] = np.maximum(avg_temp - base_temp, 0)
    return df
