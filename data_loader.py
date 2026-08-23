"""
Data loading and temporal/calendar feature engineering module.
"""

import os
import pandas as pd
import numpy as np
from config import (
    TRAIN_EXCEL, TEST_EXCEL, TRAIN_PKL, TEST_PKL,
    YEAR_MAP, US_HOLIDAYS, MONTH_COLS
)


def load_raw_data(use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test datasets. Uses pickle cache for fast loading if available.
    """
    # Load Train
    if use_cache and os.path.exists(TRAIN_PKL):
        train_df = pd.read_pickle(TRAIN_PKL)
    else:
        train_df = pd.read_excel(TRAIN_EXCEL)
        train_df.to_pickle(TRAIN_PKL)

    # Load Test
    if use_cache and os.path.exists(TEST_PKL):
        test_df = pd.read_pickle(TEST_PKL)
    else:
        test_df = pd.read_excel(TEST_EXCEL)
        test_df.to_pickle(TEST_PKL)

    return train_df.copy(), test_df.copy()


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct calendar dates based on year deduction:
    Year 1 = 2020 (Leap year starting on Wednesday)
    Year 2 = 2021
    Year 3 = 2022

    Creates:
    - Real_Year (2020, 2021, 2022)
    - Date (datetime)
    - DayOfWeek (0=Monday, 6=Sunday)
    - Is_Weekend (1 for Saturday/Sunday, 0 for Weekdays)
    - Is_Holiday (1 for US Federal Holidays, 0 otherwise)
    - Month_1 .. Month_12 (One-hot monthly dummies)
    """
    df = df.copy()

    # Map year to actual calendar year
    df['Real_Year'] = df['Year'].map(YEAR_MAP)

    # Create timestamp
    date_series = pd.to_datetime({
        'year': df['Real_Year'],
        'month': df['Month'],
        'day': df['Day']
    })
    df['Date'] = date_series
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    # Weekend feature (5 = Saturday, 6 = Sunday)
    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)

    # Holiday feature (US Federal holidays)
    date_str = df['Date'].dt.strftime('%Y-%m-%d')
    df['Is_Holiday'] = date_str.isin(US_HOLIDAYS).astype(int)

    # Monthly one-hot dummies
    for m in range(1, 13):
        df[f'Month_{m}'] = (df['Month'] == m).astype(int)

    return df


def load_and_preprocess_data(use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and augment both Train and Test sets with calendar features.
    """
    train_raw, test_raw = load_raw_data(use_cache=use_cache)
    train_df = add_calendar_features(train_raw)
    test_df = add_calendar_features(test_raw)

    return train_df, test_df


if __name__ == '__main__':
    train_df, test_df = load_and_preprocess_data()
    print(f"Train shape: {train_df.shape}, Years: {train_df['Year'].unique()} ({train_df['Real_Year'].unique()})")
    print(f"Test shape: {test_df.shape}, Years: {test_df['Year'].unique()} ({test_df['Real_Year'].unique()})")
    print(f"Weekend days count in train: {train_df['Is_Weekend'].sum() // 24} / {len(train_df) // 24}")
    print(f"Holiday days count in train: {train_df['Is_Holiday'].sum() // 24}")
