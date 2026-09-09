"""
Data loading and calendar feature generation.
"""

import os
import pandas as pd
from .config import (
    TRAIN_EXCEL, TEST_EXCEL, TRAIN_PKL, TEST_PKL,
    YEAR_MAP, US_HOLIDAYS
)


def load_raw_data(use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if use_cache and os.path.exists(TRAIN_PKL):
        train_df = pd.read_pickle(TRAIN_PKL)
    else:
        train_df = pd.read_excel(TRAIN_EXCEL)
        train_df.to_pickle(TRAIN_PKL)

    if use_cache and os.path.exists(TEST_PKL):
        test_df = pd.read_pickle(TEST_PKL)
    else:
        test_df = pd.read_excel(TEST_EXCEL)
        test_df.to_pickle(TEST_PKL)

    return train_df.copy(), test_df.copy()


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Real_Year'] = df['Year'].map(YEAR_MAP)
    date_series = pd.to_datetime({
        'year': df['Real_Year'],
        'month': df['Month'],
        'day': df['Day']
    })
    df['Date'] = date_series
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
    date_str = df['Date'].dt.strftime('%Y-%m-%d')
    df['Is_Holiday'] = date_str.isin(US_HOLIDAYS).astype(int)

    for m in range(1, 13):
        df[f'Month_{m}'] = (df['Month'] == m).astype(int)

    return df


def load_and_preprocess_data(use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_raw, test_raw = load_raw_data(use_cache=use_cache)
    train_df = add_calendar_features(train_raw)
    test_df = add_calendar_features(test_raw)
    return train_df, test_df
