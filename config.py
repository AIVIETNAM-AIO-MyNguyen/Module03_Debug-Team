"""
Configuration and constants for replicating the load forecasting paper:
'IISE PG&E Energy Analytics Challenge 2025: Hourly-Binned Regression Models Beat Transformers in Load Forecasting'
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_EXCEL = os.path.join(BASE_DIR, 'Train.xlsx')
TEST_EXCEL = os.path.join(BASE_DIR, 'Test.xlsx')
TRAIN_PKL = os.path.join(BASE_DIR, 'train.pkl')
TEST_PKL = os.path.join(BASE_DIR, 'test.pkl')

OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

os.makedirs(FIGURES_DIR, exist_ok=True)

# Year mappings based on paper footnote 1:
# Year 1 starts on a Wednesday in a leap year -> 2020
# Year 2 -> 2021
# Year 3 (Test) -> 2022
YEAR_MAP = {1: 2020, 2: 2021, 3: 2022}

# Temperature and GHI site columns
TEMP_COLS = [f'Site-{i} Temp' for i in range(1, 6)]
GHI_COLS = [f'Site-{i} GHI' for i in range(1, 6)]
TARGET_COL = 'Load'

# US Federal Holidays for 2020-2022
US_HOLIDAYS = [
    # 2020
    '2020-01-01',  # New Year's Day
    '2020-01-20',  # MLK Day
    '2020-02-17',  # Washington's Birthday / Presidents' Day
    '2020-05-25',  # Memorial Day
    '2020-07-03',  # Independence Day (observed)
    '2020-07-04',  # Independence Day
    '2020-09-07',  # Labor Day
    '2020-10-12',  # Columbus Day
    '2020-11-11',  # Veterans Day
    '2020-11-26',  # Thanksgiving Day
    '2020-12-25',  # Christmas Day
    # 2021
    '2021-01-01',  # New Year's Day
    '2021-01-18',  # MLK Day
    '2021-02-15',  # Presidents' Day
    '2021-05-31',  # Memorial Day
    '2021-06-18',  # Juneteenth (observed)
    '2021-06-19',  # Juneteenth
    '2021-07-04',  # Independence Day
    '2021-07-05',  # Independence Day (observed)
    '2021-09-06',  # Labor Day
    '2021-10-11',  # Columbus Day
    '2021-11-11',  # Veterans Day
    '2021-11-25',  # Thanksgiving Day
    '2021-12-24',  # Christmas Day (observed)
    '2021-12-25',  # Christmas Day
    # 2022
    '2022-01-01',  # New Year's Day
    '2022-01-17',  # MLK Day
    '2022-02-21',  # Presidents' Day
    '2022-05-30',  # Memorial Day
    '2022-06-19',  # Juneteenth
    '2022-06-20',  # Juneteenth (observed)
    '2022-07-04',  # Independence Day
    '2022-09-05',  # Labor Day
    '2022-10-10',  # Columbus Day
    '2022-11-11',  # Veterans Day
    '2022-11-24',  # Thanksgiving Day
    '2022-12-25',  # Christmas Day
    '2022-12-26'   # Christmas Day (observed)
]

# Base temporal dummy features
MONTH_COLS = [f'Month_{m}' for m in range(1, 13)]
CALENDAR_COLS = ['Is_Weekend', 'Is_Holiday'] + MONTH_COLS

# Default XGBoost parameter grid for hourly tuning
OPTUNA_XGB_PARAM_GRID = {
    'n_estimators': (50, 400),
    'max_depth': (3, 8),
    'learning_rate': (0.01, 0.2),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'reg_alpha': (0.0, 5.0),
    'reg_lambda': (0.1, 10.0)
}
