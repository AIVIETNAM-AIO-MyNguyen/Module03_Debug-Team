"""
Configuration parameters, file paths, and constants for the research pipeline.
"""

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
TRAIN_EXCEL = os.path.join(ROOT_DIR, 'Train.xlsx')
TEST_EXCEL = os.path.join(ROOT_DIR, 'Test.xlsx')
TRAIN_PKL = os.path.join(ROOT_DIR, 'train_cache.pkl')
TEST_PKL = os.path.join(ROOT_DIR, 'test_cache.pkl')

# Output directories
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Feature definitions
TEMP_COLS = [f'Site-{i} Temp' for i in range(1, 6)]
GHI_COLS = [f'Site-{i} GHI' for i in range(1, 6)]
TARGET_COL = 'Load'

CALENDAR_COLS = ['Is_Weekend', 'Is_Holiday'] + [f'Month_{m}' for m in range(1, 13)]
YEAR_MAP = {1: 2020, 2: 2021, 3: 2022}

US_HOLIDAYS = [
    '2020-01-01', '2020-01-20', '2020-02-17', '2020-05-25', '2020-07-03', '2020-07-04',
    '2020-09-07', '2020-10-12', '2020-11-11', '2020-11-26', '2020-12-25',
    '2021-01-01', '2021-01-18', '2021-02-15', '2021-05-31', '2021-06-18', '2021-06-19',
    '2021-07-05', '2021-09-06', '2021-10-11', '2021-11-11', '2021-11-25', '2021-12-24', '2021-12-25',
    '2022-01-01', '2022-01-17', '2022-02-21', '2022-05-30', '2022-06-19', '2022-06-20',
    '2022-07-04', '2022-09-05', '2022-10-10', '2022-11-11', '2022-11-24', '2022-12-25', '2022-12-26'
]
