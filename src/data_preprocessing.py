"""
Data Preprocessing and Pipeline Utilities for Airfare Price Estimation.
Problem: SIH 2026 - SIH26056 (Topic 17: Airfare Component Price Estimation)
Course: Supervised Learning (INT395)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

CAT_COLS = [
    'airline',
    'source_city',
    'destination_city',
    'departure_time',
    'arrival_time',
    'stops',
    'class',
    'route'
]

NUM_COLS = [
    'duration',
    'days_left'
]

TARGET_COL = 'price'

def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """
    Loads dataset, handles missing/redundant columns, and performs feature engineering.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Drop index column if present
    for drop_col in ['Unnamed: 0', 'flight']:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])

    # Feature engineering: route interaction
    if 'source_city' in df.columns and 'destination_city' in df.columns:
        df['route'] = df['source_city'] + ' -> ' + df['destination_city']

    # Drop missing values if any
    df = df.dropna().reset_index(drop=True)

    return df

def get_preprocessor():
    """
    Constructs a ColumnTransformer that scales numeric features and
    one-hot encodes nominal categorical features with drop='first'.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_COLS),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), CAT_COLS)
        ],
        remainder='drop'
    )
    return preprocessor

def prepare_train_test_data(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Loads data and splits into train and test sets.
    """
    df = load_and_clean_data(csv_path)
    X = df[CAT_COLS + NUM_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, df
