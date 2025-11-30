"""Preprocessing utilities for the perceptron pipeline."""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TEST_SIZE = 0.2
RANDOM_STATE = 42


def train_test_split_scaled(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Split the dataset and apply standard scaling.

    The perceptron uses gradient-like weight updates, so scaling features to
    zero mean and unit variance helps the algorithm converge more reliably.

    Returns
    -------
    Tuple containing training and test splits along with the fitted scaler.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
