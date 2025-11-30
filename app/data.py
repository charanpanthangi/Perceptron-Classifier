"""Data loading utilities for the perceptron classifier example."""

from typing import Tuple
import pandas as pd
from sklearn.datasets import load_iris


def load_iris_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the classic Iris dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        A tuple of feature matrix ``X`` and target labels ``y`` as pandas objects.
    """
    iris = load_iris(as_frame=True)
    return iris.data, iris.target
