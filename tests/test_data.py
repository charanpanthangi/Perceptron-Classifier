import pandas as pd

from app.data import load_iris_dataset


def test_load_iris_dataset_shapes():
    X, y = load_iris_dataset()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert X.shape[1] == 4  # iris has four features
