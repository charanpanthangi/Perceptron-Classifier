"""Model definition utilities for the perceptron classifier."""

from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_MAX_ITER = 1000
DEFAULT_TOL = 1e-3
DEFAULT_RANDOM_STATE = 42


def build_perceptron_pipeline() -> Pipeline:
    """
    Create a scikit-learn Pipeline with scaling and a perceptron classifier.

    The perceptron learning rule performs weight updates based on misclassified
    samples. Combining ``StandardScaler`` with the linear model keeps feature
    magnitudes balanced, making the gradient-like updates stable.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                Perceptron(
                    max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL, random_state=DEFAULT_RANDOM_STATE
                ),
            ),
        ]
    )
    return pipeline
