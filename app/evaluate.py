"""Evaluation helpers for the perceptron classifier."""

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


MetricDict = Dict[str, float]


def evaluate_predictions(y_true, y_pred) -> MetricDict:
    """
    Compute common classification metrics.

    Returns a dictionary with accuracy, precision, recall, and F1-score.
    """
    metrics: MetricDict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    return metrics


def confusion(y_true, y_pred) -> np.ndarray:
    """Return the confusion matrix for the predictions."""
    return confusion_matrix(y_true, y_pred)


def full_classification_report(y_true, y_pred) -> str:
    """Generate a text classification report."""
    return classification_report(y_true, y_pred, zero_division=0)
