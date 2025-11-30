import numpy as np

from app.evaluate import evaluate_predictions


def test_evaluate_predictions_scores():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
    assert metrics["precision_macro"] > 0
    assert metrics["recall_macro"] > 0
    assert metrics["f1_macro"] > 0
