"""Run a full perceptron classification experiment on the Iris dataset."""

from pprint import pprint

from app.data import load_iris_dataset
from app.evaluate import confusion, evaluate_predictions, full_classification_report
from app.model import build_perceptron_pipeline
from app.preprocess import train_test_split_scaled
from app.visualize import plot_confusion_matrix, plot_pca_scatter


def run_pipeline() -> None:
    """Load data, train the model, evaluate, and visualize results."""
    X, y = load_iris_dataset()
    X_train, X_test, y_train, y_test, scaler = train_test_split_scaled(X, y)

    model = build_perceptron_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    conf_matrix = confusion(y_test, y_pred)

    print("Perceptron evaluation metrics:")
    pprint(metrics)
    print("\nClassification report:")
    print(full_classification_report(y_test, y_pred))

    # Visualize and save artifacts
    plot_confusion_matrix(conf_matrix, class_labels=y.unique())
    scaled_features = scaler.transform(X)
    plot_pca_scatter(scaled_features, y)


if __name__ == "__main__":
    run_pipeline()
