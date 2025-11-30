from app.data import load_iris_dataset
from app.model import build_perceptron_pipeline
from app.preprocess import train_test_split_scaled


def test_model_train_and_predict():
    X, y = load_iris_dataset()
    X_train, X_test, y_train, y_test, _ = train_test_split_scaled(X, y)
    model = build_perceptron_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)
