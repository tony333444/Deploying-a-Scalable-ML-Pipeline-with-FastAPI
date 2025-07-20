import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Load and prepare test data
data = pd.read_csv("data/census.csv")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

train = data.sample(frac=0.8, random_state=1)
test = data.drop(train.index)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

model = train_model(X_train, y_train)


def test_one():
    """
    Test that the trained model is an instance of RandomForestClassifier.
    """
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier"


def test_two():
    """
    Test that inference returns the same number of predictions as test inputs.
    """
    preds = inference(model, X_test)
    assert preds.shape[0] == X_test.shape[0], "Mismatch between number of predictions and input rows"


def test_three():
    """
    Test that compute_model_metrics returns values of type float.
    """
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert all(isinstance(val, float) for val in [precision, recall, fbeta]), "Metric outputs are not all floats"
