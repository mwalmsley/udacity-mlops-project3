import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Write unit tests for at least 3 functions in the model code

from starter.ml import model

@pytest.fixture()
def X_train():
    return np.random.rand(100, 5)

@pytest.fixture()
def y_train():
    return np.random.randint(low=0, high=2, size=100)

@pytest.fixture()
def preds(y_train):
    # expect perfect predictions
    return y_train


def test_train_model(X_train, y_train):
    trained_model = model.train_model(X_train, y_train)
    assert isinstance(trained_model, RandomForestClassifier)


def test_compute_model_metrics(y_train, preds):
    precision, recall, fbeta = model.compute_model_metrics(y_train, preds)
    assert precision == 1.
    assert recall == 1.
    assert fbeta == 1.

# could use pytest mocks but not necessary here
# just as easy to be explicit
class MockModel():

    def __init__(self):
        pass

    def predict(self, x):
        return np.random.randint(low=0, high=2, size=len(x))

def test_inference(X_train):
    mock_model = MockModel()
    inference_predictions = model.inference(mock_model, X_train)
    assert isinstance(inference_predictions, np.ndarray)
    assert len(inference_predictions) == len(X_train)
