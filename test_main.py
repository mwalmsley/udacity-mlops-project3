import json

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

@pytest.fixture()
def example_under_50k():
    return {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": 'Bachelors',
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": 'Not-in-family',
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }


@pytest.fixture()
def example_over_50k():
    return {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 280464,
        "education": 'Bachelors',
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": 'Husband',
        # this should really not be a feature!
        "race": "White",
        "sex": "Male",
        "capital-gain": 20000,
        "capital-loss": 0,
        "hours-per-week": 80,
        "native-country": "United-States"
    }


# A test case for the GET method. 
# This MUST test both the status code as well as the contents of the request object.
def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome to the API"}

# One test case for EACH of the possible inferences (results/outputs)
# of the ML model.
def test_run_mirror(example_under_50k):
    # hypens as with original csv
    # pydantic handles _ conversion automatically
    r = client.post("/mirror/", data=json.dumps(example_under_50k))
    assert r.status_code == 200
    assert r.json()['age'] ==  39

def test_run_inference(example_under_50k):
    # hypens as with original csv
    # pydantic handles _ conversion automatically
    r = client.post("/inference/", data=json.dumps(example_under_50k))
    assert r.status_code == 200
    assert r.json() == {'predicted_salary': '<=50K'}


def test_run_inference(example_over_50k):
    r = client.post("/inference/", data=json.dumps(example_over_50k))
    assert r.status_code == 200
    assert r.json() == {'predicted_salary': '>50K'}
