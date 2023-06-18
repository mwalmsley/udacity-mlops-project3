import requests
import pytest
import json

# essentially a copy, of test_main, but now with requests and to remote url
# probably there's an automatic way to convert these

@pytest.fixture()
def example():
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
        "native-country": "United-States",
        "salary": "<=50K"
    }


def test_api_locally_get_root():
    r = requests.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome to the API"}

def test_run_mirror(example):
    # hypens as with original csv
    # pydantic handles _ conversion automatically
    r = requests.post("/mirror/", data=json.dumps(example))
    assert r.status_code == 200
    assert r.json()['age'] ==  39

def test_run_inference(example):
    # hypens as with original csv
    # pydantic handles _ conversion automatically
    r = requests.post("/inference/", data=json.dumps(example))
    assert r.status_code == 200
    assert r.json() == True
