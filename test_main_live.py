import requests
import pytest
import json

# essentially a copy, of test_main, but now with requests and to remote url
# probably there's an automatic way to convert these

BASE_LIVE_URL = 'https://udacity-mlops-project3.onrender.com'

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




def test_api_locally_get_root():
    r = requests.get(BASE_LIVE_URL)
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome to the API"}

def test_run_mirror(example_under_50k):
    # hypens as with original csv
    # pydantic handles _ conversion automatically
    r = requests.post(BASE_LIVE_URL + "/mirror/", data=json.dumps(example_under_50k))
    assert r.status_code == 200
    assert r.json()['age'] ==  39

def test_run_inference_under_50k(example_under_50k):
    r = requests.post(BASE_LIVE_URL + "/inference/", data=json.dumps(example_under_50k))
    assert r.status_code == 200
    assert r.json() == {'predicted_salary': '<=50K'}


def test_run_inference_over_50k(example_over_50k):
    r = requests.post(BASE_LIVE_URL + "/inference/", data=json.dumps(example_over_50k))
    assert r.status_code == 200
    assert r.json() == {'predicted_salary': '>50K'}
