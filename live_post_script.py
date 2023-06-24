import requests
import json

BASE_LIVE_URL = 'https://udacity-mlops-project3.onrender.com'

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


if __name__ == '__main__':

    root_response = requests.get(BASE_LIVE_URL)
    print(root_response.status_code, root_response.json())

    mirror_response = requests.post(BASE_LIVE_URL + "/mirror/", data=json.dumps(example_under_50k()))
    print(mirror_response.status_code, mirror_response.json())

    inference_response_a = requests.post(BASE_LIVE_URL + "/inference/", data=json.dumps(example_under_50k()))
    print(inference_response_a.status_code, inference_response_a.json())

    inference_response_b = requests.post(BASE_LIVE_URL + "/inference/", data=json.dumps(example_over_50k()))
    print(inference_response_b.status_code, inference_response_b.json())
