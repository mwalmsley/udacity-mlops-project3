import pickle
# from dataclasses import dataclass, asdict
import json

import pandas as pd
from fastapi import FastAPI
from typing import Union

from pydantic import BaseModel, Field

from starter.ml import data

# using alias for hypens
# https://github.com/pydantic/pydantic/issues/2266#issuecomment-760993721
class CensusRow(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')
    # salary: str

    # https://fastapi.tiangolo.com/tutorial/schema-extra-example/#pydantic-schema_extra
    class Config:
        schema_extra = {
            "example": {
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
                # "salary": "<=50K"
            }
        }


# Instantiate the app.
app = FastAPI()

# GET on the root giving a welcome message.
@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to the API"}

# POST that does model inference
# Use a Pydantic model to ingest the body from POST. This model should contain an example. 
# the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
@app.post("/mirror/")
async def run_mirror(row: CensusRow):
    return row

@app.post("/inference/")
async def run_inference(row: CensusRow):
    
    row_pd = pd.Series(data=dict(row))
    row_df = pd.DataFrame(data=[row_pd])  # one row df

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    # label = "salary"

    with open('model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('model/label_binarizer.pkl', 'rb') as f:
        lb = pickle.load(f)

    X, _, _, _ = data.process_data(row_df, encoder=encoder, categorical_features=cat_features, training=False)

    y_pred = model.predict(X)

    salary_pred = lb.inverse_transform(y_pred)[0]

    return {'predicted_salary': salary_pred}
