import pickle
import os
from typing import Union

import wandb
from pydantic import BaseModel, Field
import pandas as pd
from fastapi import FastAPI

from starter.ml import data



# not an API call
def load_artifacts(run):

    # download from wandb

    model_path = run.use_artifact('trained_census_model:latest').download('artifacts')
    encoder_path = run.use_artifact('categorical_encoder:latest').download('artifacts')
    label_binarizer_path = run.use_artifact('label_binarizer:latest').download('artifacts')

    with open(model_path+'/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(encoder_path + '/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open(label_binarizer_path + '/label_binarizer.pkl', 'rb') as f:
        lb = pickle.load(f)

    return model, encoder, lb


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

CAT_FEATURES = [
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



wandb.login(key=os.environ['WANDB_API_KEY'])
run = wandb.init(
    project='udacity-mlops-project3',
    job_type='inference_server'
)
MODEL, ENCODER, LB = load_artifacts(run)


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

# model etc. loaded on start and passed via closure
@app.post("/inference/")
async def run_inference(row: CensusRow):
    
    row_pd = pd.Series(data=dict(row))
    row_df = pd.DataFrame(data=[row_pd])  # one row df


    X, _, _, _ = data.process_data(row_df, encoder=ENCODER, categorical_features=CAT_FEATURES, training=False)

    y_pred = MODEL.predict(X)

    salary_pred = LB.inverse_transform(y_pred)[0]

    return {'predicted_salary': salary_pred}
