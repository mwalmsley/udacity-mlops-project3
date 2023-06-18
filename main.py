from fastapi import FastAPI
from typing import Union

from pydantic import BaseModel

# Declare the data object with its components and their type.
class CensusRow(BaseModel):
    age: int
    # workclass: str
    # fnlgt: int
    # education: str
    # # todo hypen
    # education_num: int
    # # todo hypen
    # marital_status: str
    # occupation: str
    # relationship: str
    # race: str
    # sex: str
    # capital_gain: int
    # # TODO hypen
    # capital_loss: int
    # # todo hypen
    # hours_per_week: int
    # # todo hypen
    # native_country: str
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
                "native-country": "United-States",
                "salary": "<=50K"
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
    return row.age > 30
