# Script to train machine learning model.
import os
import shutil

from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow

# Add the necessary imports for the starter code.
from starter.ml.data import process_data
from starter.ml.model import train_model, inference, evaluate_model_on_slices

if __name__ == '__main__':

    # Add code to load in the data.
    df = pd.read_csv('data/cleaned_data.csv')


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
    label = "salary"

    data = df[cat_features + [label]]

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=label, training=True
    )

    # Process the test data with the process_data function
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label=label, training=False, encoder=encoder
    )

    # Train and save a model.
    # TODO pipeline with encoder?
    model = train_model(X_train, y_train)

    # TODO save encoder
    # https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.save_model
    save_dir = 'model/latest/'
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)

    mlflow.sklearn.save_model(model, save_dir)

    # new - evaluate on slices
    test_preds = inference(model, X_test)
    evaluate_model_on_slices(X_test, y_test, test_preds)
