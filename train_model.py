# Script to train machine learning model.
import pickle

from sklearn.model_selection import train_test_split
import pandas as pd
import wandb

# Add the necessary imports for the starter code.
from starter.ml.data import process_data
from starter.ml.model import train_model, inference, compute_model_metrics, evaluate_model_on_slices

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

    # cat features + cont. features + label
    data = df.copy()

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, label_binarizer = process_data(
        train, categorical_features=cat_features, label=label, training=True
    )


    # Train and save a model.

    model_path = 'model/model.pkl'
    encoder_path = 'model/encoder.pkl'
    label_binarizer_path = 'model/label_binarizer.pkl'

    model = train_model(X_train, y_train)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)
    # also saving lb to allow for inference to be converted back to str
    with open(label_binarizer_path, 'wb') as f:
        pickle.dump(label_binarizer, f)

    # upload to wandb for production server
    run = wandb.init(
        project='udacity-mlops-project3',
        job_type='upload-trained-model'
    )

    model_artifact = wandb.Artifact(
        name='trained_census_model', 
        type='trained_model'
    )
    model_artifact.add_file(local_path=model_path)
    run.log_artifact(model_artifact)

    encoder_artifact = wandb.Artifact(
        name='categorical_encoder', 
        type='CategoricalEncoder'
    )
    encoder_artifact.add_file(local_path=encoder_path)
    run.log_artifact(encoder_artifact)

    label_binarizer_artifact = wandb.Artifact(
        name='label_binarizer', 
        type='LabelBinarizer'
    )
    label_binarizer_artifact.add_file(local_path=label_binarizer_path)
    run.log_artifact(label_binarizer_artifact)


    # closure func. to repeat process_data with encoder/lb
    # useful below, but mostly useful for evaluate_model_on_slices
    def scoring_preprocess_func(df):
        # convert clean data to numeric X, y. Also adjust labels.
        X, y, _, _ = process_data(
            df, categorical_features=cat_features, label=label, training=False, encoder=encoder, lb=label_binarizer
        )
        return X, y
    X_test, y_test = scoring_preprocess_func(test)
    test_preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, test_preds)
    result_str = 'Overall: precision: {:.2f}, recall: {:.2f}, fbeta: {:.2f}'.format(precision, recall, fbeta)
    print(result_str)

    evaluate_model_on_slices(test, model, scoring_preprocess_func)
