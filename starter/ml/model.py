from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: RandomForestClassifier, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def evaluate_model_on_slices(X, y, preds, save_loc='slice_output.txt'):

    """
    Write a script that runs this function
    (or include it as part of the training script)
    that iterates through the distinct values in one of the features
    and prints out the model metrics for each value.

    Output the printout to a file named slice_output.txt.
    """

    cat_features_to_slice = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    with open(save_loc, 'w+') as f:

        for cat_feature in cat_features_to_slice:
            feature_slices = list(X[cat_feature].unique())
            for feature_slice in feature_slices:
                row_mask = X[cat_feature] == feature_slice
                y_slice, preds_slice = y[row_mask], preds[row_mask]
                precision, recall, fbeta = compute_model_metrics(y, preds)
                result_str = 'Slice: {} == {}. precision: {:.2f}, recall: {:.2f}, fbeta: {:.2f}'.format(
                    cat_feature, feature_slice, precision, recall, fbeta
                )
                print(result_str)
                f.write(result_str + '\n')

