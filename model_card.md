# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Random Forest Classifier (scikit-learn) for predicting salary (above or below $50k) using census data.

## Intended Use

Acme Inc. API for credit assessment

## Training Data

Random split of US Census data (see cleaned_data.csv in this repo)

## Evaluation Data

Remainder of US Census data not used for training. Currently split is stochastic as model is under development.

## Metrics

The metrics used are precision, recall, and f-beta (weighted harmonic of precision and recall).

Overall test set performance is:

    precision: 0.73
    recall: 0.61
    fbeta: 0.66

Performance on each dataset slice is recorded in slice_output.txt

## Ethical Considerations

Model is trained on sensitive columns e.g. race. This may be illegal (in the EU, unless required for specific purpose) or otherwise inadvisable.

## Caveats and Recommendations

Model is under development and not yet suitable for production use.
