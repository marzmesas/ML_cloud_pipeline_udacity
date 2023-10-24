# Script to train machine learning model.

import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics


# Add code to load in the data.

df = pd.read_csv('../data/census.csv')


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.