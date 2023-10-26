# Script to train machine learning model.

import pandas as pd
import logging
import joblib
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference


# Add code to load in the data.

df = pd.read_csv('../data/census.csv')
logging.basicConfig(level=logging.INFO)

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

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary",
    training=False, encoder=encoder, lb=lb
)

# Train and save a model.
logging.info('TRAINING MODEL...')
rf = train_model(X_train, y_train)

#Inference and metric evaluation
logging.info('INFERENCE AND MODEL METRICS CALCULATION...')
preds = inference(rf, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
logging.info(f"Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}")

# Save artifacts
logging.info("Saving artifacts")
joblib.dump(rf, './model.pkl')
joblib.dump(encoder, './encoder.pkl')
joblib.dump(lb, './lb.pkl')

