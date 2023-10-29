"""
This module includes utility functions for training the model
"""

import joblib


def load_artifact(artifact_path):
    "Load artifact"

    return joblib.load(artifact_path)


def get_cat_features():
    """Return a list of categorical features"""

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

    return cat_features
