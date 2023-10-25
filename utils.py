"""
This module includes utility functions for training the model
"""

import joblib

def load_artifact(artifact_path):
    "Load artifact"

    return joblib.load(artifact_path)
