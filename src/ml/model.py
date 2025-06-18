# TRD_BOT_V3/src/ml/model.py

import os
import pickle
import numpy as np
import pandas as pd
from typing import Union

class MLModel:
    """
    Wrapper around a scikit-learn classifier for predicting P(Up).
    Expects model files saved as pickles.
    """
    def __init__(self, model_path: str):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ML model not found at {model_path}")
        with open(model_path, "rb") as f:
            self.clf = pickle.load(f)

        # Check that the classifier has predict_proba
        if not hasattr(self.clf, "predict_proba"):
            raise ValueError("Loaded model does not support predict_proba()")

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Return the probability of the “Up” class for each row in X.
        If `clf.classes_` is [0,1], index 1 corresponds to P(Up).
        """
        probs = self.clf.predict_proba(X)
        # Find index of class '1'
        idx_up = list(self.clf.classes_).index(1)
        return probs[:, idx_up]

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Return binary predictions (0 or 1).
        """
        return self.clf.predict(X)
