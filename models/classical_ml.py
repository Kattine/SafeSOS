"""
Classical ML pipeline:
  Input image -> MediaPipe Hands -> 21 keypoints (x, y, z) -> SVM / RF.

Wraps everything behind a sklearn-compatible interface so it can be
evaluated identically to the other two models.
"""

from typing import Literal

import numpy as np


class KeypointClassifier:
    """SVM or Random Forest on top of MediaPipe hand keypoints."""

    def __init__(self, backend: Literal["svm", "rf"] = "svm") -> None:
        self.backend = backend
        self.model = None  # initialised in fit()

    def fit(self, X_keypoints: np.ndarray, y: np.ndarray) -> "KeypointClassifier":
        """X_keypoints: (N, 63) flattened (x, y, z) for 21 landmarks."""
        # TODO: instantiate SVC or RandomForestClassifier and fit
        return self

    def predict(self, X_keypoints: np.ndarray) -> np.ndarray:
        # TODO
        raise NotImplementedError

    def predict_proba(self, X_keypoints: np.ndarray) -> np.ndarray:
        # TODO — required for Platt-scaled selective prediction
        raise NotImplementedError
