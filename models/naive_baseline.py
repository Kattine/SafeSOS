"""
Naive baseline: majority-class predictor.
Always predicts whichever class is most frequent in training.
"""

from collections import Counter
from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class MajorityClassifier(BaseEstimator, ClassifierMixin):
    """Predicts the most frequent class regardless of input."""

    def __init__(self) -> None:
        self.majority_class_: int | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X, y: Sequence[int]) -> "MajorityClassifier":
        """Memorize the majority class from y."""
        counts = Counter(y)
        self.majority_class_ = counts.most_common(1)[0][0]
        self.classes_ = np.array(sorted(set(y)))
        return self

    def predict(self, X) -> np.ndarray:
        if self.majority_class_ is None:
            raise RuntimeError("Call fit() before predict().")
        n = len(X)
        return np.full(n, self.majority_class_, dtype=int)

    def predict_proba(self, X) -> np.ndarray:
        """Return one-hot probabilities over self.classes_."""
        if self.classes_ is None or self.majority_class_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        n = len(X)
        k = len(self.classes_)
        proba = np.zeros((n, k))
        idx = int(np.where(self.classes_ == self.majority_class_)[0][0])
        proba[:, idx] = 1.0
        return proba
