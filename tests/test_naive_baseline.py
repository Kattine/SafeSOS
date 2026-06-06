"""Basic sanity tests for the majority classifier."""

import numpy as np
from models.naive_baseline import MajorityClassifier


def test_majority_predicts_most_frequent() -> None:
    X = np.zeros((10, 5))
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 2])  # majority is 0
    clf = MajorityClassifier().fit(X, y)
    preds = clf.predict(X)
    assert (preds == 0).all()


def test_proba_sums_to_one() -> None:
    X = np.zeros((4, 5))
    y = np.array([0, 0, 1, 2])
    clf = MajorityClassifier().fit(X, y)
    proba = clf.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)
