"""Evaluation metrics (Section 3.7) and Cohen's kappa."""
import math

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from evaluation import metrics as m


def test_known_confusion_matrix():
    # TP=3, FN=1, FP=2, TN=4
    y_true = ["Deepfake"] * 4 + ["Real"] * 6
    y_pred = ["Deepfake"] * 3 + ["Real"] + ["Deepfake"] * 2 + ["Real"] * 4
    r = m.classification_metrics(y_true, y_pred)
    assert r["confusion_matrix"] == [[4, 2], [1, 3]]      # rows true [Real, Deepfake]
    assert r["accuracy"] == pytest.approx(7 / 10)
    assert r["precision"] == pytest.approx(3 / 5)
    assert r["recall"] == pytest.approx(3 / 4)
    assert r["f1"] == pytest.approx(2 * 0.6 * 0.75 / (0.6 + 0.75))


def test_perfect_classifier_auc():
    y_true = [0, 0, 1, 1]
    r = m.classification_metrics(y_true, y_true, y_score=[0.1, 0.2, 0.8, 0.9])
    assert r["accuracy"] == 1.0 and r["roc_auc"] == 1.0
    assert r["confusion_matrix"] == [[2, 0], [0, 2]]


def test_auc_none_when_single_class():
    assert m.classification_metrics([1, 1], [1, 0], y_score=[0.9, 0.2])["roc_auc"] is None


def test_agreement_rate():
    assert m.agreement_rate(["Real", "Deepfake", "Real"], ["Real", "Real", "Real"]) == pytest.approx(2 / 3)
    assert m.agreement_rate([], []) is None


def test_kappa_textbook_example():
    # Classic example: 50 items, p_o = 0.7, p_e = 0.5 -> kappa = 0.4
    a = ["Y"] * 20 + ["Y"] * 5 + ["N"] * 10 + ["N"] * 15
    b = ["Y"] * 20 + ["N"] * 5 + ["Y"] * 10 + ["N"] * 15
    assert m.cohens_kappa_manual(a, b) == pytest.approx(0.4)
    assert m.cohens_kappa(a, b) == pytest.approx(0.4)


@pytest.mark.parametrize("seed", range(20))
def test_manual_kappa_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    cats = ["Real", "Deepfake", "Inconclusive"][: rng.integers(2, 4)]
    n = int(rng.integers(5, 60))
    a = list(rng.choice(cats, n))
    b = [x if rng.random() < 0.7 else rng.choice(cats) for x in a]
    expected = cohen_kappa_score(a, b)
    got = m.cohens_kappa_manual(a, b)
    if math.isnan(expected):
        assert math.isnan(got)
    else:
        assert got == pytest.approx(expected)


def test_kappa_degenerate_is_nan():
    assert math.isnan(m.cohens_kappa_manual(["Real"] * 3, ["Real"] * 3))
