"""Threshold classifier (Section 3.6)."""
import numpy as np
import pytest

from pipeline import classifier as clf


@pytest.mark.parametrize("score", [0.70, 0.71, 0.95, 1.0])
def test_score_at_or_above_tau_is_real(score):
    assert clf.classify(score, 0.70) == "Real"


@pytest.mark.parametrize("score", [0.0, 0.3, 0.6999])
def test_score_below_tau_is_deepfake(score):
    assert clf.classify(score, 0.70) == "Deepfake"


@pytest.mark.parametrize("tau", [0.05, 0.3, 0.5, 0.7, 0.95])
def test_confidence_bounded(tau):
    for s in np.linspace(0, 1, 101):
        c = clf.confidence(s, tau)
        assert 0.0 <= c <= 1.0
        assert c >= 0.5


def test_confidence_formula():
    assert clf.confidence(0.70, 0.70) == pytest.approx(0.5)
    assert clf.confidence(0.0, 0.70) == pytest.approx(1.0)            # |0-0.7|/0.7 = 1
    assert clf.confidence(1.0, 0.70) == pytest.approx(0.5 + 0.5 * 0.3 / 0.7)
    # further from tau -> more confident
    assert clf.confidence(0.2, 0.7) > clf.confidence(0.5, 0.7)


def test_invalid_tau():
    for tau in (0, 1, -0.1, 1.5):
        with pytest.raises(ValueError):
            clf.classify(0.5, tau)


def test_opposite():
    assert clf.opposite("Real") == "Deepfake"
    assert clf.opposite("Deepfake") == "Real"
    with pytest.raises(ValueError):
        clf.opposite("Inconclusive")
