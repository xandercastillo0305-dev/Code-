"""Evaluation metrics (Section 3.7, Appendix A).

Positive class for all detection metrics = "Deepfake".

    Accuracy  = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 * Precision * Recall / (Precision + Recall)
    ROC-AUC   = area under the ROC curve of the Deepfake score
    Agreement = (# cases where AI label == analyst label) / (# cases)
    Cohen's k = (p_o - p_e) / (1 - p_e)
"""
from collections import Counter

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

POSITIVE = "Deepfake"
LABELS = ["Real", "Deepfake"]


def _to_binary(labels):
    """Map 'Deepfake'/1/True -> 1, 'Real'/0/False -> 0."""
    out = []
    for y in labels:
        if isinstance(y, str):
            if y not in LABELS:
                raise ValueError(f"Unknown label {y!r}")
            out.append(1 if y == POSITIVE else 0)
        else:
            out.append(int(y))
    return np.asarray(out, dtype=int)


def classification_metrics(y_true, y_pred, y_score=None):
    """Accuracy, precision, recall, F1, ROC-AUC and confusion matrix.

    y_score is the Deepfake-probability (higher = more likely fake); ROC-AUC
    is omitted (None) when it is not given or only one class is present.
    Confusion matrix rows = true [Real, Deepfake], columns = predicted.
    """
    t, p = _to_binary(y_true), _to_binary(y_pred)
    result = {
        "accuracy": float(accuracy_score(t, p)),
        "precision": float(precision_score(t, p, zero_division=0)),
        "recall": float(recall_score(t, p, zero_division=0)),
        "f1": float(f1_score(t, p, zero_division=0)),
        "roc_auc": None,
        "confusion_matrix": confusion_matrix(t, p, labels=[0, 1]).tolist(),
        "n": int(len(t)),
    }
    if y_score is not None and len(set(t.tolist())) == 2:
        result["roc_auc"] = float(roc_auc_score(t, np.asarray(y_score, dtype=float)))
    return result


def agreement_rate(ai_labels, human_labels):
    """Fraction of cases where the AI label equals the analyst's final label."""
    if len(ai_labels) != len(human_labels):
        raise ValueError("Label lists must have the same length.")
    if not ai_labels:
        return None
    return sum(a == h for a, h in zip(ai_labels, human_labels)) / len(ai_labels)


def cohens_kappa_manual(rater_a, rater_b):
    """Cohen's kappa k = (p_o - p_e) / (1 - p_e), implemented from the formula.

    p_o = observed agreement, p_e = agreement expected by chance
        = sum_k P_a(k) * P_b(k).
    Returns nan when p_e == 1 (both raters used a single identical category),
    the same as scikit-learn.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("Rater lists must have the same length.")
    n = len(rater_a)
    if n == 0:
        return float("nan")
    p_o = sum(a == b for a, b in zip(rater_a, rater_b)) / n
    ca, cb = Counter(rater_a), Counter(rater_b)
    p_e = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if np.isclose(p_e, 1.0):
        return float("nan")
    return (p_o - p_e) / (1 - p_e)


def cohens_kappa(rater_a, rater_b):
    """Cohen's kappa via scikit-learn (the reference implementation)."""
    if not rater_a:
        return float("nan")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(cohen_kappa_score(list(rater_a), list(rater_b)))
