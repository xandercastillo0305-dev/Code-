"""AI-Assisted Preliminary Classification (Section 3.3, Section 3.6).

The output of this module is a *preliminary* label only. It is never the
final verdict: every case goes to a forensic analyst (review/workflow.py).
"""
REAL = "Real"
DEEPFAKE = "Deepfake"
INCONCLUSIVE = "Inconclusive"
AI_LABELS = (REAL, DEEPFAKE)


def validate_threshold(tau):
    tau = float(tau)
    if not 0.0 < tau < 1.0:
        raise ValueError(f"Threshold tau must be in (0, 1), got {tau}.")
    return tau


def classify(score, tau):
    """Threshold rule: Real if S >= tau, else Deepfake."""
    tau = validate_threshold(tau)
    return REAL if float(score) >= tau else DEEPFAKE


def confidence(score, tau):
    """Confidence in [0.5, 1] from the distance between S and tau.

        confidence = 0.5 + 0.5 * min(1, |S - tau| / max(tau, 1 - tau))

    * S exactly at tau          -> 0.5 (the model cannot tell)
    * S at the far end of [0,1] -> 1.0 at most
    The denominator max(tau, 1 - tau) is the largest possible distance from
    tau inside [0, 1], so the ratio is normalised to [0, 1] for any tau.
    """
    tau = validate_threshold(tau)
    s = float(score)
    return 0.5 + 0.5 * min(1.0, abs(s - tau) / max(tau, 1.0 - tau))


def preliminary_classification(score, tau):
    """Return (label, confidence)."""
    return classify(score, tau), confidence(score, tau)


def opposite(label):
    if label not in AI_LABELS:
        raise ValueError(f"Unknown label {label!r}.")
    return DEEPFAKE if label == REAL else REAL
