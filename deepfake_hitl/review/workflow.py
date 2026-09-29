"""Human-in-the-Loop review workflow (Section 3.4).

State machine:

    pending --Confirm--> verified   final_classification = ai_classification
    pending --Override-> verified   final_classification = the other class
    pending --Flag-----> flagged    final_classification = "Inconclusive"

verified and flagged are terminal (read-only). A rationale is always required,
and an analyst may not review a case they submitted.
"""
import config
from pipeline.classifier import INCONCLUSIVE, opposite
from review.case_store import now_iso

PENDING, VERIFIED, FLAGGED = "pending", "verified", "flagged"
STATUSES = (PENDING, VERIFIED, FLAGGED)
CONFIRM, OVERRIDE, FLAG = "Confirm", "Override", "Flag"
DECISIONS = (CONFIRM, OVERRIDE, FLAG)


class ReviewError(ValueError):
    """Raised for any invalid review (bad state, missing rationale, conflict of interest)."""


def validate_rationale(rationale, min_length=config.MIN_RATIONALE_LENGTH):
    text = (rationale or "").strip()
    if len(text) < min_length:
        raise ReviewError(f"A rationale of at least {min_length} characters is required.")
    return text


def apply_review(case, reviewer_id, decision, rationale, override_classification=None,
                 reviewed_at=None, min_rationale_length=config.MIN_RATIONALE_LENGTH):
    """Pure function: return the updated case dict or raise ReviewError."""
    if case["status"] != PENDING:
        raise ReviewError(f"{case['case_id']} is {case['status']} and can no longer be reviewed.")
    if not reviewer_id:
        raise ReviewError("Reviewer ID is required.")
    if reviewer_id == case["submitted_by"]:
        raise ReviewError("An analyst cannot review a case they submitted.")
    if decision not in DECISIONS:
        raise ReviewError(f"Decision must be one of {DECISIONS}.")
    text = validate_rationale(rationale, min_rationale_length)

    ai = case["ai_classification"]
    if decision == CONFIRM:
        final, status = ai, VERIFIED
    elif decision == OVERRIDE:
        final = opposite(ai)
        if override_classification not in (None, "", final):
            raise ReviewError(f"Override must change the classification from {ai} to {final}.")
        status = VERIFIED
    else:
        final, status = INCONCLUSIVE, FLAGGED

    case.update({
        "status": status,
        "reviewer_id": reviewer_id,
        "review_decision": decision,
        "final_classification": final,
        "rationale": text,
        "reviewed_at": reviewed_at or now_iso(),
    })
    return case


def submit_review(store, case_id, reviewer_id, decision, rationale, override_classification=None):
    """Apply the review atomically inside the case store."""
    return store.update(case_id, lambda c: apply_review(c, reviewer_id, decision, rationale,
                                                         override_classification))


def is_read_only(case):
    return case["status"] in (VERIFIED, FLAGGED)
