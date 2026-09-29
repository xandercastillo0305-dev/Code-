"""Human-in-the-loop state transitions (Section 3.4) and the JSON case store."""
import json
import threading

import pytest

from review import workflow as wf
from review.audit_log import log_event, read_events
from review.case_store import CASE_FIELDS, CaseStore

RATIONALE = "Blending artefacts along the jawline and inconsistent lighting on the left cheek."


def new_case(store, ai="Deepfake", submitted_by="INV-01"):
    cid = store.reserve_case_id()
    record = {f: None for f in CASE_FIELDS}
    record.update(case_id=cid, submitted_by=submitted_by, submitted_at="2026-09-29T00:00:00+00:00",
                  status="pending", ai_classification=ai, confidence_score=0.8,
                  cosine_similarity=0.6, euclidean_similarity=0.5, ssim=0.4,
                  aggregated_score=0.5, threshold=0.7, model_trained=False)
    return store.add_case(record)


@pytest.fixture
def store(tmp_path):
    return CaseStore(str(tmp_path / "cases.json"))


def test_confirm_sets_verified_and_keeps_ai_label(store):
    c = new_case(store, ai="Deepfake")
    out = wf.submit_review(store, c["case_id"], "ANA-01", "Confirm", RATIONALE)
    assert out["status"] == "verified"
    assert out["final_classification"] == "Deepfake"
    assert out["review_decision"] == "Confirm" and out["reviewer_id"] == "ANA-01"
    assert out["reviewed_at"]


@pytest.mark.parametrize("ai,expected", [("Deepfake", "Real"), ("Real", "Deepfake")])
def test_override_flips_final_class(store, ai, expected):
    c = new_case(store, ai=ai)
    out = wf.submit_review(store, c["case_id"], "ANA-01", "Override", RATIONALE)
    assert out["status"] == "verified" and out["final_classification"] == expected
    assert out["ai_classification"] == ai            # AI result preserved separately


def test_override_to_same_class_rejected(store):
    c = new_case(store, ai="Real")
    with pytest.raises(wf.ReviewError):
        wf.submit_review(store, c["case_id"], "ANA-01", "Override", RATIONALE, "Real")


def test_flag_sets_flagged_inconclusive(store):
    c = new_case(store)
    out = wf.submit_review(store, c["case_id"], "ANA-01", "Flag", RATIONALE)
    assert out["status"] == "flagged" and out["final_classification"] == "Inconclusive"


@pytest.mark.parametrize("rationale", [None, "", "   ", "too short"])
def test_rationale_required(store, rationale):
    c = new_case(store)
    with pytest.raises(wf.ReviewError):
        wf.submit_review(store, c["case_id"], "ANA-01", "Confirm", rationale)
    assert store.get(c["case_id"])["status"] == "pending"


@pytest.mark.parametrize("decision", ["Confirm", "Flag"])
def test_reviewed_case_cannot_be_re_reviewed(store, decision):
    c = new_case(store)
    wf.submit_review(store, c["case_id"], "ANA-01", decision, RATIONALE)
    with pytest.raises(wf.ReviewError):
        wf.submit_review(store, c["case_id"], "ANA-02", "Override", RATIONALE)


def test_analyst_cannot_review_own_submission(store):
    c = new_case(store, submitted_by="ANA-01")
    with pytest.raises(wf.ReviewError):
        wf.submit_review(store, c["case_id"], "ANA-01", "Confirm", RATIONALE)


def test_invalid_decision(store):
    c = new_case(store)
    with pytest.raises(wf.ReviewError):
        wf.submit_review(store, c["case_id"], "ANA-01", "Approve", RATIONALE)


def test_case_ids_sequential_and_never_reused(store):
    a, b = new_case(store), new_case(store)
    assert (a["case_id"], b["case_id"]) == ("CASE-0001", "CASE-0002")
    store.delete("CASE-0002")
    assert new_case(store)["case_id"] == "CASE-0003"


def test_record_has_exact_appendix_c_fields(store):
    c = new_case(store)
    assert list(c)[: len(CASE_FIELDS)] == list(CASE_FIELDS)
    with pytest.raises(ValueError):
        store.add_case({"case_id": "CASE-9999"})


def test_concurrent_writes_are_safe(store):
    ids = []
    threads = [threading.Thread(target=lambda: ids.append(new_case(store)["case_id"])) for _ in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(set(ids)) == 20 and len(store.list()) == 20
    json.load(open(store.path))  # file still valid JSON


def test_audit_log_is_append_only(tmp_path):
    p = str(tmp_path / "audit.jsonl")
    log_event(p, "login", "ANA-01")
    log_event(p, "image_revealed", "ANA-01", case_id="CASE-0001", which="suspect")
    ev = read_events(p)
    assert [e["event"] for e in ev] == ["login", "image_revealed"]
    assert read_events(p, "CASE-0001")[0]["details"] == {"which": "suspect"}
    with pytest.raises(ValueError):
        log_event(p, "not_an_event", "X")
