"""submit -> pipeline -> review -> report, through the real Flask routes.

MTCNN and the hybrid model are mocked (StubDetector / TinyHybrid) so this runs
on CI without downloads; everything else (upload validation, hashing, case
store, workflow, audit log, PDF generation) is the real code.
"""
import io
import json
import os

import pytest

from app import create_app
from pipeline.preprocessing import FacePreprocessor
from pipeline.verify import verify_media
from review.case_store import CASE_FIELDS, EXTRA_FIELDS
from storage import sha256_file
from tests.conftest import NoFaceDetector, StubDetector, TinyHybrid, make_face_image
from pipeline.model import ModelBundle

PASSWORD = "test-pass-123"
RATIONALE = "Visible blending seam along the hairline and mismatched specular highlights."


def make_app(tmp_path, detector=None):
    bundle = ModelBundle(TinyHybrid().eval(), "cpu", "test-stub-untrained", False, "random-init")
    pre = FacePreprocessor(detector=detector or StubDetector())
    data = tmp_path / "data"
    app = create_app({
        "TESTING": True, "CSRF_ENABLED": False,
        "DATA_DIR": str(data), "UPLOAD_DIR": str(data / "uploads"), "REPORT_DIR": str(data / "reports"),
        "CASES_PATH": str(data / "cases.json"), "USERS_PATH": str(data / "users.json"),
        "AUDIT_LOG_PATH": str(data / "audit_log.jsonl"),
        "HYBRID_WEIGHTS_PATH": str(tmp_path / "none.pt"),
        "VERIFY_FN": lambda s, r, tau: verify_media(s, r, tau, preprocessor=pre, model_bundle=bundle),
    })
    us = app.extensions["user_store"]
    us.create("INV-01", "Investigator", "investigator", PASSWORD)
    us.create("ANA-01", "Analyst", "analyst", PASSWORD)
    us.create("ADM-01", "Admin", "admin", PASSWORD)
    return app


@pytest.fixture
def app(tmp_path):
    return make_app(tmp_path)


def login(client, uid):
    client.post("/logout")
    r = client.post("/login", data={"user_id": uid, "password": PASSWORD})
    assert r.status_code == 302, r.data
    return client


def upload(client, tmp_path, attest=True, sus_name="s.png"):
    s = make_face_image(tmp_path / sus_name, seed=1, variant=1)
    r = make_face_image(tmp_path / "r.png", seed=2)
    data = {"suspect": (open(s, "rb"), sus_name), "reference": (open(r, "rb"), "r.png")}
    if attest:
        data["attest"] = "yes"
    return client.post("/cases/new", data=data, content_type="multipart/form-data"), s, r


def test_full_case_lifecycle(app, tmp_path):
    c = login(app.test_client(), "INV-01")
    resp, s_path, r_path = upload(c, tmp_path)
    assert resp.status_code == 302
    store = app.extensions["case_store"]
    case = store.get("CASE-0001")

    # Appendix C: every field present, pending, hashes recorded
    assert all(f in case for f in CASE_FIELDS) and all(f in case for f in EXTRA_FIELDS)
    assert case["status"] == "pending" and case["submitted_by"] == "INV-01"
    assert case["suspect_sha256"] == sha256_file(s_path)
    assert case["reference_sha256"] == sha256_file(r_path)
    assert case["model_trained"] is False
    assert (case["aggregated_score"] >= case["threshold"]) == (case["ai_classification"] == "Real")
    assert not case["suspect_image"].startswith("static")

    # investigator cannot see a report before review, nor review
    assert c.get("/cases/CASE-0001/review").status_code == 403
    assert c.get("/cases/CASE-0001/report", follow_redirects=False).status_code == 302

    # analyst reviews (Override)
    login(c, "ANA-01")
    page = c.get("/cases/CASE-0001/review")
    assert page.status_code == 200 and b"UNTRAINED MODEL" in page.data and b"AUTOMATED" in page.data
    assert c.get("/cases/CASE-0001/image/suspect/face").status_code == 200
    assert c.post("/cases/CASE-0001/reveal/suspect").status_code == 200
    bad = c.post("/cases/CASE-0001/review", data={"decision": "Override", "rationale": "short"})
    assert bad.status_code == 400 and store.get("CASE-0001")["status"] == "pending"
    ok = c.post("/cases/CASE-0001/review", data={"decision": "Override", "rationale": RATIONALE})
    assert ok.status_code == 302
    case = store.get("CASE-0001")
    assert case["status"] == "verified" and case["reviewer_id"] == "ANA-01"
    assert case["final_classification"] != case["ai_classification"]
    pdf_path = os.path.join(app.config["DATA_DIR"], case["report_path"])
    assert os.path.exists(pdf_path) and open(pdf_path, "rb").read(5) == b"%PDF-"

    # read-only after review
    again = c.post("/cases/CASE-0001/review", data={"decision": "Confirm", "rationale": RATIONALE})
    assert again.status_code == 400 and store.get("CASE-0001")["review_decision"] == "Override"

    # investigator gets the report (HTML + PDF)
    login(c, "INV-01")
    html = c.get("/cases/CASE-0001/report")
    assert html.status_code == 200
    for needle in (b"I. Case Information", b"II. Submitted Images", b"III. AI Analysis",
                   b"IV. Human Review", b"V. Sign-off", b"AUTOMATED", b"HUMAN-VERIFIED", b"UNTRAINED MODEL"):
        assert needle in html.data
    pdf = c.get("/cases/CASE-0001/report.pdf")
    assert pdf.status_code == 200 and pdf.data[:5] == b"%PDF-"
    # investigator only gets blurred face crops, never full / unblurred images
    assert c.get("/cases/CASE-0001/image/suspect/face_blur").status_code == 200
    assert c.get("/cases/CASE-0001/image/suspect/face").status_code == 403
    assert c.post("/cases/CASE-0001/reveal/suspect").status_code == 403

    events = [json.loads(l)["event"] for l in open(app.config["AUDIT_LOG_PATH"])]
    for ev in ("login", "case_submitted", "case_viewed", "image_revealed", "review_submitted",
               "report_viewed", "report_downloaded"):
        assert ev in events


def test_attestation_required(app, tmp_path):
    c = login(app.test_client(), "INV-01")
    resp, _, _ = upload(c, tmp_path, attest=False)
    assert resp.status_code == 400 and app.extensions["case_store"].list() == []


def test_rejects_non_image_upload(app, tmp_path):
    c = login(app.test_client(), "INV-01")
    r = make_face_image(tmp_path / "r.png", seed=2)
    resp = c.post("/cases/new", data={"attest": "yes", "suspect": (io.BytesIO(b"not an image"), "x.png"),
                                      "reference": (open(r, "rb"), "r.png")}, content_type="multipart/form-data")
    assert resp.status_code == 400 and b"not a valid image" in resp.data
    resp = c.post("/cases/new", data={"attest": "yes", "suspect": (io.BytesIO(b"GIF89a"), "x.gif"),
                                      "reference": (open(r, "rb"), "r.png")}, content_type="multipart/form-data")
    assert resp.status_code == 400 and b"only JPG and PNG" in resp.data


def test_no_face_shows_error_and_creates_no_case(tmp_path):
    app = make_app(tmp_path, detector=NoFaceDetector())
    c = login(app.test_client(), "INV-01")
    resp, _, _ = upload(c, tmp_path)
    assert resp.status_code == 422 and b"No face detected. Case cannot be analyzed." in resp.data
    assert app.extensions["case_store"].list() == []
    incoming = os.path.join(app.config["UPLOAD_DIR"], "_incoming")
    assert not os.path.exists(incoming) or os.listdir(incoming) == []


@pytest.mark.parametrize("decision,status,final", [("Confirm", "verified", None), ("Flag", "flagged", "Inconclusive")])
def test_confirm_and_flag_via_web(app, tmp_path, decision, status, final):
    c = login(app.test_client(), "INV-01")
    upload(c, tmp_path)
    login(c, "ANA-01")
    c.post("/cases/CASE-0001/review", data={"decision": decision, "rationale": RATIONALE})
    case = app.extensions["case_store"].get("CASE-0001")
    assert case["status"] == status
    assert case["final_classification"] == (final or case["ai_classification"])
