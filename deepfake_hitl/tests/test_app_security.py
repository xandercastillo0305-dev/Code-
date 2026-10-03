"""Access control, CSRF, admin pages and the purge CLI (Appendix D)."""
import os

from tests.test_end_to_end import PASSWORD, RATIONALE, login, make_app, upload


def test_login_required_everywhere(tmp_path):
    c = make_app(tmp_path).test_client()
    for url in ("/", "/dashboard", "/cases/new", "/cases/mine", "/evaluation", "/admin/users",
                "/cases/CASE-0001/review", "/cases/CASE-0001/report.pdf",
                "/cases/CASE-0001/image/suspect/face_blur"):
        r = c.get(url)
        assert r.status_code == 302 and "/login" in r.headers["Location"], url


def test_role_enforcement(tmp_path):
    app = make_app(tmp_path)
    c = login(app.test_client(), "INV-01")
    upload(c, tmp_path)
    for url in ("/dashboard", "/evaluation", "/admin/users", "/cases/CASE-0001/review"):
        assert c.get(url).status_code == 403, url
    login(c, "ANA-01")
    assert c.get("/cases/new").status_code == 403
    assert c.get("/evaluation").status_code == 403
    login(c, "ADM-01")
    assert c.get("/evaluation").status_code == 200
    assert c.get("/dashboard").status_code == 200
    assert c.get("/cases/CASE-0001/review").status_code == 403       # admins do not review
    assert c.post("/cases/CASE-0001/reveal/suspect").status_code == 403


def test_other_investigator_cannot_see_case(tmp_path):
    app = make_app(tmp_path)
    app.extensions["user_store"].create("INV-02", "Other", "investigator", PASSWORD)
    c = login(app.test_client(), "INV-01")
    upload(c, tmp_path)
    login(c, "ANA-01")
    c.post("/cases/CASE-0001/review", data={"decision": "Confirm", "rationale": RATIONALE})
    login(c, "INV-02")
    assert c.get("/cases/CASE-0001/report").status_code == 403
    assert c.get("/cases/CASE-0001/image/suspect/face_blur").status_code == 403


def test_csrf_enforced(tmp_path):
    app = make_app(tmp_path)
    app.config["CSRF_ENABLED"] = True
    c = app.test_client()
    assert c.post("/login", data={"user_id": "INV-01", "password": PASSWORD, "role": "investigator"}).status_code == 400
    c.get("/login")
    with c.session_transaction() as s:
        token = s["_csrf"]
    r = c.post("/login", data={"user_id": "INV-01", "password": PASSWORD, "role": "investigator",
                               "csrf_token": token})
    assert r.status_code == 302


def test_uploads_not_under_static(tmp_path):
    import config
    assert not os.path.abspath(config.UPLOAD_DIR).startswith(os.path.abspath(os.path.join(config.BASE_DIR, "static")))


def test_purge_case_cli(tmp_path):
    app = make_app(tmp_path)
    c = login(app.test_client(), "INV-01")
    upload(c, tmp_path)
    login(c, "ANA-01")
    c.post("/cases/CASE-0001/review", data={"decision": "Confirm", "rationale": RATIONALE})
    case = app.extensions["case_store"].get("CASE-0001")
    folder = os.path.join(app.config["UPLOAD_DIR"], "CASE-0001")
    pdf = os.path.join(app.config["DATA_DIR"], case["report_path"])
    assert os.path.isdir(folder) and os.path.exists(pdf)

    runner = app.test_cli_runner()
    denied = runner.invoke(args=["purge-case", "CASE-0001", "--admin-id", "ANA-01", "--password", PASSWORD])
    assert denied.exit_code != 0 and os.path.isdir(folder)
    ok = runner.invoke(args=["purge-case", "CASE-0001", "--admin-id", "ADM-01", "--password", PASSWORD])
    assert ok.exit_code == 0, ok.output
    assert not os.path.exists(folder) and not os.path.exists(pdf)
    assert app.extensions["case_store"].list() == []
    log = open(app.config["AUDIT_LOG_PATH"]).read()
    assert '"case_deleted"' in log and '"ADM-01"' in log


def test_purge_all_cli(tmp_path):
    app = make_app(tmp_path)
    c = login(app.test_client(), "INV-01")
    upload(c, tmp_path)
    upload(c, tmp_path, sus_name="s2.png")
    r = app.test_cli_runner().invoke(args=["purge-all", "--admin-id", "ADM-01", "--password", PASSWORD, "--yes"])
    assert r.exit_code == 0, r.output
    assert app.extensions["case_store"].list() == []


def test_admin_can_create_and_disable_users(tmp_path):
    app = make_app(tmp_path)
    c = login(app.test_client(), "ADM-01")
    c.post("/admin/users", data={"action": "create", "user_id": "ana-09", "name": "New", "role": "analyst",
                                 "password": "longpassword"})
    assert app.extensions["user_store"].get("ANA-09").role == "analyst"
    c.post("/admin/users", data={"action": "disable", "user_id": "ANA-09"})
    assert app.extensions["user_store"].authenticate("ANA-09", "longpassword") is None


def test_login_role_must_match_account(tmp_path):
    app = make_app(tmp_path)
    c = app.test_client()
    page = c.get("/login").data
    assert b"Log in as" in page and b"Investigator" in page and b"Analyst" in page
    # right password, wrong role -> refused and logged
    r = c.post("/login", data={"user_id": "ANA-01", "password": PASSWORD, "role": "investigator"})
    assert r.status_code == 401 and b"Invalid user ID, password, or role." in r.data
    assert c.get("/dashboard").status_code == 302                       # still not logged in
    r = c.post("/login", data={"user_id": "ANA-01", "password": PASSWORD})   # no role chosen
    assert r.status_code == 401
    assert '"role_mismatch"' in open(app.config["AUDIT_LOG_PATH"]).read()
    # right role -> logged in
    r = c.post("/login", data={"user_id": "ANA-01", "password": PASSWORD, "role": "analyst"})
    assert r.status_code == 302 and c.get("/dashboard").status_code == 200


def test_admin_can_delete_case_from_web(tmp_path):
    app = make_app(tmp_path)
    c = login(app.test_client(), "INV-01")
    upload(c, tmp_path)
    folder = os.path.join(app.config["UPLOAD_DIR"], "CASE-0001")
    assert os.path.isdir(folder)
    # investigators and analysts cannot delete
    assert c.post("/cases/CASE-0001/delete", data={"reason": "testing"}).status_code == 403
    login(c, "ANA-01")
    assert c.post("/cases/CASE-0001/delete", data={"reason": "testing"}).status_code == 403
    # admin sees the button; a reason is required
    login(c, "ADM-01")
    assert b"Delete case" in c.get("/dashboard").data
    c.post("/cases/CASE-0001/delete", data={"reason": ""})
    assert app.extensions["case_store"].get("CASE-0001")
    r = c.post("/cases/CASE-0001/delete", data={"reason": "Data-retention policy"})
    assert r.status_code == 302
    assert app.extensions["case_store"].list() == [] and not os.path.exists(folder)
    log = open(app.config["AUDIT_LOG_PATH"]).read()
    assert '"case_deleted"' in log and "Data-retention policy" in log
    assert c.post("/cases/CASE-0001/delete", data={"reason": "again please"}).status_code == 404
