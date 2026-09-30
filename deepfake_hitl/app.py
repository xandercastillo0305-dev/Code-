"""Flask web application: Input Module, Reviewer Dashboard, Review Interface,
Forensic Report and admin pages (Sections 3.3-3.4, Appendices B-D).

Run:   flask --app app run
CLI:   flask --app app seed-users | purge-case CASE-0001 | purge-all
"""
import functools
import hmac
import os
import secrets
import shutil
import uuid

import click
from flask import (Flask, Response, abort, flash, redirect, render_template, request,
                   send_file, session, url_for)
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
from PIL import Image

import config
import storage
from evaluation.evaluate_agreement import agreement_summary
from pipeline.classifier import opposite
from pipeline.preprocessing import NoFaceDetectedError
from reports.report_generator import UNTRAINED_BANNER, build_report_data, generate_pdf, report_path_for
from review import workflow
from review.audit_log import log_event, read_events
from review.case_store import CaseNotFoundError, CaseStore, now_iso
from users import ADMIN, ANALYST, INVESTIGATOR, ROLES, UserStore

# "Log in as" choices on the login page: (role value, label)
LOGIN_ROLES = [
    (INVESTIGATOR, "Investigator"),
    (ANALYST, "Analyst"),
    (ADMIN, "Administrator"),
]

ATTESTATION_TEXT = ("I attest that the subject is an adult (18+) and that I am authorized "
                    "to submit this material.")


def default_verify(suspect_path, reference_path, tau):
    from pipeline.verify import verify_media
    return verify_media(suspect_path, reference_path, tau)


def create_app(overrides=None):
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=config.SECRET_KEY,
        DATA_DIR=config.DATA_DIR,
        UPLOAD_DIR=config.UPLOAD_DIR,
        REPORT_DIR=config.REPORT_DIR,
        CASES_PATH=config.CASES_PATH,
        USERS_PATH=config.USERS_PATH,
        AUDIT_LOG_PATH=config.AUDIT_LOG_PATH,
        HYBRID_WEIGHTS_PATH=config.HYBRID_WEIGHTS_PATH,
        THRESHOLD_TAU=config.THRESHOLD_TAU,
        MIN_RATIONALE_LENGTH=config.MIN_RATIONALE_LENGTH,
        MAX_CONTENT_LENGTH=(2 * config.MAX_UPLOAD_MB + 1) * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        CSRF_ENABLED=True,
        VERIFY_FN=default_verify,
        MODEL_TRAINED=None,          # None = detect from weights file
    )
    if overrides:
        app.config.update(overrides)
    for key in ("UPLOAD_DIR", "REPORT_DIR"):
        os.makedirs(app.config[key], exist_ok=True)

    cases = CaseStore(app.config["CASES_PATH"])
    users = UserStore(app.config["USERS_PATH"])
    app.extensions["case_store"] = cases
    app.extensions["user_store"] = users

    app.jinja_env.globals["opposite"] = opposite

    login_manager = LoginManager(app)
    login_manager.login_view = "login"
    login_manager.login_message_category = "warning"
    login_manager.user_loader(users.get)

    def audit(event, case_id=None, user_id=None, **details):
        uid = user_id or (current_user.id if current_user.is_authenticated else "anonymous")
        log_event(app.config["AUDIT_LOG_PATH"], event, uid, case_id, **details)

    def model_trained():
        if app.config["MODEL_TRAINED"] is not None:
            return app.config["MODEL_TRAINED"]
        return os.path.exists(app.config["HYBRID_WEIGHTS_PATH"])

    # ---- CSRF (session token, checked on every POST) --------------------------
    def csrf_token():
        if "_csrf" not in session:
            session["_csrf"] = secrets.token_urlsafe(32)
        return session["_csrf"]

    @app.before_request
    def check_csrf():
        if request.method == "POST" and app.config["CSRF_ENABLED"]:
            sent = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token", "")
            if not sent or not hmac.compare_digest(sent, session.get("_csrf", "")):
                abort(400, "Invalid or missing CSRF token. Reload the page and try again.")

    @app.context_processor
    def inject_globals():
        return {
            "csrf_token": csrf_token,
            "model_untrained": not model_trained(),
            "untrained_banner": UNTRAINED_BANNER,
            "min_rationale": app.config["MIN_RATIONALE_LENGTH"],
            "tau": app.config["THRESHOLD_TAU"],
            "max_upload_mb": config.MAX_UPLOAD_MB,
            "metric_weights": config.METRIC_WEIGHTS,
        }

    # ---- access-control helpers ------------------------------------------------
    def roles_required(*roles):
        def deco(fn):
            @functools.wraps(fn)
            @login_required
            def wrapper(*a, **kw):
                if current_user.role not in roles:
                    abort(403)
                return fn(*a, **kw)
            return wrapper
        return deco

    def load_case(case_id):
        try:
            return cases.get(case_id)
        except CaseNotFoundError:
            abort(404)

    def can_view_case(case):
        """Submitting investigator, any analyst, or admin."""
        if current_user.role in (ANALYST, ADMIN):
            return True
        return current_user.role == INVESTIGATOR and case["submitted_by"] == current_user.id

    # ---- auth -----------------------------------------------------------------
    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            uid = request.form.get("user_id", "").strip().upper()
            role = request.form.get("role", "")
            user = users.authenticate(uid, request.form.get("password", ""))
            # The chosen "Log in as" role must match the account's role.
            if user and user.role == role:
                login_user(user)
                session["_csrf"] = secrets.token_urlsafe(32)
                audit("login", user_id=user.id, role=role)
                return redirect(url_for("index"))
            reason = "role_mismatch" if user else "bad_credentials"
            audit("login_failed", user_id=uid or "unknown", role=role or None, reason=reason)
            flash("Invalid user ID, password, or role.", "danger")
            return render_template("login.html", login_roles=LOGIN_ROLES,
                                   selected_role=role, user_id=uid), 401
        return render_template("login.html", login_roles=LOGIN_ROLES, selected_role="", user_id="")

    @app.route("/logout", methods=["POST"])
    @login_required
    def logout():
        audit("logout")
        logout_user()
        return redirect(url_for("login"))

    @app.route("/")
    @login_required
    def index():
        if current_user.role == INVESTIGATOR:
            return redirect(url_for("my_cases"))
        return redirect(url_for("dashboard"))

    # ---- Input Module: case submission (investigator) ----------------------------
    @app.route("/cases/new", methods=["GET", "POST"])
    @roles_required(INVESTIGATOR)
    def submit_case():
        if request.method == "GET":
            return render_template("submit.html", attestation=ATTESTATION_TEXT)
        if request.form.get("attest") != "yes":
            flash("You must confirm the adult-subject and authorization attestation.", "danger")
            return render_template("submit.html", attestation=ATTESTATION_TEXT), 400

        files = {}
        try:
            for which in storage.ROLES_IMAGES:
                f = request.files.get(which)
                if f is None or not f.filename:
                    raise storage.UploadError(f"The {which} image is required.")
                data = f.read()
                files[which] = (storage.validate_image_bytes(f.filename, data), data)
        except storage.UploadError as exc:
            flash(str(exc), "danger")
            return render_template("submit.html", attestation=ATTESTATION_TEXT), 400

        incoming = os.path.join(app.config["UPLOAD_DIR"], "_incoming", uuid.uuid4().hex)
        os.makedirs(incoming)
        paths, hashes = {}, {}
        for which, (ext, data) in files.items():
            paths[which] = os.path.join(incoming, which + ext)
            with open(paths[which], "wb") as out:
                out.write(data)
            hashes[which] = storage.sha256_bytes(data)       # chain of custody

        try:
            result = app.config["VERIFY_FN"](paths["suspect"], paths["reference"],
                                             app.config["THRESHOLD_TAU"])
        except NoFaceDetectedError as exc:
            storage.secure_delete_tree(incoming)
            audit("case_rejected", reason=str(exc), suspect_sha256=hashes["suspect"],
                  reference_sha256=hashes["reference"])
            flash(str(exc), "danger")
            return render_template("submit.html", attestation=ATTESTATION_TEXT), 422

        case_id = cases.reserve_case_id()
        final_dir = storage.case_dir(case_id, app.config["UPLOAD_DIR"])
        shutil.move(incoming, final_dir)
        for which in storage.ROLES_IMAGES:
            result[f"{which}_face"].save(storage.face_path(case_id, which, app.config["UPLOAD_DIR"]))

        rel = lambda which: os.path.relpath(  # noqa: E731
            os.path.join(final_dir, os.path.basename(paths[which])), app.config["DATA_DIR"])
        record = {
            "case_id": case_id,
            "submitted_by": current_user.id,
            "submitted_at": now_iso(),
            "suspect_image": rel("suspect"),
            "reference_image": rel("reference"),
            "status": workflow.PENDING,
            "ai_classification": result["ai_classification"],
            "confidence_score": result["confidence_score"],
            "cosine_similarity": result["cosine_similarity"],
            "euclidean_similarity": result["euclidean_similarity"],
            "ssim": result["ssim"],
            "aggregated_score": result["aggregated_score"],
            "threshold": result["threshold"],
            "reviewer_id": None,
            "review_decision": None,
            "final_classification": None,
            "rationale": None,
            "reviewed_at": None,
            "report_path": None,
            "suspect_sha256": hashes["suspect"],
            "reference_sha256": hashes["reference"],
            "model_version": result["model_version"],
            "model_trained": result["model_trained"],
        }
        cases.add_case(record)
        audit("case_submitted", case_id, suspect_sha256=hashes["suspect"],
              reference_sha256=hashes["reference"], attestation=True)
        flash(f"{case_id} submitted and queued for analyst review.", "success")
        return redirect(url_for("my_cases"))

    @app.route("/cases/mine")
    @roles_required(INVESTIGATOR)
    def my_cases():
        return render_template("my_cases.html", cases=cases.list(submitted_by=current_user.id))

    # ---- Reviewer dashboard ---------------------------------------------------------
    @app.route("/dashboard")
    @roles_required(ANALYST, ADMIN)
    def dashboard():
        status = request.args.get("status") or None
        if status and status not in workflow.STATUSES:
            abort(400)
        all_cases = cases.list()
        shown = [c for c in all_cases if not status or c["status"] == status]
        return render_template("dashboard.html", cases=shown, status=status,
                               summary=agreement_summary(all_cases))

    # ---- Review interface (analyst) ---------------------------------------------------
    @app.route("/cases/<case_id>/review", methods=["GET", "POST"])
    @roles_required(ANALYST)
    def review_case(case_id):
        case = load_case(case_id)
        if request.method == "POST":
            try:
                case = workflow.submit_review(
                    cases, case_id, current_user.id, request.form.get("decision"),
                    request.form.get("rationale"), request.form.get("override_classification"))
            except workflow.ReviewError as exc:
                flash(str(exc), "danger")
                return render_template("review.html", case=case, form=request.form,
                                       conflict=case["submitted_by"] == current_user.id), 400
            pdf = generate_pdf(case, report_path_for(case_id, app.config["REPORT_DIR"]),
                               users=users.names(), upload_dir=app.config["UPLOAD_DIR"])
            rel_pdf = os.path.relpath(pdf, app.config["DATA_DIR"])
            cases.update(case_id, lambda c: {**c, "report_path": rel_pdf})
            audit("review_submitted", case_id, decision=case["review_decision"],
                  final_classification=case["final_classification"])
            flash(f"Review recorded for {case_id}. Forensic report generated.", "success")
            return redirect(url_for("view_report", case_id=case_id))
        audit("case_viewed", case_id)
        return render_template("review.html", case=case, form={},
                               conflict=case["submitted_by"] == current_user.id)

    # ---- Evidence images (never from static/) -----------------------------------------
    def _image_response(img, max_side=None):
        resp = Response(storage.image_png_bytes(img, max_side), mimetype="image/png")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    def _original_path(case, which):
        return os.path.join(app.config["DATA_DIR"], case[f"{which}_image"])

    @app.route("/cases/<case_id>/image/<which>/<variant>")
    @login_required
    def case_image(case_id, which, variant):
        case = load_case(case_id)
        if which not in storage.ROLES_IMAGES or not can_view_case(case):
            abort(404 if which not in storage.ROLES_IMAGES else 403)
        if variant == "face_blur":                                  # everyone allowed on the case
            with Image.open(storage.face_path(case_id, which, app.config["UPLOAD_DIR"])) as im:
                return _image_response(storage.redact(im, "blur"), 300)
        if current_user.role != ANALYST:
            abort(403)
        if variant == "face":                                       # analyst: aligned face crop
            with Image.open(storage.face_path(case_id, which, app.config["UPLOAD_DIR"])) as im:
                return _image_response(im.convert("RGB"))
        if variant == "full_pixelated":                             # analyst: redacted full image
            with Image.open(_original_path(case, which)) as im:
                return _image_response(storage.redact(im, "pixelate"), 640)
        abort(404)

    @app.route("/cases/<case_id>/reveal/<which>", methods=["POST"])
    @roles_required(ANALYST)
    def reveal_image(case_id, which):
        """Full, unredacted image. POST-only so it cannot be prefetched; every reveal is logged."""
        case = load_case(case_id)
        if which not in storage.ROLES_IMAGES:
            abort(404)
        audit("image_revealed", case_id, which=which)
        with Image.open(_original_path(case, which)) as im:
            return _image_response(im.convert("RGB"), 1600)

    # ---- Forensic report ---------------------------------------------------------------
    def _reviewed_case_or_404(case_id):
        case = load_case(case_id)
        if not can_view_case(case):
            abort(403)
        if case["status"] == workflow.PENDING:
            flash(f"{case_id} has not been reviewed yet; no report is available.", "warning")
            return None
        return case

    @app.route("/cases/<case_id>/report")
    @login_required
    def view_report(case_id):
        case = _reviewed_case_or_404(case_id)
        if case is None:
            return redirect(url_for("index"))
        audit("report_viewed", case_id)
        return render_template("report.html", case=case,
                               report=build_report_data(case, users.names()))

    @app.route("/cases/<case_id>/report.pdf")
    @login_required
    def download_report(case_id):
        case = _reviewed_case_or_404(case_id)
        if case is None:
            return redirect(url_for("index"))
        path = os.path.join(app.config["DATA_DIR"], case["report_path"] or "")
        if not case["report_path"] or not os.path.exists(path):
            path = generate_pdf(case, report_path_for(case_id, app.config["REPORT_DIR"]),
                                users=users.names(), upload_dir=app.config["UPLOAD_DIR"])
        audit("report_downloaded", case_id)
        return send_file(path, mimetype="application/pdf", as_attachment=True,
                         download_name=f"{case_id}_forensic_report.pdf")

    # ---- Admin ---------------------------------------------------------------------------
    @app.route("/evaluation")
    @roles_required(ADMIN)
    def evaluation():
        return render_template("evaluation.html", summary=agreement_summary(cases.list()))

    @app.route("/admin/users", methods=["GET", "POST"])
    @roles_required(ADMIN)
    def manage_users():
        if request.method == "POST":
            action = request.form.get("action")
            try:
                if action == "create":
                    u = users.create(request.form.get("user_id"), request.form.get("name"),
                                     request.form.get("role"), request.form.get("password"))
                    audit("user_created", target=u.id, role=u.role)
                    flash(f"User {u.id} created.", "success")
                elif action in ("disable", "enable"):
                    target = request.form.get("user_id")
                    if target == current_user.id:
                        raise ValueError("You cannot disable your own account.")
                    users.set_active(target, action == "enable")
                    audit("user_disabled", target=target, active=action == "enable")
                    flash(f"User {target} {action}d.", "success")
            except (ValueError, KeyError) as exc:
                flash(str(exc), "danger")
            return redirect(url_for("manage_users"))
        return render_template("users.html", users=users.all(), roles=ROLES)

    @app.route("/admin/audit")
    @roles_required(ADMIN)
    def audit_view():
        events = read_events(app.config["AUDIT_LOG_PATH"])[-300:][::-1]
        return render_template("audit.html", events=events)

    @app.errorhandler(403)
    def forbidden(e):
        return render_template("error.html", code=403, message="You do not have access to this page."), 403

    @app.errorhandler(404)
    def not_found(e):
        return render_template("error.html", code=404, message="Not found."), 404

    @app.errorhandler(400)
    def bad_request(e):
        return render_template("error.html", code=400, message=getattr(e, "description", "Bad request.")), 400

    @app.errorhandler(413)
    def too_large(e):
        return render_template("error.html", code=413,
                               message=f"Upload too large (max {config.MAX_UPLOAD_MB} MB per image)."), 413

    register_cli(app, cases, users, audit)
    return app


# ---------------------------------------------------------------------------------------
# CLI: seeding + data-retention purge (Appendix D)
# ---------------------------------------------------------------------------------------
def _require_admin(users, admin_id, password):
    user = users.authenticate(admin_id, password)
    if not user or user.role != ADMIN:
        raise click.ClickException("Admin authentication failed. Purge refused.")
    return user


def purge_case(app, case_id, admin_id):
    """Securely delete a case's images and report and remove the record."""
    cases = app.extensions["case_store"]
    case = cases.get(case_id)
    removed = storage.secure_delete_tree(storage.case_dir(case_id, app.config["UPLOAD_DIR"]))
    if case.get("report_path"):
        removed += storage.secure_delete_file(os.path.join(app.config["DATA_DIR"], case["report_path"]))
    cases.delete(case_id)
    log_event(app.config["AUDIT_LOG_PATH"], "case_deleted", admin_id, case_id,
              files_removed=removed, suspect_sha256=case.get("suspect_sha256"),
              reference_sha256=case.get("reference_sha256"), reason="data-retention purge")
    return removed


def register_cli(app, cases, users, audit):
    @app.cli.command("seed-users")
    @click.option("--password", default=None, help="Password for all demo accounts.")
    def seed_users_cmd(password):
        """Create demo accounts INV-01, ANA-01, ANA-02 and ADM-01."""
        from seed_users import seed
        seed(users, password)

    @app.cli.command("purge-case")
    @click.argument("case_id")
    @click.option("--admin-id", prompt="Admin user ID")
    @click.option("--password", prompt=True, hide_input=True)
    def purge_case_cmd(case_id, admin_id, password):
        """Securely delete one case's images, report and record."""
        admin = _require_admin(users, admin_id, password)
        try:
            n = purge_case(app, case_id.upper(), admin.id)
        except CaseNotFoundError:
            raise click.ClickException(f"{case_id} not found.")
        click.echo(f"{case_id.upper()} purged ({n} files securely deleted).")

    @app.cli.command("purge-all")
    @click.option("--admin-id", prompt="Admin user ID")
    @click.option("--password", prompt=True, hide_input=True)
    @click.confirmation_option(prompt="This permanently deletes ALL cases, images and reports. Continue?")
    def purge_all_cmd(admin_id, password):
        """Securely delete every case (data-retention policy)."""
        admin = _require_admin(users, admin_id, password)
        ids = [c["case_id"] for c in cases.list()]
        for cid in ids:
            purge_case(app, cid, admin.id)
        storage.secure_delete_tree(os.path.join(app.config["UPLOAD_DIR"], "_incoming"))
        click.echo(f"Purged {len(ids)} cases.")

