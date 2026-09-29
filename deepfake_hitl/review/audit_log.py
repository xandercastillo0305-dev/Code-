"""Append-only audit log (Appendix D), one JSON object per line.

Events: login, logout, login_failed, case_submitted, case_rejected, case_viewed,
image_revealed, review_submitted, report_viewed, report_downloaded,
case_deleted, user_created, user_disabled.
"""
import json
import os
import threading

from review.case_store import now_iso

_lock = threading.Lock()

EVENTS = {
    "login", "logout", "login_failed", "case_submitted", "case_rejected", "case_viewed",
    "image_revealed", "review_submitted", "report_viewed", "report_downloaded", "case_deleted",
    "user_created", "user_disabled",
}


def log_event(path, event, user_id, case_id=None, **details):
    if event not in EVENTS:
        raise ValueError(f"Unknown audit event {event!r}")
    entry = {"timestamp": now_iso(), "event": event, "user": user_id}
    if case_id:
        entry["case_id"] = case_id
    if details:
        entry["details"] = details
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with _lock:
        # "a" mode = append-only; existing lines are never rewritten.
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
    return entry


def read_events(path, case_id=None):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    if case_id:
        events = [e for e in events if e.get("case_id") == case_id]
    return events
