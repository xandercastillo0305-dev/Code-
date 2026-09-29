"""JSON-backed review case queue (Section 3.4, Appendix C).

File layout of data/cases.json:

    {"last_case_number": 3, "cases": [ {<case record>}, ... ]}

* Thread-safe: every read-modify-write runs under one re-entrant lock, and
  on POSIX an fcntl file lock also guards against a second process.
* Atomic save: data is written to a temp file in the same directory, fsynced
  and swapped in with os.replace, so a crash never leaves a half-written file.
* Case IDs are never reused (CASE-0001, CASE-0002, ...), even after a purge.
"""
import contextlib
import copy
import json
import os
import tempfile
import threading
from datetime import datetime, timezone

try:
    import fcntl  # POSIX only
except ImportError:  # Windows: in-process lock only
    fcntl = None

# Appendix C data dictionary, in order.
CASE_FIELDS = (
    "case_id", "submitted_by", "submitted_at", "suspect_image", "reference_image", "status",
    "ai_classification", "confidence_score", "cosine_similarity", "euclidean_similarity", "ssim",
    "aggregated_score", "threshold", "reviewer_id", "review_decision", "final_classification",
    "rationale", "reviewed_at", "report_path",
)
# Permitted additional fields (chain of custody + model provenance).
EXTRA_FIELDS = ("suspect_sha256", "reference_sha256", "model_version", "model_trained")
ALL_FIELDS = CASE_FIELDS + EXTRA_FIELDS


def now_iso():
    """ISO 8601 timestamp in UTC, e.g. 2026-09-29T08:15:00+00:00."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_case_id(n):
    return f"CASE-{n:04d}"


class CaseNotFoundError(KeyError):
    pass


class CaseStore:
    def __init__(self, path):
        self.path = path
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if not os.path.exists(path):
            self._write({"last_case_number": 0, "cases": []})

    # -- low-level I/O -------------------------------------------------------
    @contextlib.contextmanager
    def _locked(self):
        with self._lock:
            if fcntl is None:
                yield
                return
            with open(self.path + ".lock", "a") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)

    def _read(self):
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data):
        directory = os.path.dirname(os.path.abspath(self.path))
        fd, tmp = tempfile.mkstemp(prefix=".cases-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.remove(tmp)
            raise

    # -- public API ------------------------------------------------------------
    def reserve_case_id(self):
        with self._locked():
            data = self._read()
            data["last_case_number"] += 1
            self._write(data)
            return format_case_id(data["last_case_number"])

    def add_case(self, record):
        """Insert a full case record (must contain every Appendix C field)."""
        missing = [f for f in CASE_FIELDS if f not in record]
        unknown = [f for f in record if f not in ALL_FIELDS]
        if missing or unknown:
            raise ValueError(f"Invalid case record. Missing={missing} unknown={unknown}")
        ordered = {k: record[k] for k in ALL_FIELDS if k in record}
        with self._locked():
            data = self._read()
            if any(c["case_id"] == ordered["case_id"] for c in data["cases"]):
                raise ValueError(f"{ordered['case_id']} already exists.")
            data["cases"].append(ordered)
            self._write(data)
        return copy.deepcopy(ordered)

    def get(self, case_id):
        with self._locked():
            for c in self._read()["cases"]:
                if c["case_id"] == case_id:
                    return c
        raise CaseNotFoundError(case_id)

    def list(self, status=None, submitted_by=None):
        with self._locked():
            cases = self._read()["cases"]
        if status:
            cases = [c for c in cases if c["status"] == status]
        if submitted_by:
            cases = [c for c in cases if c["submitted_by"] == submitted_by]
        return cases

    def update(self, case_id, mutate):
        """Atomically apply ``mutate(case) -> new_case`` and save; returns the new case."""
        with self._locked():
            data = self._read()
            for i, c in enumerate(data["cases"]):
                if c["case_id"] == case_id:
                    new = mutate(copy.deepcopy(c))
                    unknown = [f for f in new if f not in ALL_FIELDS]
                    if unknown:
                        raise ValueError(f"Unknown fields {unknown}")
                    data["cases"][i] = new
                    self._write(data)
                    return copy.deepcopy(new)
        raise CaseNotFoundError(case_id)

    def delete(self, case_id):
        with self._locked():
            data = self._read()
            before = len(data["cases"])
            data["cases"] = [c for c in data["cases"] if c["case_id"] != case_id]
            if len(data["cases"]) == before:
                raise CaseNotFoundError(case_id)
            self._write(data)
