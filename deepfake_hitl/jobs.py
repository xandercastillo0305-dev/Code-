"""Background processing for case submissions.

A submission with many images (e.g. a whole folder) can take minutes on a
CPU, longer than a browser should wait for one request. The upload request
therefore only validates and stores the files, then hands the AI analysis to
a single background worker and sends the investigator to a progress page.

* One worker thread: submissions are analysed one after another, so two big
  uploads never compete for the CPU / memory at the same time.
* Jobs are kept in memory only. They are short-lived; the case record (the
  thing that matters) is written to the JSON case store when the job ends.
* ``sync=True`` (used by the tests) runs the job immediately in the request.
"""
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from review.case_store import now_iso

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"


class JobRegistry:
    def __init__(self, sync=False):
        self._jobs = {}
        self._lock = threading.Lock()
        self._sync = sync
        self._pool = None if sync else ThreadPoolExecutor(max_workers=1, thread_name_prefix="case-job")

    def create(self, user_id, n_suspect, n_reference, skipped):
        job = {"id": uuid.uuid4().hex[:12], "user_id": user_id, "created_at": now_iso(), "state": QUEUED,
               "stage": "Waiting to start", "done": 0, "total": 0, "case_id": None, "error": None,
               "n_suspect": n_suspect, "n_reference": n_reference, "skipped": list(skipped)}
        with self._lock:
            self._jobs[job["id"]] = job
        return dict(job)

    def get(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def update(self, job_id, **fields):
        with self._lock:
            self._jobs[job_id].update(fields)

    def queue_position(self, job_id):
        with self._lock:
            waiting = [j for j in self._jobs.values() if j["state"] in (QUEUED, RUNNING)]
            waiting.sort(key=lambda j: j["created_at"])
            ids = [j["id"] for j in waiting]
            return ids.index(job_id) if job_id in ids else 0

    def submit(self, job_id, fn):
        """Run ``fn(progress)`` for the job; it returns the new case ID."""
        def run():
            self.update(job_id, state=RUNNING, stage="Starting")

            def progress(stage, done, total):
                self.update(job_id, stage=stage, done=done, total=total)
            try:
                case_id = fn(progress)
                self.update(job_id, state=DONE, stage="Finished", case_id=case_id)
            except Exception as exc:          # reported on the progress page
                self.update(job_id, state=FAILED, stage="Stopped", error=str(exc) or exc.__class__.__name__)
        if self._sync:
            run()
        else:
            self._pool.submit(run)
