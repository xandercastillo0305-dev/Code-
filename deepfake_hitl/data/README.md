Runtime data folder (git-ignored). Created automatically on first run:

    cases.json        JSON case queue (Appendix C records)
    users.json        user accounts (hashed passwords)
    audit_log.jsonl   append-only audit log
    uploads/CASE-XXXX original uploads + aligned face crops (never under static/)
    reports/          generated forensic report PDFs

Training data (face crops) also goes here when training: data/train|val|test/{real,fake}.
Do not commit any of it.
