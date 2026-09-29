"""Create the demo accounts (Section 3.4 roles).

    python seed_users.py [--password PASSWORD]
    flask --app app seed-users

Demo accounts: INV-01 (investigator), ANA-01 and ANA-02 (analysts), ADM-01 (admin).
Change the passwords before any real use.
"""
import argparse

import config
from users import UserStore

DEMO_USERS = [
    ("INV-01", "Demo Investigator", "investigator"),
    ("ANA-01", "Demo Analyst", "analyst"),
    ("ANA-02", "Second Analyst", "analyst"),
    ("ADM-01", "System Administrator", "admin"),
]
DEFAULT_PASSWORD = "demo1234"


def seed(store=None, password=None):
    store = store or UserStore(config.USERS_PATH)
    password = password or DEFAULT_PASSWORD
    existing = {u["user_id"] for u in store.all()}
    for uid, name, role in DEMO_USERS:
        if uid in existing:
            print(f"  {uid:<7} already exists, skipped")
            continue
        store.create(uid, name, role, password)
        print(f"  {uid:<7} {role:<13} created")
    print(f"Demo password: {password}  (change before real use)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--password", default=None)
    seed(password=ap.parse_args().password)
