"""User accounts (Section 3.4 roles) stored in data/users.json.

Passwords are stored only as werkzeug salted hashes (scrypt/pbkdf2).
Roles: investigator (submits cases), analyst (reviews cases), admin (manages users).
"""
import json
import os
import tempfile
import threading

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

INVESTIGATOR, ANALYST, ADMIN = "investigator", "analyst", "admin"
ROLES = (INVESTIGATOR, ANALYST, ADMIN)


class User(UserMixin):
    def __init__(self, record):
        self.record = record
        self.id = record["user_id"]
        self.name = record["name"]
        self.role = record["role"]
        self.active = record.get("active", True)

    @property
    def is_active(self):
        return self.active

    def has_role(self, *roles):
        return self.role in roles


class UserStore:
    def __init__(self, path):
        self.path = path
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if not os.path.exists(path):
            self._save({"users": []})

    def _load(self):
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data):
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(self.path)), suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)

    def all(self):
        with self._lock:
            return self._load()["users"]

    def get(self, user_id):
        for u in self.all():
            if u["user_id"] == user_id:
                return User(u)
        return None

    def names(self):
        return {u["user_id"]: {"name": u["name"], "role": u["role"]} for u in self.all()}

    def create(self, user_id, name, role, password):
        user_id = (user_id or "").strip().upper()
        if not user_id or not name:
            raise ValueError("User ID and name are required.")
        if role not in ROLES:
            raise ValueError(f"Role must be one of {ROLES}.")
        if len(password or "") < 8:
            raise ValueError("Password must be at least 8 characters.")
        with self._lock:
            data = self._load()
            if any(u["user_id"] == user_id for u in data["users"]):
                raise ValueError(f"User {user_id} already exists.")
            data["users"].append({"user_id": user_id, "name": name.strip(), "role": role,
                                  "password_hash": generate_password_hash(password),
                                  "active": True})
            self._save(data)
        return self.get(user_id)

    def set_active(self, user_id, active):
        with self._lock:
            data = self._load()
            for u in data["users"]:
                if u["user_id"] == user_id:
                    u["active"] = bool(active)
                    self._save(data)
                    return
        raise KeyError(user_id)

    def authenticate(self, user_id, password):
        user_id = (user_id or "").strip().upper()
        for u in self.all():
            if u["user_id"] == user_id and u.get("active", True):
                if check_password_hash(u["password_hash"], password or ""):
                    return User(u)
        return None
