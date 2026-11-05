"""
LegalSaarthi — Authentication & User Role Management
Supports roles: LAWYER, RESEARCHER, STUDENT
Local secure JSON persistence in data/users.json with PBKDF2-HMAC-SHA256 password hashing.
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import os
import pathlib
import secrets
import time
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("legalsaarthi.auth")

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data"
USERS_FILE = DATA_DIR / "users.json"
SESSIONS_FILE = DATA_DIR / "sessions.json"

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class UserRole(str, enum.Enum):
    LAWYER = "LAWYER"
    RESEARCHER = "RESEARCHER"
    STUDENT = "STUDENT"


class UserSignupRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="Password (min 6 characters)")
    full_name: str = Field(..., min_length=2, description="User full name")
    role: UserRole = Field(..., description="Role: LAWYER, RESEARCHER, or STUDENT")
    organization: Optional[str] = Field(None, description="Court / University / Organization name")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        clean = v.strip().lower()
        if not EMAIL_REGEX.match(clean):
            raise ValueError("Invalid email format.")
        return clean


class UserLoginRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="Password")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return v.strip().lower()


class UserProfile(BaseModel):
    user_id: str
    email: str
    full_name: str
    role: UserRole
    organization: Optional[str] = None
    created_at: str


class AuthResponse(BaseModel):
    status: str = "success"
    message: str
    token: str
    user: UserProfile


def _hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
    """Hash password with PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = secrets.token_bytes(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return key.hex(), salt.hex()


def _verify_password(password: str, stored_hash: str, salt_hex: str) -> bool:
    """Verify password against stored salt and hash."""
    salt = bytes.fromhex(salt_hex)
    computed_key, _ = _hash_password(password, salt)
    return secrets.compare_digest(computed_key, stored_hash)


class AuthManager:
    """Manages persistent users and in-memory/persistent tokens."""

    _instance: Optional[AuthManager] = None

    def __init__(self, users_path: pathlib.Path = USERS_FILE, sessions_path: pathlib.Path = SESSIONS_FILE):
        self.users_path = users_path
        self.sessions_path = sessions_path
        self.users_path.parent.mkdir(parents=True, exist_ok=True)
        self.users: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._load_users()
        self._seed_default_users()

    @classmethod
    def get_instance(cls) -> AuthManager:
        if cls._instance is None:
            cls._instance = AuthManager()
        return cls._instance

    def _load_users(self):
        if self.users_path.exists():
            try:
                with open(self.users_path, "r", encoding="utf-8") as f:
                    self.users = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load users from {self.users_path}: {e}")
                self.users = {}

        if self.sessions_path.exists():
            try:
                with open(self.sessions_path, "r", encoding="utf-8") as f:
                    self.sessions = json.load(f)
            except Exception:
                self.sessions = {}

    def _save_users(self):
        try:
            with open(self.users_path, "w", encoding="utf-8") as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist users: {e}")

    def _save_sessions(self):
        try:
            with open(self.sessions_path, "w", encoding="utf-8") as f:
                json.dump(self.sessions, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist sessions: {e}")

    def _seed_default_users(self):
        """Seed pre-configured demo users for instant 1-click evaluation."""
        defaults = [
            {
                "email": "advocate.sharma@legal.in",
                "password": "Password@123",
                "full_name": "Adv. Rajesh Sharma",
                "role": UserRole.LAWYER.value,
                "organization": "Supreme Court Bar Association",
            },
            {
                "email": "dr.verma@research.ac.in",
                "password": "Password@123",
                "full_name": "Dr. Sunita Verma",
                "role": UserRole.RESEARCHER.value,
                "organization": "National Law University, Delhi",
            },
            {
                "email": "aryan.mca@university.edu",
                "password": "Password@123",
                "full_name": "Aryan Patel",
                "role": UserRole.STUDENT.value,
                "organization": "Department of Computer Applications (MCA)",
            },
        ]
        modified = False
        for d in defaults:
            email_key = d["email"].lower()
            if email_key not in self.users:
                pwd_hash, salt_hex = _hash_password(d["password"])
                user_id = f"user_{secrets.token_hex(6)}"
                self.users[email_key] = {
                    "user_id": user_id,
                    "email": d["email"],
                    "full_name": d["full_name"],
                    "role": d["role"],
                    "organization": d["organization"],
                    "password_hash": pwd_hash,
                    "salt": salt_hex,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                modified = True

        if modified:
            self._save_users()
            logger.info("Default demo users initialized for LAWYER, RESEARCHER, and STUDENT.")

    def signup(self, req: UserSignupRequest) -> AuthResponse:
        email_key = req.email.lower()
        if email_key in self.users:
            raise ValueError(f"An account with email '{req.email}' already exists.")

        pwd_hash, salt_hex = _hash_password(req.password)
        user_id = f"user_{secrets.token_hex(6)}"
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        user_record = {
            "user_id": user_id,
            "email": req.email,
            "full_name": req.full_name,
            "role": req.role.value,
            "organization": req.organization or "Independent",
            "password_hash": pwd_hash,
            "salt": salt_hex,
            "created_at": created_at,
        }
        self.users[email_key] = user_record
        self._save_users()

        token = self._create_session(user_record)
        profile = UserProfile(
            user_id=user_id,
            email=req.email,
            full_name=req.full_name,
            role=req.role,
            organization=req.organization,
            created_at=created_at,
        )
        return AuthResponse(
            status="success",
            message=f"Registered successfully as {req.role.value}.",
            token=token,
            user=profile,
        )

    def login(self, req: UserLoginRequest) -> AuthResponse:
        email_key = req.email.lower()
        if email_key not in self.users:
            raise ValueError("Invalid email or password.")

        record = self.users[email_key]
        if not _verify_password(req.password, record["password_hash"], record["salt"]):
            raise ValueError("Invalid email or password.")

        token = self._create_session(record)
        profile = UserProfile(
            user_id=record["user_id"],
            email=record["email"],
            full_name=record["full_name"],
            role=UserRole(record["role"]),
            organization=record.get("organization"),
            created_at=record.get("created_at", ""),
        )
        return AuthResponse(
            status="success",
            message=f"Logged in successfully as {record['role']}.",
            token=token,
            user=profile,
        )

    def _create_session(self, record: Dict[str, Any]) -> str:
        token = f"ls_tok_{secrets.token_hex(24)}"
        self.sessions[token] = {
            "user_id": record["user_id"],
            "email": record["email"],
            "full_name": record["full_name"],
            "role": record["role"],
            "organization": record.get("organization"),
            "created_at": time.time(),
        }
        self._save_sessions()
        return token

    def verify_token(self, token: str) -> Optional[UserProfile]:
        if not token or token not in self.sessions:
            return None
        sess = self.sessions[token]
        # Valid for 30 days
        if time.time() - sess.get("created_at", 0) > 30 * 86400:
            del self.sessions[token]
            self._save_sessions()
            return None
        return UserProfile(
            user_id=sess["user_id"],
            email=sess["email"],
            full_name=sess["full_name"],
            role=UserRole(sess["role"]),
            organization=sess.get("organization"),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(sess.get("created_at", time.time()))),
        )

    def logout(self, token: str) -> bool:
        if token in self.sessions:
            del self.sessions[token]
            self._save_sessions()
            return True
        return False
