"""SQLite telemetry for router sessions and chat events.

Disabled by default. Set ROUTER_TELEMETRY_DB to a file path to enable.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DB_PATH: Path | None = (
    Path(os.environ["ROUTER_TELEMETRY_DB"])
    if os.environ.get("ROUTER_TELEMETRY_DB")
    else None
)
_conn: sqlite3.Connection | None = None


def enabled() -> bool:
    """Return True if telemetry is configured."""
    return _DB_PATH is not None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _DB_PATH is None:
        raise RuntimeError(
            "Telemetry not enabled. Set ROUTER_TELEMETRY_DB env var to a file path."
        )
    if _conn is None:
        _conn = sqlite3.connect(str(_DB_PATH))
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL
            )
        """)
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                question TEXT NOT NULL,
                selected_model TEXT NOT NULL,
                latency_ms REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        _conn.commit()
    return _conn


def create_session() -> int:
    """Create a new session and return its ID."""
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO sessions (created_at) VALUES (?)",
        (datetime.now(timezone.utc).isoformat(),),
    )
    conn.commit()
    return cur.lastrowid


def log_chat(
    session_id: int | None,
    question: str,
    selected_model: str,
    latency_ms: float | None = None,
) -> int:
    """Log a chat event and return its ID."""
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO chat_events (session_id, question, selected_model, latency_ms, created_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, question, selected_model, latency_ms, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return cur.lastrowid


def get_stats() -> dict[str, Any]:
    """Return aggregate stats from chat_events."""
    conn = _get_conn()
    cur = conn.execute("""
        SELECT
            COUNT(*) as total_events,
            COUNT(DISTINCT session_id) as total_sessions,
            AVG(latency_ms) as avg_latency_ms
        FROM chat_events
    """)
    row = cur.fetchone()
    cur2 = conn.execute("""
        SELECT selected_model, COUNT(*) as cnt
        FROM chat_events
        GROUP BY selected_model
        ORDER BY cnt DESC
    """)
    by_model = {r[0]: r[1] for r in cur2.fetchall()}
    return {
        "total_events": row[0] or 0,
        "total_sessions": row[1] or 0,
        "avg_latency_ms": round(row[2], 2) if row[2] else None,
        "by_model": by_model,
    }
