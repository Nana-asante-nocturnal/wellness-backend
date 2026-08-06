"""
Session history persistence using SQLite (aiosqlite).

Stores exam reports and wellness session summaries for the History tab.
One SQLite file — no server setup, no external database process.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.db")


def _get_conn() -> sqlite3.Connection:
    """Get a synchronous SQLite connection (thread-safe per request)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Call once at startup."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            duration_sec REAL,
            summary_json TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_session(session_type: str, duration_sec: float | None,
                 summary: dict[str, Any]) -> int:
    """
    Persist a session summary to the history database.

    Args:
        session_type: "wellness" or "neuro_exam".
        duration_sec: Total session duration in seconds.
        summary: Full session summary dict (serializable to JSON).

    Returns:
        int: Row ID of the inserted record.
    """
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO sessions (session_type, created_at, duration_sec, summary_json) VALUES (?, ?, ?, ?)",
        (session_type, datetime.now(timezone.utc).isoformat(), duration_sec,
         json.dumps(summary, default=str)),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def list_sessions(limit: int = 50, session_type: str | None = None
                  ) -> list[dict]:
    """
    List recent sessions from history.

    Args:
        limit: Maximum number of records to return.
        session_type: Optional filter by type ("wellness" or "neuro_exam").

    Returns:
        list of dicts with id, session_type, created_at, duration_sec, summary.
    """
    conn = _get_conn()
    if session_type:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE session_type = ? ORDER BY created_at DESC LIMIT ?",
            (session_type, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        try:
            d["summary"] = json.loads(d.pop("summary_json"))
        except (json.JSONDecodeError, KeyError):
            d["summary"] = {}
        results.append(d)
    conn.close()
    return results


def get_session(session_id: int) -> dict | None:
    """
    Retrieve a single session by ID.

    Args:
        session_id: Row ID of the session.

    Returns:
        dict or None if not found.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    try:
        d["summary"] = json.loads(d.pop("summary_json"))
    except (json.JSONDecodeError, KeyError):
        d["summary"] = {}
    return d


def delete_session(session_id: int) -> bool:
    """
    Delete a session record by ID.

    Args:
        session_id: Row ID of the session to delete.

    Returns:
        bool: True if a record was deleted.
    """
    conn = _get_conn()
    cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def clear_history(session_type: str | None = None) -> int:
    """
    Clear all history records, optionally filtered by type.

    Args:
        session_type: If provided, only delete records of this type.

    Returns:
        int: Number of records deleted.
    """
    conn = _get_conn()
    if session_type:
        cur = conn.execute(
            "DELETE FROM sessions WHERE session_type = ?", (session_type,)
        )
    else:
        cur = conn.execute("DELETE FROM sessions")
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    return deleted
