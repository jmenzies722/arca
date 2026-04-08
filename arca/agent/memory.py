"""
Session memory — SQLite-backed conversation history.

Stores the full message history so Claude has context across the session.
Each "session" is one terminal run. History is persisted to disk so
you can review past sessions (future: semantic search across sessions).

Schema:
  sessions(id, created_at, project_path)
  messages(id, session_id, role, content_json, created_at)

The messages list returned by load_history() is Claude API-compatible:
  [{"role": "user" | "assistant", "content": "..."}]
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Memory:
    """SQLite-backed conversation memory."""

    def __init__(self, db_path: str = "~/.arca/memory.db", max_history: int = 50):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        self._session_id: int | None = None
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._setup()

    def _setup(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                project_path TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL,
                content_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, id);
        """)
        self._conn.commit()

    def start_session(self, project_path: str | None = None) -> int:
        """Create a new session. Call once at startup."""
        cursor = self._conn.execute(
            "INSERT INTO sessions(created_at, project_path) VALUES (?, ?)",
            (datetime.now(timezone.utc).isoformat(), project_path or os.getcwd()),
        )
        self._conn.commit()
        self._session_id = cursor.lastrowid
        return self._session_id

    def add_message(self, role: str, content: Any) -> None:
        """
        Append a message to the current session.

        content can be:
          - str (simple text)
          - list (Claude API content blocks, e.g. tool use/results)
        """
        if self._session_id is None:
            self.start_session()

        self._conn.execute(
            "INSERT INTO messages(session_id, role, content_json, created_at) VALUES (?, ?, ?, ?)",
            (
                self._session_id,
                role,
                json.dumps(content),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def load_history(self) -> list[dict]:
        """
        Return the last max_history messages as Claude API message list.
        Trims to keep within token budget while preserving conversation coherence.
        """
        if self._session_id is None:
            return []

        rows = self._conn.execute(
            """
            SELECT role, content_json FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (self._session_id, self.max_history),
        ).fetchall()

        # Reverse to chronological order
        messages = [
            {"role": row[0], "content": json.loads(row[1])}
            for row in reversed(rows)
        ]

        return messages

    def clear_session(self) -> None:
        """Clear all messages in the current session (start fresh)."""
        if self._session_id is None:
            return
        self._conn.execute(
            "DELETE FROM messages WHERE session_id = ?",
            (self._session_id,),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
