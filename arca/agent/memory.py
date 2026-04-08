"""
Session memory — SQLite conversation history + chromadb semantic search.

Phase 1: SQLite rolling window (last N messages sent to Claude)
Phase 2: chromadb vector store for semantic retrieval across sessions

The two layers complement each other:
  SQLite: Exact recent history → short-term context (what we just said)
  chromadb: Semantic search → long-term context (what we said about this topic weeks ago)

On each Claude call, memory contributes two things:
  1. load_history() → last N messages (sliding window, chronological)
  2. semantic_search(query) → top-K similar past exchanges (any session)

Both are injected into the Claude API messages list:
  - Semantic hits go first as a "here's relevant past context" system addition
  - Recent history follows as the live conversation

Schema (SQLite):
  sessions(id, created_at, project_path)
  messages(id, session_id, role, content_json, created_at, text_preview)
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Memory:
    """SQLite conversation history + optional chromadb semantic search."""

    def __init__(
        self,
        db_path: str = "~/.arca/memory.db",
        max_history: int = 50,
        semantic_search: bool = True,
        semantic_db_path: str = "~/.arca/vectors",
        semantic_top_k: int = 5,
    ):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        self.semantic_enabled = semantic_search
        self.semantic_top_k = semantic_top_k

        self._session_id: int | None = None
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._chroma = None
        self._chroma_collection = None

        self._setup_sqlite()

        if semantic_search:
            self._setup_chromadb(Path(semantic_db_path).expanduser())

    def _setup_sqlite(self) -> None:
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
                text_preview TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, id);
        """)
        self._conn.commit()

    def _setup_chromadb(self, db_path: Path) -> None:
        """Initialize chromadb persistent client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings

            db_path.mkdir(parents=True, exist_ok=True)
            self._chroma = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(anonymized_telemetry=False),
            )
            self._chroma_collection = self._chroma.get_or_create_collection(
                name="arca_messages",
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError:
            print("[arca/memory] chromadb not installed — semantic search disabled.")
            print("[arca/memory] Install: pip install chromadb")
            self.semantic_enabled = False
        except Exception as e:
            print(f"[arca/memory] chromadb init failed: {e} — semantic search disabled.")
            self.semantic_enabled = False

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
        Also indexes in chromadb for semantic search.
        """
        if self._session_id is None:
            self.start_session()

        content_json = json.dumps(content)

        # Extract plain text preview for semantic indexing
        if isinstance(content, str):
            text_preview = content[:500]
        elif isinstance(content, list):
            texts = [
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            text_preview = " ".join(texts)[:500]
        else:
            text_preview = str(content)[:500]

        cursor = self._conn.execute(
            "INSERT INTO messages(session_id, role, content_json, text_preview, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                self._session_id,
                role,
                content_json,
                text_preview,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        msg_id = cursor.lastrowid

        # Index in chromadb (skip tool results — too noisy)
        if self.semantic_enabled and self._chroma_collection and text_preview and role != "tool":
            try:
                self._chroma_collection.add(
                    ids=[f"msg_{msg_id}"],
                    documents=[text_preview],
                    metadatas=[{
                        "session_id": self._session_id,
                        "role": role,
                        "msg_id": msg_id,
                    }],
                )
            except Exception:
                pass  # Indexing failures are non-fatal

    def load_history(self) -> list[dict]:
        """Return the last max_history messages as Claude API message list."""
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

        return [
            {"role": row[0], "content": json.loads(row[1])}
            for row in reversed(rows)
        ]

    def semantic_search(self, query: str, exclude_current_session: bool = True) -> list[dict]:
        """
        Find semantically similar past messages across all sessions.

        Returns up to semantic_top_k messages as Claude API message dicts,
        to be injected as context before the current conversation.

        Args:
            query: The user's current message (used as the search query).
            exclude_current_session: Skip results from the current session
                                     (already in load_history()).
        """
        if not self.semantic_enabled or not self._chroma_collection:
            return []

        try:
            where = None
            if exclude_current_session and self._session_id:
                where = {"session_id": {"$ne": self._session_id}}

            results = self._chroma_collection.query(
                query_texts=[query],
                n_results=self.semantic_top_k,
                where=where,
            )

            if not results or not results["documents"]:
                return []

            docs = results["documents"][0]
            metas = results["metadatas"][0]

            messages = []
            for doc, meta in zip(docs, metas):
                role = meta.get("role", "user")
                messages.append({"role": role, "content": f"[Past context] {doc}"})

            return messages
        except Exception:
            return []

    def clear_session(self) -> None:
        """Clear all messages in the current session."""
        if self._session_id is None:
            return
        self._conn.execute(
            "DELETE FROM messages WHERE session_id = ?",
            (self._session_id,),
        )
        self._conn.commit()

    def clear_all_sessions(self) -> None:
        """Wipe all conversation history across all sessions (used by --reset flag)."""
        self._conn.executescript("DELETE FROM messages; DELETE FROM sessions;")
        self._conn.commit()
        if self._chroma_collection:
            try:
                self._chroma_collection.delete(where={"msg_id": {"$gte": 0}})
            except Exception:
                pass
        self._session_id = None

    def close(self) -> None:
        self._conn.close()
