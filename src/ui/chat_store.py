"""
chat_store.py — persistent storage for GPT4YOUTH chat conversations.

WHY THIS EXISTS
---------------
Streamlit reruns the whole script on every interaction, and `st.session_state`
is per-browser-session and lost on refresh. To let a logged-in user find, load,
continue and delete *past* conversations, we need on-disk persistence.

DESIGN
------
- SQLite: single file, zero extra services, ACID. Enough for lab / early prod.
  (Swap to Postgres later WITHOUT touching the UI: keep these function
  signatures, replace the connection + SQL with psycopg. The UI only calls the
  functions below, never SQL.)
- Short-lived connection PER operation with check_same_thread=False. Streamlit
  serves each session on its own thread; a per-op connection is the simplest
  thread-safe pattern and the overhead is negligible at this scale.
- WAL journal mode -> concurrent reads don't block; foreign keys ON -> cascade
  delete of a conversation removes its messages.

SCHEMA
------
conversations(id, user_id, title, created_at, updated_at)
messages(id, conversation_id -> conversations.id CASCADE, seq, role, content, created_at)
  - `seq` is a 0-based per-conversation ordering of NON-system messages.
  - The system prompt is NOT stored (it lives in config.yaml and may change);
    it is re-prepended from the CURRENT config on load, so prompt improvements
    apply retroactively to old chats.
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from contextlib import contextmanager

# DB path: overridable via env (e.g. point it at a mounted volume in Docker).
# Default: <project_root>/data/gpt4youth_chats.db  (gitignore the data/ dir).
DB_PATH = os.environ.get(
    "GPT4YOUTH_DB_PATH",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "gpt4youth_chats.db"
    ),
)


def _now() -> float:
    return time.time()


@contextmanager
def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create tables/indexes if missing. Call once at startup (idempotent)."""
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                title       TEXT NOT NULL DEFAULT 'New chat',
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_conv_user
                ON conversations(user_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS messages (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id  TEXT NOT NULL,
                seq              INTEGER NOT NULL,
                role             TEXT NOT NULL,
                content          TEXT NOT NULL,
                created_at       REAL NOT NULL,
                UNIQUE(conversation_id, seq),
                FOREIGN KEY(conversation_id)
                    REFERENCES conversations(id) ON DELETE CASCADE
            );
            """
        )


# --- conversations -----------------------------------------------------------

def create_conversation(user_id: str, title: str) -> str:
    """Create a conversation, return its id (uuid4 hex)."""
    conv_id = uuid.uuid4().hex
    ts = _now()
    title = (title or "New chat").strip().replace("\n", " ")[:80] or "New chat"
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversations (id, user_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conv_id, user_id, title, ts, ts),
        )
    return conv_id


def list_conversations(user_id: str) -> list[dict]:
    """Return this user's conversations, most-recently-updated first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, updated_at FROM conversations "
            "WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def rename_conversation(conv_id: str, title: str) -> None:
    title = (title or "").strip().replace("\n", " ")[:80] or "New chat"
    with _connect() as conn:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), conv_id),
        )


def delete_conversation(conv_id: str) -> None:
    """Delete a conversation and (via CASCADE) all its messages."""
    with _connect() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))


# --- messages ----------------------------------------------------------------

def get_messages(conv_id: str) -> list[dict]:
    """Return non-system messages [{role, content}, ...] in order."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages "
            "WHERE conversation_id = ? ORDER BY seq ASC",
            (conv_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def _next_seq(conn, conv_id: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(seq), -1) AS m FROM messages WHERE conversation_id = ?",
        (conv_id,),
    ).fetchone()
    return int(row["m"]) + 1


def append_message(conv_id: str, role: str, content: str) -> int:
    """Append a message at the next seq; bump the conversation's updated_at."""
    ts = _now()
    with _connect() as conn:
        seq = _next_seq(conn, conv_id)
        conn.execute(
            "INSERT INTO messages (conversation_id, seq, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conv_id, seq, role, content, ts),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (ts, conv_id)
        )
    return seq


def truncate_from(conv_id: str, seq: int) -> None:
    """Delete messages with seq >= `seq`. Used when a user edits & regenerates."""
    with _connect() as conn:
        conn.execute(
            "DELETE FROM messages WHERE conversation_id = ? AND seq >= ?",
            (conv_id, seq),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), conv_id)
        )
