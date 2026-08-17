"""SQLite index over the generated avatars.

SQLite rather than Postgres because the whole point of this project is that it
starts with one command. The write volume is one row per generated image, and
generation is serialised behind a single GPU, so there is no concurrency
pressure worth a second service.

WAL mode is on so the gallery can read while the worker writes.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS avatars (
    id              TEXT PRIMARY KEY,
    seed            INTEGER NOT NULL,
    seed_key        TEXT,
    attributes      TEXT NOT NULL,
    persona         TEXT,
    prompt          TEXT NOT NULL,
    negative_prompt TEXT NOT NULL,
    model           TEXT NOT NULL,
    backend         TEXT NOT NULL,
    composer        TEXT NOT NULL,
    sizes           TEXT NOT NULL,
    duration_ms     INTEGER,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_avatars_seed_key ON avatars(seed_key);
CREATE INDEX IF NOT EXISTS idx_avatars_created  ON avatars(created_at DESC);

-- Composed prompts are cached so a `by-seed` avatar keeps the exact wording it
-- was first generated with, even if the LLM would phrase it differently today.
-- Stored as the four halves the two text encoders receive, not as one string.
CREATE TABLE IF NOT EXISTS prompt_cache (
    key             TEXT PRIMARY KEY,
    subject         TEXT NOT NULL,
    style           TEXT NOT NULL,
    negative_subject TEXT NOT NULL,
    negative_style  TEXT NOT NULL,
    persona         TEXT,
    source          TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

-- Used by batch generation to avoid producing the same combination twice.
CREATE TABLE IF NOT EXISTS attr_combos (
    combo     TEXT PRIMARY KEY,
    count     INTEGER NOT NULL DEFAULT 1,
    last_seen TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._migrate()
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def _migrate(self) -> None:
        """Bring an older database up to the current schema.

        ``prompt_cache`` is a pure cache, so a schema change is handled by
        dropping it rather than by writing a migration - the worst case is one
        extra LLM call per avatar. Anything holding real data would get a
        proper migration instead.
        """
        existing = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='prompt_cache'"
        ).fetchone()
        if not existing:
            return
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(prompt_cache)")}
        if "subject" not in columns:
            self._conn.execute("DROP TABLE prompt_cache")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- avatars --------------------------------------------------------
    def upsert_avatar(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO avatars (id, seed, seed_key, attributes, persona, prompt,
                                     negative_prompt, model, backend, composer, sizes,
                                     duration_ms, created_at)
                VALUES (:id, :seed, :seed_key, :attributes, :persona, :prompt,
                        :negative_prompt, :model, :backend, :composer, :sizes,
                        :duration_ms, :created_at)
                ON CONFLICT(id) DO UPDATE SET
                    seed_key   = COALESCE(excluded.seed_key, avatars.seed_key),
                    created_at = avatars.created_at
                """,
                {
                    "id": record["id"],
                    "seed": record["seed"],
                    "seed_key": record.get("seed_key"),
                    "attributes": json.dumps(record["attributes"], ensure_ascii=False),
                    "persona": json.dumps(record["persona"], ensure_ascii=False)
                    if record.get("persona")
                    else None,
                    "prompt": record["prompt"],
                    "negative_prompt": record["negative_prompt"],
                    "model": record["model"],
                    "backend": record["backend"],
                    "composer": record["composer"],
                    "sizes": json.dumps(record["sizes"]),
                    "duration_ms": record.get("duration_ms"),
                    "created_at": record.get("created_at") or _now(),
                },
            )
            self._conn.commit()

    def get_avatar(self, avatar_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM avatars WHERE id = ?", (avatar_id,)).fetchone()
        return _row_to_avatar(row) if row else None

    def find_by_seed_key(self, seed_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM avatars WHERE seed_key = ? ORDER BY created_at LIMIT 1",
                (seed_key,),
            ).fetchone()
        return _row_to_avatar(row) if row else None

    def list_avatars(self, limit: int = 60, offset: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM avatars ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [_row_to_avatar(row) for row in rows]

    def count_avatars(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM avatars").fetchone()
        return int(row["n"])

    def delete_avatar(self, avatar_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM avatars WHERE id = ?", (avatar_id,))
            self._conn.commit()
        return cursor.rowcount > 0

    def all_avatar_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT id FROM avatars").fetchall()
        return [row["id"] for row in rows]

    def delete_all_avatars(self) -> int:
        """Empty the index.

        ``attr_combos`` goes too: it exists so batch generation avoids
        repeating a combination, and keeping it after a clear would make the
        next batch skip perfectly good combinations to avoid duplicating
        avatars that no longer exist. The prompt cache is deliberately kept -
        it holds no images, and preserving it means a regenerated `by-seed`
        avatar comes back byte-identical.
        """
        with self._lock:
            cursor = self._conn.execute("DELETE FROM avatars")
            self._conn.execute("DELETE FROM attr_combos")
            self._conn.commit()
        return cursor.rowcount

    # -- prompt cache ---------------------------------------------------
    def get_prompt(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM prompt_cache WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        return {
            "subject": row["subject"],
            "style": row["style"],
            "negative_subject": row["negative_subject"],
            "negative_style": row["negative_style"],
            "persona": json.loads(row["persona"]) if row["persona"] else None,
            "source": row["source"],
        }

    def put_prompt(
        self,
        key: str,
        subject: str,
        style: str,
        negative_subject: str,
        negative_style: str,
        persona: dict[str, Any] | None,
        source: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO prompt_cache (key, subject, style, negative_subject,
                                          negative_style, persona, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO NOTHING
                """,
                (
                    key,
                    subject,
                    style,
                    negative_subject,
                    negative_style,
                    json.dumps(persona, ensure_ascii=False) if persona else None,
                    source,
                    _now(),
                ),
            )
            self._conn.commit()

    # -- attribute combinations ----------------------------------------
    def seen_combo(self, combo: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM attr_combos WHERE combo = ?", (combo,)
            ).fetchone()
        return row is not None

    def record_combo(self, combo: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO attr_combos (combo, count, last_seen) VALUES (?, 1, ?)
                ON CONFLICT(combo) DO UPDATE SET
                    count = attr_combos.count + 1,
                    last_seen = excluded.last_seen
                """,
                (combo, _now()),
            )
            self._conn.commit()


def _row_to_avatar(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "seed": row["seed"],
        "seed_key": row["seed_key"],
        "attributes": json.loads(row["attributes"]),
        "persona": json.loads(row["persona"]) if row["persona"] else None,
        "prompt": row["prompt"],
        "negative_prompt": row["negative_prompt"],
        "model": row["model"],
        "backend": row["backend"],
        "composer": row["composer"],
        "sizes": json.loads(row["sizes"]),
        "duration_ms": row["duration_ms"],
        "created_at": row["created_at"],
    }
