from __future__ import annotations

import sqlite3
from pathlib import Path

from discord_ai_assistant.ai.memory import is_disallowed_memory
from discord_ai_assistant.models import Track, UserActivity, UserMemory

MAX_MEMORY_CONTEXT_CHARACTERS = 1_200


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self._migrate()

    def close(self) -> None:
        self.connection.close()

    def _migrate(self) -> None:
        self.connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS library_tracks (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                original_name TEXT NOT NULL,
                stored_name TEXT NOT NULL UNIQUE,
                uploaded_by INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS playlists (
                id INTEGER PRIMARY KEY,
                guild_id INTEGER NOT NULL,
                name TEXT NOT NULL COLLATE NOCASE,
                owner_id INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(guild_id, name)
            );
            CREATE TABLE IF NOT EXISTS playlist_items (
                playlist_id INTEGER NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
                track_id INTEGER NOT NULL REFERENCES library_tracks(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                PRIMARY KEY (playlist_id, position)
            );
            CREATE TABLE IF NOT EXISTS bot_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS user_memories (
                id INTEGER PRIMARY KEY,
                guild_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_user_memories_owner
                ON user_memories(guild_id, user_id, id DESC);
            CREATE TABLE IF NOT EXISTS user_activity (
                guild_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0,
                ai_request_count INTEGER NOT NULL DEFAULT 0,
                first_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(guild_id, user_id)
            );
            """
        )
        self.connection.commit()

    def add_track(self, title: str, original_name: str, stored_name: str, uploaded_by: int) -> Track:
        cursor = self.connection.execute(
            "INSERT INTO library_tracks(title, original_name, stored_name, uploaded_by) VALUES (?, ?, ?, ?)",
            (title, original_name, stored_name, uploaded_by),
        )
        self.connection.commit()
        return Track(cursor.lastrowid, title, original_name, stored_name, uploaded_by)

    def get_track(self, track_id: int) -> Track | None:
        row = self.connection.execute("SELECT * FROM library_tracks WHERE id = ?", (track_id,)).fetchone()
        return self._track(row) if row else None

    def search_tracks(self, query: str, limit: int = 10) -> list[Track]:
        rows = self.connection.execute(
            "SELECT * FROM library_tracks WHERE title LIKE ? OR original_name LIKE ? ORDER BY title LIMIT ?",
            (f"%{query}%", f"%{query}%", limit),
        ).fetchall()
        return [self._track(row) for row in rows]

    def random_track(self, exclude_track_id: int | None = None) -> Track | None:
        if exclude_track_id is None:
            row = self.connection.execute("SELECT * FROM library_tracks ORDER BY RANDOM() LIMIT 1").fetchone()
        else:
            row = self.connection.execute(
                "SELECT * FROM library_tracks WHERE id != ? ORDER BY RANDOM() LIMIT 1", (exclude_track_id,)
            ).fetchone()
        return self._track(row) if row else None

    def list_tracks(self, order: str, page: int, per_page: int = 20) -> tuple[list[Track], int]:
        ordering = {
            "title": "title COLLATE NOCASE ASC, id ASC",
            "newest": "id DESC",
            "oldest": "id ASC",
            "id": "id ASC",
        }.get(order)
        if ordering is None:
            raise ValueError("不支援的排序方式。")
        if page < 1 or not 1 <= per_page <= 20:
            raise ValueError("頁碼或每頁數量不正確。")
        total = int(self.connection.execute("SELECT COUNT(*) FROM library_tracks").fetchone()[0])
        rows = self.connection.execute(
            f"SELECT * FROM library_tracks ORDER BY {ordering} LIMIT ? OFFSET ?",
            (per_page, (page - 1) * per_page),
        ).fetchall()
        return [self._track(row) for row in rows], total

    def delete_track(self, track_id: int) -> Track | None:
        track = self.get_track(track_id)
        if track:
            self.connection.execute("DELETE FROM library_tracks WHERE id = ?", (track_id,))
            self.connection.commit()
        return track

    def create_playlist(self, guild_id: int, name: str, owner_id: int) -> None:
        try:
            self.connection.execute(
                "INSERT INTO playlists(guild_id, name, owner_id) VALUES (?, ?, ?)",
                (guild_id, name, owner_id),
            )
            self.connection.commit()
        except sqlite3.IntegrityError as error:
            raise ValueError("此伺服器已有同名播放清單。") from error

    def create_playlist_with_tracks(
        self, guild_id: int, name: str, owner_id: int, track_ids: list[int]
    ) -> None:
        try:
            with self.connection:
                cursor = self.connection.execute(
                    "INSERT INTO playlists(guild_id, name, owner_id) VALUES (?, ?, ?)",
                    (guild_id, name, owner_id),
                )
                playlist_id = int(cursor.lastrowid)
                for position, track_id in enumerate(track_ids, start=1):
                    if not self.get_track(track_id):
                        raise ValueError(f"找不到歌曲 ID：{track_id}。")
                    self.connection.execute(
                        "INSERT INTO playlist_items(playlist_id, track_id, position) VALUES (?, ?, ?)",
                        (playlist_id, track_id, position),
                    )
        except sqlite3.IntegrityError as error:
            raise ValueError("此伺服器已有同名播放清單。") from error

    def list_playlists(self, guild_id: int) -> list[str]:
        rows = self.connection.execute(
            "SELECT name FROM playlists WHERE guild_id = ? ORDER BY name", (guild_id,)
        ).fetchall()
        return [str(row["name"]) for row in rows]

    def add_to_playlist(self, guild_id: int, name: str, track_id: int) -> None:
        playlist = self._playlist_id(guild_id, name)
        position = self.connection.execute(
            "SELECT COALESCE(MAX(position), 0) + 1 FROM playlist_items WHERE playlist_id = ?", (playlist,)
        ).fetchone()[0]
        self.connection.execute(
            "INSERT INTO playlist_items(playlist_id, track_id, position) VALUES (?, ?, ?)",
            (playlist, track_id, position),
        )
        self.connection.commit()

    def playlist_tracks(self, guild_id: int, name: str) -> list[Track]:
        playlist = self._playlist_id(guild_id, name)
        rows = self.connection.execute(
            """SELECT t.* FROM playlist_items i JOIN library_tracks t ON t.id = i.track_id
               WHERE i.playlist_id = ? ORDER BY i.position""",
            (playlist,),
        ).fetchall()
        return [self._track(row) for row in rows]

    def get_state(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM bot_state WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def set_state(self, key: str, value: str) -> None:
        self.connection.execute(
            """INSERT INTO bot_state(key, value) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP""",
            (key, value),
        )
        self.connection.commit()

    def add_user_memory(self, guild_id: int, user_id: int, category: str, content: str) -> UserMemory:
        normalized_category = " ".join(category.split())
        normalized_content = " ".join(content.split())
        if not normalized_category or len(normalized_category) > 40:
            raise ValueError("記憶分類需介於 1 到 40 個字元。")
        if not normalized_content or len(normalized_content) > 300:
            raise ValueError("記憶內容需介於 1 到 300 個字元。")
        if is_disallowed_memory(normalized_category, normalized_content):
            raise ValueError("這項內容像是敏感資料或修改墨雪規則的指令，因此不會存入記憶。")
        cursor = self.connection.execute(
            "INSERT INTO user_memories(guild_id, user_id, category, content) VALUES (?, ?, ?, ?)",
            (guild_id, user_id, normalized_category, normalized_content),
        )
        self.connection.commit()
        row = self.connection.execute("SELECT * FROM user_memories WHERE id = ?", (cursor.lastrowid,)).fetchone()
        assert row is not None
        return self._user_memory(row)

    def add_user_memory_if_new(self, guild_id: int, user_id: int, category: str, content: str) -> UserMemory | None:
        normalized_content = " ".join(content.split())
        existing = self.connection.execute(
            "SELECT * FROM user_memories WHERE guild_id = ? AND user_id = ? AND content = ?",
            (guild_id, user_id, normalized_content),
        ).fetchone()
        return None if existing else self.add_user_memory(guild_id, user_id, category, normalized_content)

    def list_user_memories(self, guild_id: int, user_id: int, limit: int = 20) -> list[UserMemory]:
        rows = self.connection.execute(
            "SELECT * FROM user_memories WHERE guild_id = ? AND user_id = ? ORDER BY id DESC LIMIT ?",
            (guild_id, user_id, limit),
        ).fetchall()
        return [self._user_memory(row) for row in rows]

    def delete_user_memory(self, guild_id: int, user_id: int, memory_id: int) -> bool:
        cursor = self.connection.execute(
            "DELETE FROM user_memories WHERE id = ? AND guild_id = ? AND user_id = ?",
            (memory_id, guild_id, user_id),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def clear_user_memories(self, guild_id: int, user_id: int) -> int:
        cursor = self.connection.execute(
            "DELETE FROM user_memories WHERE guild_id = ? AND user_id = ?", (guild_id, user_id)
        )
        self.connection.commit()
        return cursor.rowcount

    def record_user_message_activity(self, guild_id: int, user_id: int) -> None:
        self.connection.execute(
            """INSERT INTO user_activity(guild_id, user_id, message_count) VALUES (?, ?, 1)
               ON CONFLICT(guild_id, user_id) DO UPDATE SET
                 message_count = message_count + 1,
                 last_seen = CURRENT_TIMESTAMP""",
            (guild_id, user_id),
        )
        self.connection.commit()

    def record_user_ai_request(self, guild_id: int, user_id: int) -> None:
        self.connection.execute(
            """INSERT INTO user_activity(guild_id, user_id, ai_request_count) VALUES (?, ?, 1)
               ON CONFLICT(guild_id, user_id) DO UPDATE SET
                 ai_request_count = ai_request_count + 1,
                 last_seen = CURRENT_TIMESTAMP""",
            (guild_id, user_id),
        )
        self.connection.commit()

    def user_activity(self, guild_id: int, user_id: int) -> UserActivity | None:
        row = self.connection.execute(
            "SELECT * FROM user_activity WHERE guild_id = ? AND user_id = ?", (guild_id, user_id)
        ).fetchone()
        return self._user_activity(row) if row else None

    def user_memory_context(self, guild_id: int, user_id: int) -> str:
        memories = self.list_user_memories(guild_id, user_id, limit=12)
        activity = self.user_activity(guild_id, user_id)
        lines: list[str] = []
        remaining = MAX_MEMORY_CONTEXT_CHARACTERS
        for memory in reversed(memories):
            if is_disallowed_memory(memory.category, memory.content):
                continue
            line = f"- [{memory.category}] {memory.content}"
            if len(line) > remaining:
                break
            lines.append(line)
            remaining -= len(line) + 1
        if activity and (activity.message_count or activity.ai_request_count):
            lines.append(
                f"- [互動紀錄] 已在此伺服器與墨雪互動；訊息 {activity.message_count} 則，AI 提問 {activity.ai_request_count} 次。"
            )
        return "\n".join(lines)

    def _playlist_id(self, guild_id: int, name: str) -> int:
        row = self.connection.execute(
            "SELECT id FROM playlists WHERE guild_id = ? AND name = ?", (guild_id, name)
        ).fetchone()
        if not row:
            raise ValueError("找不到播放清單。")
        return int(row["id"])

    @staticmethod
    def _track(row: sqlite3.Row) -> Track:
        return Track(
            id=int(row["id"]),
            title=str(row["title"]),
            original_name=str(row["original_name"]),
            stored_name=str(row["stored_name"]),
            uploaded_by=int(row["uploaded_by"]),
        )

    @staticmethod
    def _user_memory(row: sqlite3.Row) -> UserMemory:
        return UserMemory(
            id=int(row["id"]),
            category=str(row["category"]),
            content=str(row["content"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _user_activity(row: sqlite3.Row) -> UserActivity:
        return UserActivity(
            message_count=int(row["message_count"]),
            ai_request_count=int(row["ai_request_count"]),
            first_seen=str(row["first_seen"]),
            last_seen=str(row["last_seen"]),
        )
