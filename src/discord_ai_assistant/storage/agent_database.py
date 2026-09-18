from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Iterable

from discord_ai_assistant.agent.events import AgentEvent, AgentEventBus, AgentEventKind
from discord_ai_assistant.ai.memory import is_disallowed_memory
from discord_ai_assistant.ai.memory_v2 import (
    MEMORY_KIND_SEMANTIC,
    MemoryMatch,
    build_fts_query,
    infer_memory_key,
    infer_memory_kind,
    infer_project_name,
    memory_search_document,
    normalize_memory_text,
    score_memory_candidate,
)
from discord_ai_assistant.models import UserMemory
from discord_ai_assistant.storage.database import Database, MAX_MEMORY_CONTEXT_CHARACTERS

LOGGER = logging.getLogger(__name__)
MEMORY_V2_CONTEXT_CHARACTERS = max(MAX_MEMORY_CONTEXT_CHARACTERS, 2_200)
MEMORY_V2_MAX_CANDIDATES = 256
MEMORY_V2_FTS_CANDIDATES = 64


class AgentDatabase(Database):
    """SQLite store with query-aware, consolidating Memory V2 metadata."""

    def __init__(self, path, event_bus: AgentEventBus | None = None) -> None:
        self.event_bus = event_bus
        self._memory_fts_enabled = False
        super().__init__(path)
        self._ensure_agent_memory_columns()
        self._ensure_memory_v2_indexes()
        self._backfill_memory_v2_metadata()
        self._ensure_memory_fts()

    def _ensure_agent_memory_columns(self) -> None:
        existing = {
            str(row[1])
            for row in self.connection.execute("PRAGMA table_info(user_memories)").fetchall()
        }
        additions = {
            "importance": "INTEGER NOT NULL DEFAULT 1",
            "last_used": "TEXT",
            "use_count": "INTEGER NOT NULL DEFAULT 0",
            "source": "TEXT NOT NULL DEFAULT 'manual'",
            "memory_kind": "TEXT NOT NULL DEFAULT 'semantic'",
            "subject": "TEXT",
            "project": "TEXT",
            "memory_key": "TEXT",
            "confidence": "REAL NOT NULL DEFAULT 1.0",
            "status": "TEXT NOT NULL DEFAULT 'active'",
            "superseded_by": "INTEGER",
            "last_confirmed": "TEXT",
        }
        for name, definition in additions.items():
            if name not in existing:
                self.connection.execute(f"ALTER TABLE user_memories ADD COLUMN {name} {definition}")
        self.connection.commit()

    def _ensure_memory_v2_indexes(self) -> None:
        self.connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_user_memories_active
                ON user_memories(guild_id, user_id, status, importance DESC, id DESC);
            CREATE INDEX IF NOT EXISTS idx_user_memories_slot
                ON user_memories(guild_id, user_id, memory_key, status);
            CREATE INDEX IF NOT EXISTS idx_user_memories_project
                ON user_memories(guild_id, user_id, project, status);
            """
        )
        self.connection.commit()

    def _backfill_memory_v2_metadata(self) -> None:
        rows = self.connection.execute(
            """SELECT id, category, content, memory_kind, subject, project, memory_key, last_confirmed
               FROM user_memories"""
        ).fetchall()
        with self.connection:
            for row in rows:
                category = str(row["category"])
                content = str(row["content"])
                inferred_kind = infer_memory_kind(category, content)
                inferred_key, inferred_subject = infer_memory_key(category, content)
                inferred_project = infer_project_name(category, content)
                current_kind = str(row["memory_kind"] or MEMORY_KIND_SEMANTIC)
                kind = inferred_kind if current_kind == MEMORY_KIND_SEMANTIC else current_kind
                self.connection.execute(
                    """UPDATE user_memories
                       SET memory_kind = ?,
                           subject = COALESCE(subject, ?),
                           project = COALESCE(project, ?),
                           memory_key = COALESCE(memory_key, ?),
                           last_confirmed = COALESCE(last_confirmed, updated_at, created_at)
                       WHERE id = ?""",
                    (kind, inferred_subject, inferred_project, inferred_key, int(row["id"])),
                )

    def _ensure_memory_fts(self) -> None:
        try:
            self.connection.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS user_memories_fts USING fts5(
                       memory_id UNINDEXED,
                       guild_id UNINDEXED,
                       user_id UNINDEXED,
                       search_text,
                       tokenize='unicode61 remove_diacritics 2'
                   )"""
            )
        except sqlite3.OperationalError as error:
            LOGGER.warning("SQLite FTS5 unavailable; Memory V2 will use lexical ranking only: %s", error)
            self._memory_fts_enabled = False
            return

        self._memory_fts_enabled = True
        with self.connection:
            self.connection.execute("DELETE FROM user_memories_fts")
            rows = self.connection.execute(
                """SELECT id, guild_id, user_id, category, content, subject, project
                   FROM user_memories WHERE status = 'active'"""
            ).fetchall()
            for row in rows:
                self._insert_fts_row(row)

    def _insert_fts_row(self, row: sqlite3.Row) -> None:
        if not self._memory_fts_enabled:
            return
        self.connection.execute(
            """INSERT INTO user_memories_fts(memory_id, guild_id, user_id, search_text)
               VALUES (?, ?, ?, ?)""",
            (
                int(row["id"]),
                int(row["guild_id"]),
                int(row["user_id"]),
                memory_search_document(
                    str(row["category"]),
                    str(row["content"]),
                    str(row["subject"]) if row["subject"] else None,
                    str(row["project"]) if row["project"] else None,
                ),
            ),
        )

    def _sync_fts_ids(self, memory_ids: Iterable[int]) -> None:
        if not self._memory_fts_enabled:
            return
        ids = tuple(dict.fromkeys(int(memory_id) for memory_id in memory_ids))
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self.connection:
            self.connection.execute(
                f"DELETE FROM user_memories_fts WHERE memory_id IN ({placeholders})",
                ids,
            )
            rows = self.connection.execute(
                f"""SELECT id, guild_id, user_id, category, content, subject, project
                    FROM user_memories
                    WHERE id IN ({placeholders}) AND status = 'active'""",
                ids,
            ).fetchall()
            for row in rows:
                self._insert_fts_row(row)

    def add_user_memory(self, guild_id: int, user_id: int, category: str, content: str) -> UserMemory:
        memory, created = self._upsert_user_memory(
            guild_id,
            user_id,
            category,
            content,
            source="manual",
            importance=2,
        )
        if created:
            self._publish_memory_created(guild_id, user_id, memory, source="manual")
        return memory

    def add_user_memory_if_new(self, guild_id: int, user_id: int, category: str, content: str) -> UserMemory | None:
        memory, created = self._upsert_user_memory(
            guild_id,
            user_id,
            category,
            content,
            source="passive",
            importance=1,
        )
        if not created:
            return None
        self._publish_memory_created(guild_id, user_id, memory, source="passive")
        return memory

    def _upsert_user_memory(
        self,
        guild_id: int,
        user_id: int,
        category: str,
        content: str,
        *,
        source: str,
        importance: int,
        memory_kind: str | None = None,
        subject: str | None = None,
        project: str | None = None,
        confidence: float = 1.0,
        memory_key: str | None = None,
    ) -> tuple[UserMemory, bool]:
        normalized_category = normalize_memory_text(category)
        normalized_content = normalize_memory_text(content)
        if not normalized_category or len(normalized_category) > 40:
            raise ValueError("記憶分類需介於 1 到 40 個字元。")
        if not normalized_content or len(normalized_content) > 300:
            raise ValueError("記憶內容需介於 1 到 300 個字元。")
        if is_disallowed_memory(normalized_category, normalized_content):
            raise ValueError("這項內容像是敏感資料或修改墨雪規則的指令，因此不會存入記憶。")

        inferred_key, inferred_subject = infer_memory_key(normalized_category, normalized_content)
        resolved_key = normalize_memory_text(memory_key) if memory_key else inferred_key
        resolved_subject = normalize_memory_text(subject) if subject else inferred_subject
        resolved_project = normalize_memory_text(project) if project else infer_project_name(
            normalized_category, normalized_content
        )
        resolved_kind = normalize_memory_text(memory_kind) if memory_kind else infer_memory_kind(
            normalized_category, normalized_content
        )
        resolved_confidence = max(0.0, min(float(confidence), 1.0))
        resolved_importance = max(1, min(int(importance), 3))

        exact = self.connection.execute(
            """SELECT * FROM user_memories
               WHERE guild_id = ? AND user_id = ? AND category = ? AND content = ? AND status = 'active'
               ORDER BY id DESC LIMIT 1""",
            (guild_id, user_id, normalized_category, normalized_content),
        ).fetchone()
        if exact:
            memory_id = int(exact["id"])
            with self.connection:
                self.connection.execute(
                    """UPDATE user_memories
                       SET importance = MAX(importance, ?),
                           source = CASE WHEN ? = 'manual' THEN 'manual' ELSE source END,
                           memory_kind = ?,
                           subject = COALESCE(subject, ?),
                           project = COALESCE(project, ?),
                           memory_key = COALESCE(memory_key, ?),
                           confidence = MAX(confidence, ?),
                           last_confirmed = CURRENT_TIMESTAMP,
                           updated_at = CURRENT_TIMESTAMP
                       WHERE id = ?""",
                    (
                        resolved_importance,
                        source,
                        resolved_kind,
                        resolved_subject,
                        resolved_project,
                        resolved_key,
                        resolved_confidence,
                        memory_id,
                    ),
                )
            self._sync_fts_ids((memory_id,))
            row = self.connection.execute("SELECT * FROM user_memories WHERE id = ?", (memory_id,)).fetchone()
            assert row is not None
            return self._user_memory(row), False

        superseded_ids: list[int] = []
        if resolved_key:
            superseded_ids = [
                int(row["id"])
                for row in self.connection.execute(
                    """SELECT id FROM user_memories
                       WHERE guild_id = ? AND user_id = ? AND memory_key = ? AND status = 'active'""",
                    (guild_id, user_id, resolved_key),
                ).fetchall()
            ]

        with self.connection:
            cursor = self.connection.execute(
                """INSERT INTO user_memories(
                       guild_id, user_id, category, content, importance, source,
                       memory_kind, subject, project, memory_key, confidence, status, last_confirmed
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', CURRENT_TIMESTAMP)""",
                (
                    guild_id,
                    user_id,
                    normalized_category,
                    normalized_content,
                    resolved_importance,
                    source,
                    resolved_kind,
                    resolved_subject,
                    resolved_project,
                    resolved_key,
                    resolved_confidence,
                ),
            )
            new_id = int(cursor.lastrowid)
            if superseded_ids:
                placeholders = ",".join("?" for _ in superseded_ids)
                self.connection.execute(
                    f"""UPDATE user_memories
                        SET status = 'superseded', superseded_by = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id IN ({placeholders})""",
                    (new_id, *superseded_ids),
                )

        self._sync_fts_ids((*superseded_ids, new_id))
        row = self.connection.execute("SELECT * FROM user_memories WHERE id = ?", (new_id,)).fetchone()
        assert row is not None
        return self._user_memory(row), True

    def list_user_memories(self, guild_id: int, user_id: int, limit: int = 20) -> list[UserMemory]:
        rows = self.connection.execute(
            """SELECT * FROM user_memories
               WHERE guild_id = ? AND user_id = ? AND status = 'active'
               ORDER BY importance DESC, COALESCE(last_confirmed, updated_at, created_at) DESC, id DESC
               LIMIT ?""",
            (guild_id, user_id, max(1, min(limit, 100))),
        ).fetchall()
        return [
            self._user_memory(row)
            for row in rows
            if not is_disallowed_memory(str(row["category"]), str(row["content"]))
        ]

    def delete_user_memory(self, guild_id: int, user_id: int, memory_id: int) -> bool:
        owned = self.connection.execute(
            "SELECT id FROM user_memories WHERE id = ? AND guild_id = ? AND user_id = ?",
            (memory_id, guild_id, user_id),
        ).fetchone()
        if not owned:
            return False
        with self.connection:
            self.connection.execute(
                "DELETE FROM user_memories WHERE id = ? AND guild_id = ? AND user_id = ?",
                (memory_id, guild_id, user_id),
            )
            self.connection.execute(
                "UPDATE user_memories SET superseded_by = NULL WHERE superseded_by = ?",
                (memory_id,),
            )
        self._sync_fts_ids((memory_id,))
        return True

    def clear_user_memories(self, guild_id: int, user_id: int) -> int:
        rows = self.connection.execute(
            "SELECT id FROM user_memories WHERE guild_id = ? AND user_id = ?",
            (guild_id, user_id),
        ).fetchall()
        ids = tuple(int(row["id"]) for row in rows)
        if not ids:
            return 0
        with self.connection:
            self.connection.execute(
                "DELETE FROM user_memories WHERE guild_id = ? AND user_id = ?",
                (guild_id, user_id),
            )
            if self._memory_fts_enabled:
                self.connection.execute(
                    "DELETE FROM user_memories_fts WHERE guild_id = ? AND user_id = ?",
                    (guild_id, user_id),
                )
        return len(ids)

    def _fts_candidate_positions(self, guild_id: int, user_id: int, query: str) -> dict[int, int]:
        if not self._memory_fts_enabled:
            return {}
        fts_query = build_fts_query(query)
        if not fts_query:
            return {}
        try:
            rows = self.connection.execute(
                """SELECT memory_id, bm25(user_memories_fts) AS rank
                   FROM user_memories_fts
                   WHERE user_memories_fts MATCH ? AND guild_id = ? AND user_id = ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, guild_id, user_id, MEMORY_V2_FTS_CANDIDATES),
            ).fetchall()
        except sqlite3.OperationalError:
            LOGGER.debug("Memory FTS query failed; falling back to lexical ranking", exc_info=True)
            return {}
        return {int(row["memory_id"]): position for position, row in enumerate(rows)}

    def search_user_memories(
        self,
        guild_id: int,
        user_id: int,
        query: str,
        *,
        limit: int = 10,
    ) -> list[MemoryMatch]:
        normalized_query = normalize_memory_text(query)
        fts_positions = self._fts_candidate_positions(guild_id, user_id, normalized_query)
        rows = self.connection.execute(
            """SELECT * FROM user_memories
               WHERE guild_id = ? AND user_id = ? AND status = 'active'
               ORDER BY importance DESC, COALESCE(last_confirmed, last_used, updated_at, created_at) DESC, id DESC
               LIMIT ?""",
            (guild_id, user_id, MEMORY_V2_MAX_CANDIDATES),
        ).fetchall()
        by_id = {int(row["id"]): row for row in rows}

        missing_fts_ids = tuple(memory_id for memory_id in fts_positions if memory_id not in by_id)
        if missing_fts_ids:
            placeholders = ",".join("?" for _ in missing_fts_ids)
            for row in self.connection.execute(
                f"SELECT * FROM user_memories WHERE id IN ({placeholders}) AND status = 'active'",
                missing_fts_ids,
            ).fetchall():
                by_id[int(row["id"])] = row

        ranked: list[tuple[float, int, MemoryMatch]] = []
        for row in by_id.values():
            category = str(row["category"])
            content = str(row["content"])
            if is_disallowed_memory(category, content):
                continue
            memory_id = int(row["id"])
            score = score_memory_candidate(
                normalized_query,
                row,
                fts_position=fts_positions.get(memory_id),
            )
            match = MemoryMatch(
                id=memory_id,
                category=category,
                content=content,
                kind=str(row["memory_kind"] or MEMORY_KIND_SEMANTIC),
                subject=str(row["subject"]) if row["subject"] else None,
                project=str(row["project"]) if row["project"] else None,
                score=score,
            )
            ranked.append((score, int(row["importance"] or 1), match))

        ranked.sort(key=lambda item: (item[0], item[1], item[2].id), reverse=True)
        return [item[2] for item in ranked[: max(1, min(limit, 20))]]

    def agent_memory_lines(
        self,
        guild_id: int,
        user_id: int,
        limit: int = 8,
        *,
        query: str = "",
    ) -> list[str]:
        matches = self.search_user_memories(guild_id, user_id, query, limit=max(1, min(limit, 12)))
        return [self._format_memory_match(match, prefix=False) for match in matches]

    def user_memory_context(self, guild_id: int, user_id: int, query: str | None = None) -> str:
        matches = self.search_user_memories(guild_id, user_id, query or "", limit=12)
        lines: list[str] = []
        used_ids: list[int] = []
        remaining = MEMORY_V2_CONTEXT_CHARACTERS
        for match in matches:
            line = self._format_memory_match(match, prefix=True)
            if len(line) > remaining:
                continue
            lines.append(line)
            used_ids.append(match.id)
            remaining -= len(line) + 1

        if used_ids:
            placeholders = ",".join("?" for _ in used_ids)
            self.connection.execute(
                f"""UPDATE user_memories SET use_count = use_count + 1, last_used = CURRENT_TIMESTAMP
                    WHERE id IN ({placeholders})""",
                used_ids,
            )
            self.connection.commit()

        activity = self.user_activity(guild_id, user_id)
        if activity and (activity.message_count or activity.ai_request_count):
            lines.append(
                f"- [互動關係] 熟悉程度：{self.familiarity_label(guild_id, user_id)}；"
                f"訊息 {activity.message_count} 則，AI 提問 {activity.ai_request_count} 次。"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_memory_match(match: MemoryMatch, *, prefix: bool) -> str:
        label = match.category
        if match.project and match.project.casefold() not in label.casefold():
            label = f"{label}/{match.project}"
        body = f"[{label}] {match.content}"
        return f"- {body}" if prefix else body

    def familiarity_label(self, guild_id: int, user_id: int) -> str:
        activity = self.user_activity(guild_id, user_id)
        if not activity:
            return "初識"
        score = activity.message_count + (activity.ai_request_count * 3)
        if score < 12:
            return "初識"
        if score < 80:
            return "認識"
        return "熟悉"

    def _publish_memory_created(
        self,
        guild_id: int,
        user_id: int,
        memory: UserMemory,
        *,
        source: str,
    ) -> None:
        if self.event_bus is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            self.event_bus.publish(
                AgentEvent(
                    AgentEventKind.MEMORY_CREATED,
                    guild_id=guild_id,
                    user_id=user_id,
                    payload={
                        "memory_id": memory.id,
                        "category": memory.category,
                        "content": memory.content,
                        "source": source,
                    },
                )
            )
        )
