from __future__ import annotations

import asyncio
import sqlite3

from discord_ai_assistant.agent.events import AgentEvent, AgentEventBus, AgentEventKind
from discord_ai_assistant.ai.memory import is_disallowed_memory
from discord_ai_assistant.models import UserMemory
from discord_ai_assistant.storage.database import Database, MAX_MEMORY_CONTEXT_CHARACTERS


class AgentDatabase(Database):
    """Existing SQLite store plus lightweight memory metadata for Agent Core."""

    def __init__(self, path, event_bus: AgentEventBus | None = None) -> None:
        self.event_bus = event_bus
        super().__init__(path)
        self._ensure_agent_memory_columns()

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
        }
        for name, definition in additions.items():
            if name not in existing:
                self.connection.execute(f"ALTER TABLE user_memories ADD COLUMN {name} {definition}")
        self.connection.commit()

    def add_user_memory(self, guild_id: int, user_id: int, category: str, content: str) -> UserMemory:
        memory = self._add_user_memory_with_metadata(
            guild_id,
            user_id,
            category,
            content,
            source="manual",
            importance=2,
        )
        self._publish_memory_created(guild_id, user_id, memory, source="manual")
        return memory

    def add_user_memory_if_new(self, guild_id: int, user_id: int, category: str, content: str) -> UserMemory | None:
        normalized_content = " ".join(content.split())
        existing = self.connection.execute(
            "SELECT id FROM user_memories WHERE guild_id = ? AND user_id = ? AND content = ?",
            (guild_id, user_id, normalized_content),
        ).fetchone()
        if existing:
            return None
        memory = self._add_user_memory_with_metadata(
            guild_id,
            user_id,
            category,
            normalized_content,
            source="passive",
            importance=1,
        )
        self._publish_memory_created(guild_id, user_id, memory, source="passive")
        return memory

    def _add_user_memory_with_metadata(
        self,
        guild_id: int,
        user_id: int,
        category: str,
        content: str,
        *,
        source: str,
        importance: int,
    ) -> UserMemory:
        normalized_category = " ".join(category.split())
        normalized_content = " ".join(content.split())
        if not normalized_category or len(normalized_category) > 40:
            raise ValueError("記憶分類需介於 1 到 40 個字元。")
        if not normalized_content or len(normalized_content) > 300:
            raise ValueError("記憶內容需介於 1 到 300 個字元。")
        if is_disallowed_memory(normalized_category, normalized_content):
            raise ValueError("這項內容像是敏感資料或修改墨雪規則的指令，因此不會存入記憶。")
        cursor = self.connection.execute(
            """INSERT INTO user_memories(guild_id, user_id, category, content, importance, source)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (guild_id, user_id, normalized_category, normalized_content, max(1, min(importance, 3)), source),
        )
        self.connection.commit()
        row = self.connection.execute("SELECT * FROM user_memories WHERE id = ?", (cursor.lastrowid,)).fetchone()
        assert row is not None
        return self._user_memory(row)

    def agent_memory_lines(self, guild_id: int, user_id: int, limit: int = 8) -> list[str]:
        rows = self.connection.execute(
            """SELECT id, category, content FROM user_memories
               WHERE guild_id = ? AND user_id = ?
               ORDER BY importance DESC, COALESCE(last_used, created_at) DESC, id DESC
               LIMIT ?""",
            (guild_id, user_id, max(1, min(limit, 12))),
        ).fetchall()
        return [
            f"[{row['category']}] {row['content']}"
            for row in reversed(rows)
            if not is_disallowed_memory(str(row["category"]), str(row["content"]))
        ]

    def user_memory_context(self, guild_id: int, user_id: int) -> str:
        rows = self.connection.execute(
            """SELECT id, category, content FROM user_memories
               WHERE guild_id = ? AND user_id = ?
               ORDER BY importance DESC, COALESCE(last_used, created_at) DESC, id DESC
               LIMIT 12""",
            (guild_id, user_id),
        ).fetchall()
        lines: list[str] = []
        used_ids: list[int] = []
        remaining = MAX_MEMORY_CONTEXT_CHARACTERS
        for row in reversed(rows):
            if is_disallowed_memory(str(row["category"]), str(row["content"])):
                continue
            line = f"- [{row['category']}] {row['content']}"
            if len(line) > remaining:
                break
            lines.append(line)
            used_ids.append(int(row["id"]))
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
