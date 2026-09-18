from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable

from discord_ai_assistant.ai.memory_project_state import role_from_category
from discord_ai_assistant.ai.memory_v2 import memory_terms, normalize_memory_text

MEMORY_URGENCY_NORMAL = "normal"
MEMORY_URGENCY_HIGH = "high"

NORMAL_IDLE_SECONDS = 5 * 60
NORMAL_MIN_INTERVAL_SECONDS = 30 * 60
HIGH_IDLE_SECONDS = 60
HIGH_MIN_INTERVAL_SECONDS = 2 * 60

_HIGH_URGENCY_MARKERS = (
    "blocker",
    "卡住",
    "卡在",
    "下一步",
    "決定",
    "改成",
    "改用",
    "換成",
    "修好",
    "修正",
    "完成",
    "通過",
    "失敗",
    "merge",
    "merged",
    "release",
    "版本",
    "現在是",
    "目前是",
    "現在用",
    "目前用",
)
_PROJECT_PREFIX = re.compile(r"^專案\s+[^：:]{1,40}[：:]\s*", re.IGNORECASE)
_FACT_STATE_MARKERS = ("是", "為", "使用", "採用", "定義", "版本", "設定", "架構", "模型", "改成", "改用")


@dataclass(frozen=True, slots=True)
class MemoryObservation:
    monotonic_at: float
    content: str
    channel_id: int
    message_id: int
    created_at: str | None
    urgency: str = MEMORY_URGENCY_NORMAL


def memory_urgency(content: str) -> str:
    normalized = normalize_memory_text(content).casefold()
    return MEMORY_URGENCY_HIGH if any(marker.casefold() in normalized for marker in _HIGH_URGENCY_MARKERS) else MEMORY_URGENCY_NORMAL


def flush_policy_for_urgency(urgency: str) -> tuple[float, float]:
    if urgency == MEMORY_URGENCY_HIGH:
        return float(HIGH_IDLE_SECONDS), float(HIGH_MIN_INTERVAL_SECONDS)
    return float(NORMAL_IDLE_SECONDS), float(NORMAL_MIN_INTERVAL_SECONDS)


def batch_flush_policy(observations: Iterable[MemoryObservation]) -> tuple[float, float]:
    values = list(observations)
    if any(item.urgency == MEMORY_URGENCY_HIGH for item in values):
        return flush_policy_for_urgency(MEMORY_URGENCY_HIGH)
    return flush_policy_for_urgency(MEMORY_URGENCY_NORMAL)


def ensure_memory_phase4_schema(database: Any) -> None:
    connection = getattr(database, "connection", None)
    if connection is None:
        return
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS memory_provenance (
            id INTEGER PRIMARY KEY,
            memory_id INTEGER NOT NULL REFERENCES user_memories(id) ON DELETE CASCADE,
            guild_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            channel_id INTEGER NOT NULL,
            message_id INTEGER NOT NULL,
            observed_at TEXT,
            batch_id TEXT NOT NULL,
            provenance_kind TEXT NOT NULL DEFAULT 'passive_batch',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(memory_id, message_id, batch_id)
        );
        CREATE INDEX IF NOT EXISTS idx_memory_provenance_owner
            ON memory_provenance(guild_id, user_id, memory_id, id DESC);

        CREATE TABLE IF NOT EXISTS memory_conflicts (
            id INTEGER PRIMARY KEY,
            guild_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            left_memory_id INTEGER NOT NULL REFERENCES user_memories(id) ON DELETE CASCADE,
            right_memory_id INTEGER NOT NULL REFERENCES user_memories(id) ON DELETE CASCADE,
            project TEXT,
            similarity REAL NOT NULL,
            reason TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(left_memory_id, right_memory_id)
        );
        CREATE INDEX IF NOT EXISTS idx_memory_conflicts_owner
            ON memory_conflicts(guild_id, user_id, status, id DESC);
        """
    )
    connection.commit()


def _batch_id(observations: Iterable[MemoryObservation]) -> str:
    payload = ",".join(str(item.message_id) for item in observations)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def record_memory_provenance(
    database: Any,
    guild_id: int,
    user_id: int,
    memory_id: int,
    observations: Iterable[MemoryObservation],
) -> int:
    items = list(observations)
    if not items:
        return 0
    ensure_memory_phase4_schema(database)
    connection = getattr(database, "connection", None)
    if connection is None:
        return 0
    batch_id = _batch_id(items)
    inserted = 0
    with connection:
        for item in items:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO memory_provenance(
                       memory_id, guild_id, user_id, channel_id, message_id,
                       observed_at, batch_id, provenance_kind
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, 'passive_batch')""",
                (
                    int(memory_id),
                    int(guild_id),
                    int(user_id),
                    int(item.channel_id),
                    int(item.message_id),
                    item.created_at,
                    batch_id,
                ),
            )
            inserted += max(0, int(cursor.rowcount))
    return inserted


def memory_provenance_rows(
    database: Any,
    guild_id: int,
    user_id: int,
    memory_id: int,
    *,
    limit: int = 10,
) -> list[Any]:
    ensure_memory_phase4_schema(database)
    connection = getattr(database, "connection", None)
    if connection is None:
        return []
    owned = connection.execute(
        "SELECT id FROM user_memories WHERE id = ? AND guild_id = ? AND user_id = ?",
        (memory_id, guild_id, user_id),
    ).fetchone()
    if not owned:
        return []
    return list(
        connection.execute(
            """SELECT channel_id, message_id, observed_at, batch_id, provenance_kind, created_at
               FROM memory_provenance
               WHERE memory_id = ? AND guild_id = ? AND user_id = ?
               ORDER BY id DESC LIMIT ?""",
            (memory_id, guild_id, user_id, max(1, min(limit, 50))),
        ).fetchall()
    )


def _fact_terms(content: str) -> set[str]:
    stripped = _PROJECT_PREFIX.sub("", normalize_memory_text(content), count=1)
    return set(memory_terms(stripped))


def _similarity(left: str, right: str) -> float:
    left_terms = _fact_terms(left)
    right_terms = _fact_terms(right)
    if not left_terms or not right_terms:
        return 0.0
    intersection = len(left_terms & right_terms)
    union = len(left_terms | right_terms)
    return intersection / union if union else 0.0


def record_potential_conflicts(database: Any, guild_id: int, user_id: int, memory_id: int) -> int:
    """Record conservative project-fact conflicts without automatically overwriting either fact."""

    ensure_memory_phase4_schema(database)
    connection = getattr(database, "connection", None)
    if connection is None:
        return 0
    current = connection.execute(
        """SELECT id, category, content, project, status FROM user_memories
           WHERE id = ? AND guild_id = ? AND user_id = ?""",
        (memory_id, guild_id, user_id),
    ).fetchone()
    if not current or current["status"] != "active" or not current["project"]:
        return 0
    if role_from_category(str(current["category"])) != "fact":
        return 0
    current_content = str(current["content"])
    if not any(marker in current_content for marker in _FACT_STATE_MARKERS):
        return 0

    others = connection.execute(
        """SELECT id, category, content FROM user_memories
           WHERE guild_id = ? AND user_id = ? AND project = ? AND status = 'active' AND id != ?
           ORDER BY id DESC LIMIT 64""",
        (guild_id, user_id, str(current["project"]), memory_id),
    ).fetchall()
    inserted = 0
    with connection:
        for other in others:
            if role_from_category(str(other["category"])) != "fact":
                continue
            other_content = str(other["content"])
            if current_content == other_content or not any(marker in other_content for marker in _FACT_STATE_MARKERS):
                continue
            similarity = _similarity(current_content, other_content)
            if similarity < 0.55:
                continue
            left_id, right_id = sorted((int(current["id"]), int(other["id"])))
            cursor = connection.execute(
                """INSERT OR IGNORE INTO memory_conflicts(
                       guild_id, user_id, left_memory_id, right_memory_id,
                       project, similarity, reason, status
                   ) VALUES (?, ?, ?, ?, ?, ?, 'overlapping_project_fact', 'open')""",
                (
                    guild_id,
                    user_id,
                    left_id,
                    right_id,
                    str(current["project"]),
                    round(similarity, 6),
                ),
            )
            inserted += max(0, int(cursor.rowcount))
    return inserted


def list_open_memory_conflicts(database: Any, guild_id: int, user_id: int, *, limit: int = 20) -> list[Any]:
    ensure_memory_phase4_schema(database)
    connection = getattr(database, "connection", None)
    if connection is None:
        return []
    return list(
        connection.execute(
            """SELECT c.id, c.project, c.similarity, c.reason,
                      l.id AS left_id, l.content AS left_content,
                      r.id AS right_id, r.content AS right_content
               FROM memory_conflicts c
               JOIN user_memories l ON l.id = c.left_memory_id
               JOIN user_memories r ON r.id = c.right_memory_id
               WHERE c.guild_id = ? AND c.user_id = ? AND c.status = 'open'
               ORDER BY c.id DESC LIMIT ?""",
            (guild_id, user_id, max(1, min(limit, 50))),
        ).fetchall()
    )
