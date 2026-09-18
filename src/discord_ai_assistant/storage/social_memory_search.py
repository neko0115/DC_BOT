from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from discord_ai_assistant.ai.memory import is_disallowed_memory
from discord_ai_assistant.ai.memory_retention import RETENTION_LONG
from discord_ai_assistant.ai.memory_v2 import (
    MEMORY_KIND_SEMANTIC,
    MemoryMatch,
    normalize_memory_text,
    score_memory_candidate,
)


SocialRowValidator = Callable[[Any], bool]


def search_socially_referenceable_memories(
    database: Any,
    guild_id: int,
    user_ids: Iterable[int],
    query: str,
    *,
    exclude_user_id: int | None = None,
    limit: int = 2,
    validator: SocialRowValidator | None = None,
) -> list[MemoryMatch]:
    """Search only explicitly referenceable memories for supplied active participants.

    This is deliberately separate from owner-scoped memory retrieval.  The caller must
    supply the participant ids from the active conversation session; no guild-wide
    discovery is performed here.
    """

    ids = tuple(
        dict.fromkeys(
            int(user_id)
            for user_id in user_ids
            if int(user_id) > 0 and (exclude_user_id is None or int(user_id) != int(exclude_user_id))
        )
    )
    if not ids:
        return []

    expire = getattr(database, "expire_due_user_memories", None)
    if callable(expire):
        expire(int(guild_id))
    connection = getattr(database, "connection", None)
    if connection is None:
        return []

    placeholders = ",".join("?" for _ in ids)
    rows = connection.execute(
        f"""SELECT * FROM user_memories
            WHERE guild_id = ?
              AND user_id IN ({placeholders})
              AND status = 'active'
              AND socially_referenceable = 1
            ORDER BY importance DESC,
                     COALESCE(last_confirmed, last_used, updated_at, created_at) DESC,
                     id DESC
            LIMIT 256""",
        (int(guild_id), *ids),
    ).fetchall()

    normalized_query = normalize_memory_text(query)
    ranked: list[tuple[float, int, MemoryMatch]] = []
    for row in rows:
        category = str(row["category"])
        content = str(row["content"])
        if is_disallowed_memory(category, content):
            continue
        if validator is not None and not validator(row):
            continue
        score = float(score_memory_candidate(normalized_query, row))
        match = MemoryMatch(
            id=int(row["id"]),
            category=category,
            content=content,
            kind=str(row["memory_kind"] or MEMORY_KIND_SEMANTIC),
            subject=str(row["subject"]) if row["subject"] else None,
            project=str(row["project"]) if row["project"] else None,
            score=score,
            domain=str(row["domain"]) if row["domain"] else None,
            subdomain=str(row["subdomain"]) if row["subdomain"] else None,
            entity_type=str(row["entity_type"]) if row["entity_type"] else None,
            entity=str(row["entity"]) if row["entity"] else None,
            retention=str(row["retention"] or RETENTION_LONG),
            reinforcement_count=int(row["reinforcement_count"] or 0),
            socially_referenceable=bool(row["socially_referenceable"]),
        )
        ranked.append((score, int(row["importance"] or 1), match))

    ranked.sort(key=lambda item: (item[0], item[1], item[2].id), reverse=True)
    return [item[2] for item in ranked[: max(1, min(int(limit), 20))]]


def install_agent_database_social_search(validator: SocialRowValidator) -> None:
    """Expose the dedicated search on AgentDatabase without widening base Database API."""

    from discord_ai_assistant.storage.agent_database import AgentDatabase

    if hasattr(AgentDatabase, "search_socially_referenceable_memories"):
        return

    def _search(
        self: AgentDatabase,
        guild_id: int,
        user_ids: Iterable[int],
        query: str,
        *,
        exclude_user_id: int | None = None,
        limit: int = 2,
    ) -> list[MemoryMatch]:
        return search_socially_referenceable_memories(
            self,
            guild_id,
            user_ids,
            query,
            exclude_user_id=exclude_user_id,
            limit=limit,
            validator=validator,
        )

    setattr(AgentDatabase, "search_socially_referenceable_memories", _search)
