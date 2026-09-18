from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import discord
from discord.ext import commands, tasks

from discord_ai_assistant.ai.gemini import GeminiRequestError
from discord_ai_assistant.ai.memory_phase2 import (
    PassiveMemoryV2Draft,
    extract_passive_memory_v2_drafts,
    is_extended_passive_memory_candidate,
    normalize_passive_message,
)
from discord_ai_assistant.ai.memory_phase4 import (
    MemoryObservation,
    batch_flush_policy,
    ensure_memory_phase4_schema,
    memory_urgency,
    record_memory_provenance,
    record_potential_conflicts,
)
from discord_ai_assistant.ai.memory_project_state import (
    refresh_project_summary,
    store_structured_project_memory,
)

LOGGER = logging.getLogger(__name__)
MEMORY_V2_BATCH_SIZE = 8


def _publish_created(database: Any, guild_id: int, user_id: int, memory: Any, source: str) -> None:
    publisher = getattr(database, "_publish_memory_created", None)
    if callable(publisher):
        publisher(guild_id, user_id, memory, source=source)


def store_passive_memory_v2_drafts(
    database: Any,
    guild_id: int,
    user_id: int,
    drafts: list[PassiveMemoryV2Draft],
    *,
    observations: list[MemoryObservation] | None = None,
) -> int:
    """Persist validated drafts, provenance, conflicts, and derived project summaries."""

    ensure_memory_phase4_schema(database)
    stored = 0
    touched_projects: set[str] = set()
    source_observations = observations or []

    for draft in drafts:
        memory: Any | None = None
        created = False
        if draft.project:
            memory, created = store_structured_project_memory(
                database,
                guild_id,
                user_id,
                draft,
            )
            if memory is not None:
                touched_projects.add(draft.project)
        else:
            upsert = getattr(database, "_upsert_user_memory", None)
            if callable(upsert):
                memory, created = upsert(
                    guild_id,
                    user_id,
                    f"自動{draft.category}",
                    draft.content,
                    source="passive",
                    importance=draft.importance,
                    confidence=draft.confidence,
                )
                if created:
                    _publish_created(database, guild_id, user_id, memory, "passive")
            else:
                memory = database.add_user_memory_if_new(
                    guild_id,
                    user_id,
                    f"自動{draft.category}",
                    draft.content,
                )
                created = memory is not None

        stored += int(created)
        if memory is not None:
            memory_id = int(memory.id)
            if source_observations:
                record_memory_provenance(
                    database,
                    guild_id,
                    user_id,
                    memory_id,
                    source_observations,
                )
            record_potential_conflicts(database, guild_id, user_id, memory_id)

    for project in touched_projects:
        refresh_project_summary(database, guild_id, user_id, project)
    return stored


class MemoryV2PassiveRuntime(commands.Cog):
    """Side-effect-free Discord observer for richer passive Memory V2 extraction.

    This Cog never replies to Discord. It batches durable-memory candidates and uses a
    shorter flush window for high-impact state changes while retaining the conservative
    normal cadence for ordinary project/context updates.
    """

    def __init__(self, bot: commands.Bot, database: Any, ai: Any) -> None:
        self.bot = bot
        self.database = database
        self.ai = ai
        ensure_memory_phase4_schema(database)
        self._pending: dict[tuple[int, int], list[MemoryObservation]] = {}
        self._last_extraction: dict[tuple[int, int], float] = {}
        self._lock = asyncio.Lock()

    async def cog_load(self) -> None:
        self._flush.start()

    def cog_unload(self) -> None:
        self._flush.cancel()
        self._pending.clear()

    def passive_enabled(self, guild_id: int, user_id: int) -> bool:
        return self.database.get_state(f"passive_memory_enabled:{guild_id}:{user_id}") != "0"

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or not message.guild:
            return
        guild_id, user_id = message.guild.id, message.author.id
        if not self.passive_enabled(guild_id, user_id):
            self._pending.pop((guild_id, user_id), None)
            return

        content = normalize_passive_message(message.clean_content)
        if not is_extended_passive_memory_candidate(content):
            return
        key = (guild_id, user_id)
        entries = self._pending.setdefault(key, [])
        if entries and entries[-1].content == content:
            return
        created_at = None
        message_created_at = getattr(message, "created_at", None)
        if message_created_at is not None:
            try:
                created_at = message_created_at.isoformat()
            except (AttributeError, ValueError):
                created_at = None
        entries.append(
            MemoryObservation(
                monotonic_at=time.monotonic(),
                content=content,
                channel_id=message.channel.id,
                message_id=message.id,
                created_at=created_at,
                urgency=memory_urgency(content),
            )
        )
        del entries[:-MEMORY_V2_BATCH_SIZE]

    @tasks.loop(minutes=1)
    async def _flush(self) -> None:
        if not getattr(self.ai, "enabled", False) or self._lock.locked():
            return
        now = time.monotonic()
        selected: tuple[tuple[int, int], list[MemoryObservation]] | None = None
        for key, entries in tuple(self._pending.items()):
            if not self.passive_enabled(*key):
                self._pending.pop(key, None)
                continue
            if not entries:
                continue
            idle_seconds, min_interval_seconds = batch_flush_policy(entries)
            if (
                now - entries[-1].monotonic_at >= idle_seconds
                and now - self._last_extraction.get(key, 0) >= min_interval_seconds
            ):
                selected = (key, list(entries))
                break
        if selected is None:
            return

        key, entries = selected
        self._pending.pop(key, None)
        self._last_extraction[key] = now
        guild_id, user_id = key
        try:
            async with self._lock:
                drafts = await extract_passive_memory_v2_drafts(
                    self.ai,
                    [item.content for item in entries],
                )
            stored = store_passive_memory_v2_drafts(
                self.database,
                guild_id,
                user_id,
                drafts,
                observations=entries,
            )
            if stored:
                LOGGER.info(
                    "Stored %s Memory V2 passive memories for user %s in guild %s",
                    stored,
                    user_id,
                    guild_id,
                )
        except (GeminiRequestError, RuntimeError, ValueError):
            LOGGER.warning(
                "Memory V2 passive extraction failed for user %s in guild %s",
                user_id,
                guild_id,
                exc_info=True,
            )
        except Exception:
            LOGGER.exception(
                "Unexpected Memory V2 passive extraction failure for user %s in guild %s",
                user_id,
                guild_id,
            )

    @_flush.before_loop
    async def _before_flush(self) -> None:
        await self.bot.wait_until_ready()
