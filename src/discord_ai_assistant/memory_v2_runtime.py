from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from datetime import datetime
from typing import Any

import discord
from discord.ext import commands, tasks

from discord_ai_assistant.ai.gemini import GeminiRequestError
from discord_ai_assistant.ai.memory_domain_registry import DomainMatch, resolve_explicit_domain
from discord_ai_assistant.ai.memory_phase2 import (
    PassiveMemoryV2Draft,
    extract_passive_memory_v2_drafts,
    is_unified_passive_memory_candidate,
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
from discord_ai_assistant.ai.memory_retention import reinforce_personal_memory
from discord_ai_assistant.ai.memory_session import ConversationSessionState, SessionMessage
from discord_ai_assistant.ai.memory_shared import build_event_signature, record_shared_candidate
from discord_ai_assistant.ai.memory_social_policy import (
    refresh_social_referenceability,
    resolve_channel_memory_policy,
)

LOGGER = logging.getLogger(__name__)
MEMORY_V2_BATCH_SIZE = 8
_SHARED_ONLY_MEMORY_KINDS = {"game_episode", "shared_episode", "inside_joke"}
_SHARED_GAME_EPISODE_MARKERS = (
    "上次",
    "那次",
    "之前",
    "昨天",
    "後來",
    "后来",
    "剛剛大家",
    "刚刚大家",
    "大家剛剛",
    "大家刚刚",
    "我們上次",
    "我们上次",
)


def _publish_created(database: Any, guild_id: int, user_id: int, memory: Any, source: str) -> None:
    publisher = getattr(database, "_publish_memory_created", None)
    if callable(publisher):
        publisher(guild_id, user_id, memory, source=source)


def _explicit_game_scope(content: str) -> DomainMatch | None:
    """Return a strong explicit game scope without requiring the exact episode wording."""

    explicit = resolve_explicit_domain(normalize_passive_message(content))
    if explicit is None or explicit.domain != "game" or not explicit.subdomain:
        return None
    return explicit


def _explicit_shared_game_scope(content: str) -> DomainMatch | None:
    """Return strong local game scope only for messages that describe a past episode."""

    normalized = normalize_passive_message(content)
    explicit = _explicit_game_scope(normalized)
    if explicit is None:
        return None
    if not any(marker in normalized for marker in _SHARED_GAME_EPISODE_MARKERS):
        return None
    return explicit


def _reconcile_game_episode_scope(
    draft: PassiveMemoryV2Draft,
    observations: list[MemoryObservation],
) -> PassiveMemoryV2Draft:
    """Let corroborated local game evidence override a mistaken AI project label.

    Gemini can occasionally interpret a named game such as Minecraft as a project and
    paraphrase away the original episode marker. For project-labelled event drafts,
    the generated text only needs to identify the same explicit game; the original
    Discord observation remains authoritative for the shared-episode signal and must
    also come from a channel where shared memory is allowed. The extractor may label
    the category as either ``事件`` or the broader ``專案`` while still assigning the
    authoritative project role ``event``; both forms are accepted here. This keeps
    true project facts/status on the project path while preventing public game episodes
    from polluting project state.
    """

    if (
        not draft.project
        or not observations
        or draft.category not in {"事件", "專案"}
        or draft.role != "event"
    ):
        return draft
    draft_scope = _explicit_game_scope(draft.content)
    if draft_scope is None:
        return draft

    matching_scopes: set[tuple[str, str]] = set()
    for item in observations:
        if not item.shared_allowed:
            continue
        observation_scope = _explicit_shared_game_scope(item.content)
        if observation_scope is None or observation_scope.subdomain != draft_scope.subdomain:
            continue
        if item.domain != observation_scope.domain or item.subdomain != observation_scope.subdomain:
            continue
        matching_scopes.add((observation_scope.domain, observation_scope.subdomain))
    if len(matching_scopes) != 1:
        return draft

    domain, subdomain = next(iter(matching_scopes))
    return replace(
        draft,
        project=None,
        role=None,
        domain=domain,
        subdomain=subdomain,
        memory_kind="shared_episode",
        shared_candidate=True,
    )


def _observations_for_draft(
    draft: PassiveMemoryV2Draft,
    observations: list[MemoryObservation],
) -> list[MemoryObservation]:
    """Use only locally corroborated provenance for domain-tagged drafts.

    Untagged legacy/generic drafts can still use the whole extraction batch. Once a
    draft claims a domain/subdomain, however, unrelated mixed-channel observations
    must never be borrowed as provenance, reinforcement, or public social evidence.
    """

    if not observations:
        return []
    if not draft.domain:
        return observations
    return [
        item
        for item in observations
        if item.domain == draft.domain
        and (draft.subdomain is None or item.subdomain == draft.subdomain)
    ]


def _shared_observations_for_draft(
    draft: PassiveMemoryV2Draft,
    observations: list[MemoryObservation],
) -> list[MemoryObservation]:
    """Shared evidence must match the draft domain and an allow-shared channel policy."""

    if not observations or not draft.domain:
        return []
    return [
        item
        for item in observations
        if item.shared_allowed
        and item.domain == draft.domain
        and (draft.subdomain is None or item.subdomain == draft.subdomain)
    ]


def _observation_datetime(observation: MemoryObservation) -> datetime | None:
    if not observation.created_at:
        return None
    candidate = observation.created_at.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None


def _reinforce_from_new_sessions(
    database: Any,
    memory_id: int,
    observations: list[MemoryObservation],
) -> None:
    """Credit each newly observed non-null session at most once for this extraction."""

    seen: set[str] = set()
    for item in observations:
        session_key = str(item.session_key or "").strip()
        if not session_key or session_key in seen:
            continue
        seen.add(session_key)
        reinforce_personal_memory(
            database,
            memory_id,
            session_key=session_key,
            now=_observation_datetime(item),
        )


def store_passive_memory_v2_drafts(
    database: Any,
    guild_id: int,
    user_id: int,
    drafts: list[PassiveMemoryV2Draft],
    *,
    observations: list[MemoryObservation] | None = None,
) -> int:
    """Persist validated personal/project drafts and route public group episodes separately."""

    ensure_memory_phase4_schema(database)
    stored = 0
    touched_projects: set[str] = set()
    source_observations = observations or []

    for draft in drafts:
        draft = _reconcile_game_episode_scope(draft, source_observations)
        shared_only = bool(
            draft.shared_candidate
            and not draft.project
            and (draft.memory_kind in _SHARED_ONLY_MEMORY_KINDS or draft.shared_group_event)
        )
        if draft.shared_candidate and not draft.project and draft.domain:
            for item in _shared_observations_for_draft(draft, source_observations):
                record_shared_candidate(
                    database,
                    guild_id=guild_id,
                    author_id=user_id,
                    content=draft.content,
                    domain=draft.domain,
                    subdomain=draft.subdomain,
                    memory_kind=draft.memory_kind or "shared_episode",
                    confidence=draft.confidence,
                    importance=draft.importance,
                    entity_type=draft.entity_type,
                    entity=draft.entity,
                    session_key=item.session_key,
                    channel_id=item.channel_id,
                    message_id=item.message_id,
                    observed_at=item.created_at,
                    event_signature=build_event_signature(item.content),
                    # This extractor batch is scoped to one author. A model label that
                    # describes a group event is not evidence that multiple members
                    # corroborated this exact event; distinct authors must provide their
                    # own observations before promotion.
                    shared_group_event=False,
                    participant_ids=(user_id,),
                )
        if shared_only:
            # A public group episode belongs to the guild store, not to a synthetic
            # personal-memory row. If policy disallows sharing, it is simply dropped.
            continue

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
                    memory_kind=draft.memory_kind,
                    domain=draft.domain,
                    subdomain=draft.subdomain,
                    entity_type=draft.entity_type,
                    entity=draft.entity,
                    retention=draft.retention,
                    # Candidate status alone never grants cross-user disclosure.
                    socially_referenceable=False,
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
            provenance = _observations_for_draft(draft, source_observations)
            if provenance:
                record_memory_provenance(
                    database,
                    guild_id,
                    user_id,
                    memory_id,
                    provenance,
                )
            if not draft.project:
                if provenance and not created:
                    _reinforce_from_new_sessions(database, memory_id, provenance)
                refresh_social_referenceability(database, memory_id)
            record_potential_conflicts(database, guild_id, user_id, memory_id)

    for project in touched_projects:
        refresh_project_summary(database, guild_id, user_id, project)
    return stored


class MemoryV2PassiveRuntime(commands.Cog):
    """The sole passive Memory V2 observer/extractor.

    The Cog never replies to Discord. It also feeds the shared ephemeral conversation
    session tracker so durable extraction can resolve short game/social context without
    paying for a Gemini classification request on every message.
    """

    def __init__(
        self,
        bot: commands.Bot,
        database: Any,
        ai: Any,
        session_state: ConversationSessionState,
    ) -> None:
        self.bot = bot
        self.database = database
        self.ai = ai
        self.session_state = session_state
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

    @staticmethod
    def _category_name(message: discord.Message) -> str | None:
        category = getattr(message.channel, "category", None)
        name = getattr(category, "name", None)
        return str(name) if name else None

    @staticmethod
    def _reply_message_id(message: discord.Message) -> int | None:
        reference = getattr(message, "reference", None)
        message_id = getattr(reference, "message_id", None)
        return int(message_id) if isinstance(message_id, int) else None

    @staticmethod
    def _policy_prior(policy: Any) -> DomainMatch | None:
        if not policy.domain_prior:
            return None
        return DomainMatch(
            domain=str(policy.domain_prior),
            subdomain=str(policy.subdomain_prior) if policy.subdomain_prior else None,
            topic=None,
            confidence=0.75,
            source="channel_policy",
        )

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or not message.guild:
            return
        guild_id, user_id = message.guild.id, message.author.id
        if not self.passive_enabled(guild_id, user_id):
            self._pending.pop((guild_id, user_id), None)
            return

        content = normalize_passive_message(message.clean_content)
        channel_name = str(getattr(message.channel, "name", "") or "")
        policy = resolve_channel_memory_policy(
            self.database,
            guild_id=guild_id,
            channel_id=message.channel.id,
            channel_name=channel_name,
            category_name=self._category_name(message),
        )
        if not policy.enabled:
            return

        explicit = resolve_explicit_domain(content)
        resolution = self.session_state.observe(
            SessionMessage(
                channel_id=message.channel.id,
                author_id=user_id,
                message_id=message.id,
                content=content,
                reply_to_message_id=self._reply_message_id(message),
            ),
            explicit=explicit,
            channel_prior=self._policy_prior(policy),
        )

        if not (policy.allow_personal or policy.allow_shared) or not is_unified_passive_memory_candidate(content):
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
                session_key=resolution.session_key if resolution else None,
                domain=resolution.domain if resolution else (explicit.domain if explicit else None),
                subdomain=resolution.subdomain if resolution else (explicit.subdomain if explicit else None),
                topic=resolution.topic if resolution else (explicit.topic if explicit else None),
                shared_allowed=bool(policy.allow_shared),
                participant_ids=tuple(resolution.participant_ids) if resolution else (user_id,),
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
