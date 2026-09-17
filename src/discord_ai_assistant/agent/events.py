from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


class AgentEventKind(StrEnum):
    BOT_READY = "bot_ready"
    MESSAGE_RECEIVED = "message_received"
    MEMBER_JOINED_VOICE = "member_joined_voice"
    MEMBER_LEFT_VOICE = "member_left_voice"
    SONG_STARTED = "song_started"
    SONG_ENDED = "song_ended"
    CONVERSATION_IDLE = "conversation_idle"
    TOOL_COMPLETED = "tool_completed"
    MEMORY_CREATED = "memory_created"


@dataclass(frozen=True, slots=True)
class AgentEvent:
    kind: AgentEventKind
    guild_id: int | None = None
    channel_id: int | None = None
    user_id: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


EventHandler = Callable[[AgentEvent], Awaitable[None]]


class AgentEventBus:
    """Small async event bus used to decouple Discord callbacks from agent decisions."""

    def __init__(self) -> None:
        self._handlers: dict[AgentEventKind, list[EventHandler]] = defaultdict(list)
        self._all_handlers: list[EventHandler] = []
        self._lock = asyncio.Lock()

    async def subscribe(self, kind: AgentEventKind | None, handler: EventHandler) -> None:
        async with self._lock:
            target = self._all_handlers if kind is None else self._handlers[kind]
            if handler not in target:
                target.append(handler)

    async def unsubscribe(self, kind: AgentEventKind | None, handler: EventHandler) -> None:
        async with self._lock:
            target = self._all_handlers if kind is None else self._handlers[kind]
            if handler in target:
                target.remove(handler)

    async def publish(self, event: AgentEvent) -> None:
        async with self._lock:
            handlers = tuple(self._all_handlers) + tuple(self._handlers.get(event.kind, ()))
        for handler in handlers:
            await handler(event)
