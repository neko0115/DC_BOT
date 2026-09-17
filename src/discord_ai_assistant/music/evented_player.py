from __future__ import annotations

from discord_ai_assistant.agent.events import AgentEvent, AgentEventBus, AgentEventKind
from discord_ai_assistant.music.enhanced_player import EnhancedMusicManager


class EventedEnhancedMusicManager(EnhancedMusicManager):
    """Enhanced player that reports track lifecycle to Agent Core without changing playback."""

    def __init__(self, *args, event_bus: AgentEventBus | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.event_bus = event_bus

    async def _start_if_idle(self, guild_id: int) -> bool:
        started = await super()._start_if_idle(guild_id)
        if started and self.event_bus is not None:
            current = self.state_for(guild_id).queue.current
            if current is not None:
                await self.event_bus.publish(
                    AgentEvent(
                        AgentEventKind.SONG_STARTED,
                        guild_id=guild_id,
                        user_id=current.requested_by or None,
                        payload={
                            "track_id": current.track.id,
                            "title": current.track.title,
                            "requested_by": current.requested_by,
                            "is_tts": current.track.id == -2,
                        },
                    )
                )
        return started

    async def _advance_after_callback(self, guild_id: int, error: Exception | None) -> None:
        finished = self.state_for(guild_id).queue.current
        if finished is not None and self.event_bus is not None:
            await self.event_bus.publish(
                AgentEvent(
                    AgentEventKind.SONG_ENDED,
                    guild_id=guild_id,
                    user_id=finished.requested_by or None,
                    payload={
                        "track_id": finished.track.id,
                        "title": finished.track.title,
                        "requested_by": finished.requested_by,
                        "is_tts": finished.track.id == -2,
                        "had_error": error is not None,
                    },
                )
            )
        await super()._advance_after_callback(guild_id, error)
