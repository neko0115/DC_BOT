from __future__ import annotations

import time
from collections import defaultdict, deque

import discord
from discord.ext import commands, tasks

from discord_ai_assistant.agent.events import AgentEvent, AgentEventBus, AgentEventKind
from discord_ai_assistant.config import Settings
from discord_ai_assistant.storage.agent_database import AgentDatabase


class AgentEventBridge(commands.Cog):
    """Translate Discord lifecycle events into bounded, passive Agent Core events.

    The bridge never calls Gemini and never sends a Discord message. It only feeds the
    coordinator with structured observations, so adding it cannot make 墨雪 noisier.
    """

    def __init__(
        self,
        bot: commands.Bot,
        settings: Settings,
        bus: AgentEventBus,
        database: AgentDatabase,
    ) -> None:
        self.bot = bot
        self.settings = settings
        self.bus = bus
        self.database = database
        self._recent: dict[tuple[int, int], deque[str]] = defaultdict(lambda: deque(maxlen=20))
        self._last_activity: dict[tuple[int, int], float] = {}
        self._last_idle_event: dict[tuple[int, int], float] = {}

    async def cog_load(self) -> None:
        self._idle_events.start()

    def cog_unload(self) -> None:
        self._idle_events.cancel()

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or not message.guild:
            return
        key = (message.guild.id, message.channel.id)
        content = " ".join(message.clean_content.split())[:1000]
        if content:
            self._recent[key].append(f"{message.author.display_name}: {content}")
        self._last_activity[key] = time.monotonic()

        memories = self.database.agent_memory_lines(message.guild.id, message.author.id, limit=8)
        relationship = self.database.familiarity_label(message.guild.id, message.author.id)
        mentioned = bool(
            self.bot.user
            and (self.bot.user.id in message.raw_mentions or any(user.id == self.bot.user.id for user in message.mentions))
        )
        await self.bus.publish(
            AgentEvent(
                AgentEventKind.MESSAGE_RECEIVED,
                guild_id=message.guild.id,
                channel_id=message.channel.id,
                user_id=message.author.id,
                payload={
                    "content": content,
                    "author_name": message.author.display_name,
                    "mentioned_bot": mentioned,
                    "is_reply": bool(message.reference),
                    "attachment_count": len(message.attachments),
                    "recent_messages": list(self._recent[key]),
                    "memories": memories,
                    "relationship": relationship,
                },
            )
        )

    @commands.Cog.listener()
    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        if member.bot or before.channel == after.channel:
            return
        if before.channel is not None:
            await self.bus.publish(
                AgentEvent(
                    AgentEventKind.MEMBER_LEFT_VOICE,
                    guild_id=member.guild.id,
                    channel_id=before.channel.id,
                    user_id=member.id,
                    payload={"display_name": member.display_name, "channel_name": before.channel.name},
                )
            )
        if after.channel is not None:
            await self.bus.publish(
                AgentEvent(
                    AgentEventKind.MEMBER_JOINED_VOICE,
                    guild_id=member.guild.id,
                    channel_id=after.channel.id,
                    user_id=member.id,
                    payload={"display_name": member.display_name, "channel_name": after.channel.name},
                )
            )

    @tasks.loop(minutes=1)
    async def _idle_events(self) -> None:
        idle_seconds = max(1, self.settings.topic_idle_minutes) * 60
        min_interval = max(1, self.settings.topic_min_interval_minutes) * 60
        now = time.monotonic()
        for key, last_activity in tuple(self._last_activity.items()):
            if now - last_activity < idle_seconds:
                continue
            if now - self._last_idle_event.get(key, 0.0) < min_interval:
                continue
            guild_id, channel_id = key
            self._last_idle_event[key] = now
            await self.bus.publish(
                AgentEvent(
                    AgentEventKind.CONVERSATION_IDLE,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    payload={"recent_messages": list(self._recent.get(key, ()))},
                )
            )

    @_idle_events.before_loop
    async def _before_idle_events(self) -> None:
        await self.bot.wait_until_ready()
