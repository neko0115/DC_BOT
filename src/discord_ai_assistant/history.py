from __future__ import annotations

from collections import defaultdict, deque

import discord


class RecentMessageHistory:
    """In-memory, bounded context only. No conversation text is persisted locally."""

    def __init__(self, per_channel_limit: int = 20) -> None:
        self._messages: dict[tuple[int, int], deque[str]] = defaultdict(lambda: deque(maxlen=per_channel_limit))

    def record(self, message: discord.Message) -> None:
        if not message.guild or message.author.bot:
            return
        text = message.clean_content.strip().replace("\n", " ")
        if text:
            self._messages[(message.guild.id, message.channel.id)].append(
                f"{message.author.display_name}: {text[:400]}"
            )

    def format_for(self, guild_id: int, channel_id: int, limit: int = 10) -> str:
        messages = list(self._messages[(guild_id, channel_id)])[-limit:]
        if not messages:
            return "（沒有可用的近期對話）"
        return "\n".join(messages)
