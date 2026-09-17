from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import quote_plus

import discord

from discord_ai_assistant.music.library import LibraryService
from discord_ai_assistant.music.player import MusicManager
from discord_ai_assistant.storage.database import Database
from discord_ai_assistant.tool_gateway.client import ToolGatewayClient, ToolGatewayUnavailable

LOGGER = logging.getLogger(__name__)

TOOL_DECLARATIONS = [
    {
        "type": "function",
        "name": "queue_library_track",
        "description": "Queue a matching track from this server's local music library.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Track title or artist search query."},
                "position": {"type": "string", "enum": ["queue", "next", "now"]},
            },
            "required": ["query", "position"],
        },
    },
    {
        "type": "function",
        "name": "show_queue",
        "description": "Show the current music queue for this server.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "youtube_search_link",
        "description": "Create a YouTube search link. Never use this to stream or download audio.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Song or artist to search."}},
            "required": ["query"],
        },
    },
]


@dataclass(frozen=True, slots=True)
class ToolContext:
    guild_id: int
    user_id: int
    is_dj: bool
    voice_channel: discord.VoiceChannel | discord.StageChannel | None

    def external_payload(self) -> dict[str, object]:
        channel = self.voice_channel
        return {
            "guild_id": self.guild_id,
            "user_id": self.user_id,
            "is_dj": self.is_dj,
            "voice_channel_id": channel.id if channel else None,
            "voice_channel_name": channel.name if channel else None,
        }


class ToolRouter:
    """Executes built-in actions and optional repository-level external tools."""

    def __init__(
        self,
        library: LibraryService,
        music: MusicManager,
        database: Database,
        external_client: ToolGatewayClient | None = None,
    ) -> None:
        self.library = library
        self.music = music
        self.database = database
        self.external_client = external_client

    async def refresh_external_tools(self) -> None:
        if self.external_client is None:
            return
        try:
            await self.external_client.refresh()
        except ToolGatewayUnavailable as error:
            LOGGER.warning("External tool gateway is unavailable: %s", error)

    def external_declarations_for(self, prompt: str) -> list[dict[str, object]]:
        if self.external_client is None:
            return []
        return self.external_client.declarations_for(prompt)

    async def execute(self, name: str, arguments: dict[str, object], context: ToolContext) -> dict[str, object]:
        if name == "queue_library_track":
            return await self._queue_library_track(arguments, context)
        if name == "show_queue":
            return self._show_queue(context)
        if name == "youtube_search_link":
            query = self._string(arguments, "query")
            return {"message": f"YouTube 搜尋連結：https://www.youtube.com/results?search_query={quote_plus(query)}"}
        if self.external_client is not None and self.external_client.handles(name):
            try:
                return await self.external_client.invoke(name, arguments, context.external_payload())
            except ToolGatewayUnavailable as error:
                return {"message": f"外接工具目前無法使用：{error}"}
        return {"message": "此操作不在允許清單內。"}

    async def _queue_library_track(
        self, arguments: dict[str, object], context: ToolContext
    ) -> dict[str, object]:
        query = self._string(arguments, "query")
        position = self._string(arguments, "position")
        if position not in {"queue", "next", "now"}:
            return {"message": "無效的播放位置。"}
        if position == "now" and not context.is_dj:
            return {"message": "立刻插播僅限 DJ 或管理員。"}
        if not context.voice_channel:
            return {"message": "使用者不在語音頻道，無法開始播放。"}

        matches = self.library.search(query)
        if not matches:
            return {"message": f"本地音樂庫找不到：{query}"}
        track = matches[0]
        await self.music.connect(context.guild_id, context.voice_channel)
        if position == "now":
            await self.music.play_now(context.guild_id, track, context.user_id)
        else:
            await self.music.enqueue(context.guild_id, track, context.user_id, next_up=position == "next")
        return {"message": f"已加入：{track.title}"}

    def _show_queue(self, context: ToolContext) -> dict[str, object]:
        current, upcoming = self.music.queue_view(context.guild_id)
        lines = [f"目前：{current.track.title}" if current else "目前沒有播放中的歌曲。"]
        lines.extend(f"{index}. {item.track.title}" for index, item in enumerate(upcoming, start=1))
        return {"message": "\n".join(lines)}

    @staticmethod
    def _string(arguments: dict[str, object], key: str) -> str:
        value = arguments.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"工具參數 {key} 不正確。")
        return value.strip()
