from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import quote_plus

import discord

from discord_ai_assistant.music.library import LibraryService
from discord_ai_assistant.music.player import MusicManager
from discord_ai_assistant.storage.database import Database
from discord_ai_assistant.tool_gateway.client import ToolGatewayClient, ToolGatewayUnavailable

LOGGER = logging.getLogger(__name__)
MAX_EXTERNAL_ARTIFACTS = 4
MAX_ARTIFACT_RELATIVE_PATH = 512
MAX_ARTIFACT_FILENAME = 180

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

    def external_declarations_for(self, prompt: str, context: ToolContext) -> list[dict[str, object]]:
        if self.external_client is None:
            return []
        return self.external_client.declarations_for(prompt, is_dj=context.is_dj)

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
                result = await self.external_client.invoke(name, arguments, context.external_payload())
                return self._attach_external_effects(result)
            except PermissionError:
                return {"message": "此工具操作僅限 DJ 或管理員。"}
            except ToolGatewayUnavailable as error:
                return {"message": f"外接工具目前無法使用：{error}"}
        return {"message": "此操作不在允許清單內。"}

    @staticmethod
    def _artifact_effect(item: object) -> dict[str, object] | None:
        if not isinstance(item, dict):
            return None
        relative_path = item.get("relative_path")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or len(relative_path) > MAX_ARTIFACT_RELATIVE_PATH
            or "\x00" in relative_path
        ):
            return None
        path = PurePosixPath(relative_path.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or not path.parts:
            return None
        mime_type = item.get("mime_type")
        if not isinstance(mime_type, str) or not mime_type or len(mime_type) > 100:
            return None
        filename = item.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            filename = path.name
        filename = PurePosixPath(filename.replace("\\", "/")).name[:MAX_ARTIFACT_FILENAME]
        if not filename:
            return None
        delete_after_send = item.get("delete_after_send", True)
        if not isinstance(delete_after_send, bool):
            delete_after_send = True
        return {
            "type": "attach_artifact",
            "relative_path": str(path),
            "filename": filename,
            "mime_type": mime_type,
            "delete_after_send": delete_after_send,
        }

    @classmethod
    def _attach_external_effects(cls, result: dict[str, object]) -> dict[str, object]:
        """Translate whitelisted plugin metadata into core-owned effects."""
        result = dict(result)
        # External plugins may request only the metadata translations below. They
        # cannot inject arbitrary internal effects directly.
        result.pop("_moxue_effects", None)
        effects: list[dict[str, object]] = []

        raw_artifacts = result.pop("_moxue_artifacts", None)
        if isinstance(raw_artifacts, list):
            for item in raw_artifacts[:MAX_EXTERNAL_ARTIFACTS]:
                effect = cls._artifact_effect(item)
                if effect is not None:
                    effects.append(effect)

        channel_id = result.get("summary_channel_id")
        summary_instruction = result.get("summary_instruction")
        if (
            isinstance(channel_id, int)
            and channel_id > 0
            and isinstance(summary_instruction, str)
            and summary_instruction.strip()
        ):
            effects.append(
                {
                    "type": "publish_final_reply",
                    "channel_id": channel_id,
                    "suppress_origin": True,
                }
            )
        if effects:
            result["_moxue_effects"] = effects
        return result

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
