from __future__ import annotations

import logging

import discord
from discord import app_commands
from discord.ext import commands

from discord_ai_assistant.ai.gemini import AssistantReply
from discord_ai_assistant.ai.persona import BASE_PERSONA_INSTRUCTION
from discord_ai_assistant.commands import VOICE_RECOGNITION_MODELS
from discord_ai_assistant.tool_effect_commands import ToolEffectAssistantCommands, VOICE_READ_STATE_PREFIX

LOGGER = logging.getLogger(__name__)

# These commands remain implemented on AssistantCommands for compatibility and reuse,
# but are removed from the top-level Discord command tree before sync. Their grouped
# wrappers below become the public Slash Command UI.
LEGACY_GROUPED_TOP_LEVEL_COMMANDS = {
    "join",
    "leave",
    "next",
    "now",
    "next_song",
    "previous",
    "repeat_one",
    "shuffle",
    "autoplay",
    "volume",
    "stop",
    "speak",
    "voice_chat_read",
    "voice_recognition_status",
    "voice_recognition_start",
    "voice_recognition_stop",
    "voice_recognition_tuning",
    "library_add",
    "library_search",
    "library_list",
    "library_delete",
    "playlist_create",
    "playlist_add",
    "playlist_list",
    "playlist_view",
    "playlist_play",
    "persona_auto",
    "comfort_auto",
    "persona_dnd",
    "persona_dnd_clear",
    "memory_add",
    "memory_list",
    "memory_delete",
    "memory_clear",
    "memory_passive",
    "youtube",
}


def remove_grouped_legacy_commands(tree: discord.app_commands.CommandTree) -> None:
    """Keep the Discord `/` menu compact by removing migrated top-level commands."""
    for name in LEGACY_GROUPED_TOP_LEVEL_COMMANDS:
        tree.remove_command(name)


class GroupedSlashCommands(commands.Cog):
    """Discord-native command groups for advanced or infrequent controls."""

    meeting = app_commands.Group(name="meeting", description="遊戲會議紀錄、逐字稿、OCR 與摘要輸出")
    comms = app_commands.Group(name="comms", description="Discord 語音通訊辨識與對話記錄")
    voice = app_commands.Group(name="voice", description="墨雪語音回答、朗讀與語音頻道控制")
    music = app_commands.Group(name="music", description="進階播放模式、插播與音量控制")
    library = app_commands.Group(name="library", description="本地音樂庫管理")
    playlist = app_commands.Group(name="playlist", description="播放清單管理")
    persona = app_commands.Group(name="persona", description="墨雪主動聊天、關懷與勿擾設定")
    memory = app_commands.Group(name="memory", description="墨雪的使用者記憶管理")

    def __init__(self, core: ToolEffectAssistantCommands) -> None:
        self.core = core
        self.bot = core.bot

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.guild:
            return True
        if interaction.response.is_done():
            await interaction.followup.send("這些指令只能在伺服器內使用。", ephemeral=True)
        else:
            await interaction.response.send_message("這些指令只能在伺服器內使用。", ephemeral=True)
        return False

    async def _legacy(self, name: str, interaction: discord.Interaction, *args: object) -> None:
        command = getattr(type(self.core), name, None)
        callback = getattr(command, "callback", None)
        if callback is None:
            raise RuntimeError(f"Legacy command callback not found: {name}")
        await callback(self.core, interaction, *args)

    async def _meeting_execute(
        self,
        interaction: discord.Interaction,
        action: str,
        arguments: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if not interaction.guild or not isinstance(interaction.user, discord.Member):
            return {"message": "這個指令只能在伺服器內使用。"}
        await self.core.ai.router.refresh_external_tools()
        return await self.core.ai.router.execute(
            f"x_game_meeting_recorder_{action}",
            arguments or {},
            self.core._tool_context(interaction.guild.id, interaction.user),
        )

    @meeting.command(name="status", description="查看目前會議紀錄工具狀態")
    async def meeting_status(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)
        result = await self._meeting_execute(interaction, "status")
        await interaction.edit_original_response(content=self.core.format_meeting_result("status", result)[:2000])

    @meeting.command(name="start", description="開始遊戲會議紀錄")
    @app_commands.describe(name="選填，本次會議或遊戲 Session 名稱")
    async def meeting_start(self, interaction: discord.Interaction, name: str | None = None) -> None:
        await interaction.response.defer(thinking=True)
        arguments: dict[str, object] = {"name": name} if name and name.strip() else {}
        result = await self._meeting_execute(interaction, "start_session", arguments)
        await interaction.edit_original_response(content=str(result.get("message") or "操作完成。")[:2000])

    @meeting.command(name="stop", description="停止紀錄、整理逐字稿並發布會議重點")
    async def meeting_stop(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True)
        result = await self._meeting_execute(interaction, "stop_session")
        if not isinstance(result.get("transcript"), str):
            await interaction.edit_original_response(content=str(result.get("message") or "操作完成。")[:2000])
            return
        effects = self.core.pop_tool_effects(result)
        summary = await self.core.summarize_meeting_result(interaction.guild_id or 0, result)
        publication = await self.core.apply_tool_effects(
            interaction.guild_id or 0,
            interaction.channel_id or 0,
            AssistantReply(text=summary, used_tools=True, effects=effects),
        )
        await interaction.edit_original_response(content=(publication or summary)[:2000])

    @meeting.command(name="transcript", description="查看目前或最近一次會議逐字稿")
    @app_commands.describe(limit="最多回傳幾筆事件，1 至 500")
    async def meeting_transcript(
        self, interaction: discord.Interaction, limit: app_commands.Range[int, 1, 500] = 200
    ) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)
        result = await self._meeting_execute(interaction, "get_transcript", {"limit": limit})
        await interaction.edit_original_response(
            content=self.core.format_meeting_result("get_transcript", result)[:2000]
        )

    @meeting.command(name="devices", description="列出 Windows 輸出與 loopback 音訊裝置")
    async def meeting_devices(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)
        result = await self._meeting_execute(interaction, "list_audio_devices")
        await interaction.edit_original_response(
            content=self.core.format_meeting_result("list_audio_devices", result)[:2000]
        )

    @meeting.command(name="output", description="設定會議摘要要發布到哪個 Discord 頻道")
    @app_commands.describe(channel="會議摘要輸出頻道")
    async def meeting_output(self, interaction: discord.Interaction, channel: discord.TextChannel) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)
        result = await self._meeting_execute(
            interaction, "configure_output", {"summary_channel_id": channel.id}
        )
        await interaction.edit_original_response(
            content=f"{result.get('message', '輸出設定已更新。')}\n摘要頻道：{channel.mention}"
        )

    @meeting.command(name="ocr", description="執行一次聊天室 ROI OCR 校正測試")
    async def meeting_ocr(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)
        result = await self._meeting_execute(interaction, "test_chat_capture")
        lines = result.get("lines") if isinstance(result.get("lines"), list) else []
        preview = "\n".join(str(item) for item in lines[:15]) or "沒有辨識到文字。"
        await interaction.edit_original_response(
            content=f"{result.get('message', 'OCR 測試完成。')}\n\n{preview}"[:2000]
        )

    @comms.command(name="status", description="查看 Discord 語音通訊辨識狀態")
    async def comms_status(self, interaction: discord.Interaction) -> None:
        await self._legacy("voice_recognition_status", interaction)

    @comms.command(name="start", description="開始 Discord 語音通訊辨識")
    async def comms_start(self, interaction: discord.Interaction) -> None:
        await self._legacy("voice_recognition_start", interaction)

    @comms.command(name="stop", description="停止 Discord 語音通訊辨識")
    async def comms_stop(self, interaction: discord.Interaction) -> None:
        await self._legacy("voice_recognition_stop", interaction)

    @comms.command(name="tuning", description="調整語音辨識模型與切段參數")
    @app_commands.describe(
        model="辨識模型；越大通常越準但越慢",
        beam_size="候選解碼數",
        silence_seconds="停頓多久後送出辨識",
        max_segment_seconds="單人連續說話最長切段秒數",
    )
    @app_commands.choices(model=[app_commands.Choice(name=name, value=name) for name in VOICE_RECOGNITION_MODELS])
    async def comms_tuning(
        self,
        interaction: discord.Interaction,
        model: str | None = None,
        beam_size: app_commands.Range[int, 1, 10] | None = None,
        silence_seconds: app_commands.Range[float, 0.5, 10.0] | None = None,
        max_segment_seconds: app_commands.Range[float, 2.0, 60.0] | None = None,
    ) -> None:
        await self._legacy(
            "voice_recognition_tuning",
            interaction,
            model,
            beam_size,
            silence_seconds,
            max_segment_seconds,
        )

    @voice.command(name="ask", description="讓墨雪回答內容並在你所在的語音頻道說出答案")
    @app_commands.describe(prompt="想問墨雪的內容")
    async def voice_ask(self, interaction: discord.Interaction, prompt: str) -> None:
        member = interaction.user if isinstance(interaction.user, discord.Member) else None
        channel = self.core._voice_channel(interaction)
        if not member or not interaction.guild or not channel:
            await interaction.response.send_message("請先加入語音頻道。", ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        if not self.core.ai.enabled:
            await interaction.edit_original_response(content="目前沒有設定 Gemini API。")
            return
        try:
            answer = await self.core.ai.social_reply(
                self.core._with_user_memory(prompt, interaction.guild.id, member.id),
                persona_instruction=(
                    f"{BASE_PERSONA_INSTRUCTION}\n"
                    "這是語音回答。請用自然、口語的繁體中文回答，盡量控制在 350 字內。\n"
                    f"{self.core.workload.instruction_for(interaction.guild.id, record_request=True)}"
                ),
            )
            audio_path = await self.core.speech.synthesize(answer.strip()[:400])
            await self.core.music.connect(interaction.guild.id, channel)
            await self.core.music.enqueue_tts(interaction.guild.id, audio_path, member.id)
        except Exception:
            LOGGER.exception("Voice ask failed in guild %s", interaction.guild.id)
            await interaction.edit_original_response(content="Voice ask 執行失敗，請查看 Bot log。")
            return
        await interaction.edit_original_response(content=answer[:2000])

    @voice.command(name="say", description="讓墨雪直接在你的語音頻道朗讀文字")
    @app_commands.describe(text="要朗讀的內容，最多 400 字")
    async def voice_say(self, interaction: discord.Interaction, text: str) -> None:
        await self._legacy("speak", interaction, text)

    @voice.command(name="read", description="持續朗讀指定文字頻道之後出現的新訊息")
    @app_commands.describe(channel="要監聽並朗讀的新訊息來源頻道")
    async def voice_read(self, interaction: discord.Interaction, channel: discord.TextChannel) -> None:
        if not self.core._is_dj(interaction):
            await interaction.response.send_message("持續朗讀只有 DJ 或管理員可以開啟。", ephemeral=True)
            return
        voice_channel = self.core._voice_channel(interaction)
        if not voice_channel:
            await interaction.response.send_message("請先加入要讓墨雪說話的語音頻道。", ephemeral=True)
            return
        await self.core.music.connect(interaction.guild_id or 0, voice_channel)
        self.core.database.set_state(f"{VOICE_READ_STATE_PREFIX}{interaction.guild_id or 0}", str(channel.id))
        await interaction.response.send_message(
            f"已開始朗讀 {channel.mention} 的新訊息；聲音會播到 {voice_channel.mention}。",
            ephemeral=True,
        )

    @voice.command(name="stopread", description="停止持續朗讀文字頻道")
    async def voice_stopread(self, interaction: discord.Interaction) -> None:
        if not self.core._is_dj(interaction):
            await interaction.response.send_message("只有 DJ 或管理員可以停止持續朗讀。", ephemeral=True)
            return
        self.core.database.set_state(f"{VOICE_READ_STATE_PREFIX}{interaction.guild_id or 0}", "0")
        await interaction.response.send_message("已停止持續朗讀文字頻道。", ephemeral=True)

    @voice.command(name="join", description="讓墨雪加入你所在的語音頻道")
    async def voice_join(self, interaction: discord.Interaction) -> None:
        await self._legacy("join", interaction)

    @voice.command(name="leave", description="讓墨雪離開語音頻道並清空佇列")
    async def voice_leave(self, interaction: discord.Interaction) -> None:
        await self._legacy("leave", interaction)

    @music.command(name="next", description="把歌曲插到播放佇列的下一首")
    async def music_next(self, interaction: discord.Interaction, query: str) -> None:
        await self._legacy("next", interaction, query)

    @music.command(name="now", description="立刻插播歌曲（DJ）")
    async def music_now(self, interaction: discord.Interaction, query: str) -> None:
        await self._legacy("now", interaction, query)

    @music.command(name="previous", description="回到上一首歌曲")
    async def music_previous(self, interaction: discord.Interaction) -> None:
        await self._legacy("previous", interaction)

    @music.command(name="repeat", description="開啟或關閉單曲循環（DJ）")
    async def music_repeat(self, interaction: discord.Interaction, enabled: bool) -> None:
        await self._legacy("repeat_one", interaction, enabled)

    @music.command(name="shuffle", description="開啟或關閉隨機播放（DJ）")
    async def music_shuffle(self, interaction: discord.Interaction, enabled: bool) -> None:
        await self._legacy("shuffle", interaction, enabled)

    @music.command(name="autoplay", description="開啟或關閉本地音樂自動推薦（DJ）")
    async def music_autoplay(self, interaction: discord.Interaction, enabled: bool) -> None:
        await self._legacy("autoplay", interaction, enabled)

    @music.command(name="volume", description="調整 Bot 音量（DJ）")
    async def music_volume(
        self, interaction: discord.Interaction, percent: app_commands.Range[int, 0, 200]
    ) -> None:
        await self._legacy("volume", interaction, percent)

    @music.command(name="stop", description="停止播放並清空佇列（DJ）")
    async def music_stop(self, interaction: discord.Interaction) -> None:
        await self._legacy("stop", interaction)

    @music.command(name="youtube", description="建立 YouTube 搜尋連結，不播放或下載")
    async def music_youtube(self, interaction: discord.Interaction, query: str) -> None:
        await self._legacy("youtube", interaction, query)

    @library.command(name="add", description="上傳音樂到本地音樂庫")
    async def library_add(self, interaction: discord.Interaction, file: discord.Attachment) -> None:
        await self._legacy("library_add", interaction, file)

    @library.command(name="search", description="搜尋本地音樂庫")
    async def library_search(self, interaction: discord.Interaction, query: str) -> None:
        await self._legacy("library_search", interaction, query)

    @library.command(name="list", description="列出本地音樂庫")
    @app_commands.choices(
        order=[
            app_commands.Choice(name="最新上傳", value="newest"),
            app_commands.Choice(name="最早上傳", value="oldest"),
            app_commands.Choice(name="歌曲名稱", value="title"),
            app_commands.Choice(name="歌曲 ID", value="id"),
        ]
    )
    async def library_list(
        self,
        interaction: discord.Interaction,
        page: app_commands.Range[int, 1, 1000] = 1,
        order: str = "newest",
    ) -> None:
        await self._legacy("library_list", interaction, page, order)

    @library.command(name="delete", description="刪除自己上傳的歌曲，DJ 可刪除其他歌曲")
    async def library_delete(self, interaction: discord.Interaction, track_id: int) -> None:
        await self._legacy("library_delete", interaction, track_id)

    @playlist.command(name="create", description="建立播放清單，可直接指定歌曲 ID 順序")
    async def playlist_create(
        self, interaction: discord.Interaction, name: str, track_ids: str = ""
    ) -> None:
        await self._legacy("playlist_create", interaction, name, track_ids)

    @playlist.command(name="add", description="把音樂庫歌曲加入播放清單")
    async def playlist_add(self, interaction: discord.Interaction, name: str, track_id: int) -> None:
        await self._legacy("playlist_add", interaction, name, track_id)

    @playlist.command(name="list", description="列出此伺服器的播放清單")
    async def playlist_list(self, interaction: discord.Interaction) -> None:
        await self._legacy("playlist_list", interaction)

    @playlist.command(name="view", description="查看播放清單內的歌曲順序")
    async def playlist_view(
        self, interaction: discord.Interaction, name: str, page: app_commands.Range[int, 1, 1000] = 1
    ) -> None:
        await self._legacy("playlist_view", interaction, name, page)

    @playlist.command(name="play", description="將播放清單加入播放佇列")
    async def playlist_play(self, interaction: discord.Interaction, name: str) -> None:
        await self._legacy("playlist_play", interaction, name)

    @persona.command(name="auto", description="開啟或關閉墨雪在貓窩的自發對話（DJ）")
    async def persona_auto(self, interaction: discord.Interaction, enabled: bool) -> None:
        await self._legacy("persona_auto", interaction, enabled)

    @persona.command(name="comfort", description="開啟或關閉低落情緒關懷（DJ）")
    async def persona_comfort(self, interaction: discord.Interaction, enabled: bool) -> None:
        await self._legacy("comfort_auto", interaction, enabled)

    @persona.command(name="dnd", description="設定墨雪每日勿擾時段（DJ）")
    async def persona_dnd(self, interaction: discord.Interaction, start: str, end: str) -> None:
        await self._legacy("persona_dnd", interaction, start, end)

    @persona.command(name="dnd_clear", description="關閉墨雪每日勿擾時段（DJ）")
    async def persona_dnd_clear(self, interaction: discord.Interaction) -> None:
        await self._legacy("persona_dnd_clear", interaction)

    @memory.command(name="add", description="請墨雪記住你的偏好、習慣或提醒")
    async def memory_add(
        self, interaction: discord.Interaction, category: str, content: str
    ) -> None:
        await self._legacy("memory_add", interaction, category, content)

    @memory.command(name="list", description="查看墨雪記住的你的資料")
    async def memory_list(self, interaction: discord.Interaction) -> None:
        await self._legacy("memory_list", interaction)

    @memory.command(name="delete", description="刪除墨雪記住的一項資料")
    async def memory_delete(self, interaction: discord.Interaction, memory_id: int) -> None:
        await self._legacy("memory_delete", interaction, memory_id)

    @memory.command(name="clear", description="清空墨雪記住的所有你的資料")
    async def memory_clear(self, interaction: discord.Interaction) -> None:
        await self._legacy("memory_clear", interaction)

    @memory.command(name="passive", description="開啟或關閉被動記憶整理")
    async def memory_passive(self, interaction: discord.Interaction, enabled: bool) -> None:
        await self._legacy("memory_passive", interaction, enabled)
