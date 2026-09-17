from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import discord
from discord import app_commands
from discord.ext import commands, tasks

from discord_ai_assistant.music.lyrics import LyricsLookupError, LyricsProvider, LyricsResult, current_line_index
from discord_ai_assistant.music.player import MusicManager
from discord_ai_assistant.storage.database import Database

LOGGER = logging.getLogger(__name__)
AUTO_LYRICS_COMMAND_PATHS = {
    ("play",),
    ("next",),
    ("now",),
    ("music", "next"),
    ("music", "now"),
}
LYRICS_CHANNEL_STATE_PREFIX = "lyrics_channel:"
LYRICS_MODE_STATE_PREFIX = "lyrics_mode:"
LYRICS_OFFSET_STATE_PREFIX = "lyrics_offset:"
LYRICS_MODES = {"panel", "bottom"}
DEFAULT_LYRICS_OFFSET_SECONDS = 1.0
MIN_LYRICS_OFFSET_SECONDS = -5.0
MAX_LYRICS_OFFSET_SECONDS = 10.0


@dataclass(slots=True)
class LiveLyricsSession:
    channel_id: int
    mode: str = "panel"
    offset_seconds: float = DEFAULT_LYRICS_OFFSET_SECONDS
    message: discord.Message | None = None
    track_key: str | None = None
    lyrics: LyricsResult | None = None
    elapsed_seconds: float = 0.0
    last_tick: float = 0.0
    render_key: tuple[object, ...] | None = None
    lookup_error: str | None = None
    needs_bump: bool = False


class LyricsCommands(commands.Cog):
    """Keeps one synchronized lyrics panel per guild and follows the current music track."""

    lyrics = app_commands.Group(name="lyrics", description="同步歌詞顯示與頻道設定")

    def __init__(self, bot: commands.Bot, music: MusicManager, database: Database) -> None:
        self.bot = bot
        self.music = music
        self.database = database
        self.provider = LyricsProvider()
        self._sessions: dict[int, LiveLyricsSession] = {}

    async def cog_load(self) -> None:
        self._lyrics_tick.start()

    def cog_unload(self) -> None:
        self._lyrics_tick.cancel()

    @commands.Cog.listener()
    async def on_interaction(self, interaction: discord.Interaction) -> None:
        if interaction.type is not discord.InteractionType.application_command:
            return
        data = interaction.data if isinstance(interaction.data, dict) else {}
        if _command_path(data) not in AUTO_LYRICS_COMMAND_PATHS:
            return
        if interaction.guild_id is None or interaction.channel_id is None:
            return
        await self._enable(interaction.guild_id, interaction.channel_id)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None or message.author.bot:
            return
        session = self._sessions.get(message.guild.id)
        if session is None or session.mode != "bottom":
            return
        if message.channel.id != session.channel_id or session.message is None:
            return
        if message.id == session.message.id:
            return
        session.needs_bump = True

    @app_commands.command(name="lyrics_live", description="開啟或關閉目前頻道的同步歌詞面板")
    @app_commands.describe(enabled="開啟後會自動跟隨目前與之後播放的歌曲")
    async def lyrics_live(self, interaction: discord.Interaction, enabled: bool = True) -> None:
        if interaction.guild_id is None or interaction.channel_id is None:
            await interaction.response.send_message("此功能只能在伺服器文字頻道使用。", ephemeral=True)
            return
        if enabled:
            await self._enable(interaction.guild_id, interaction.channel_id)
            target = self._sessions[interaction.guild_id].channel_id
            suffix = f" <#{target}>" if target != interaction.channel_id else "這個頻道"
            await interaction.response.send_message(
                f"已開啟同步歌詞；歌詞面板會顯示在 {suffix}。", ephemeral=True
            )
            return

        session = self._sessions.pop(interaction.guild_id, None)
        if session and session.message:
            try:
                await session.message.edit(content="同步歌詞已關閉。", embed=None)
            except discord.HTTPException:
                pass
        await interaction.response.send_message("已關閉同步歌詞。", ephemeral=True)

    @lyrics.command(name="channel", description="指定固定顯示同步歌詞的文字頻道")
    @app_commands.describe(channel="歌詞專用頻道；不填則使用目前頻道")
    async def lyrics_channel(
        self,
        interaction: discord.Interaction,
        channel: discord.TextChannel | None = None,
    ) -> None:
        if interaction.guild_id is None or interaction.channel_id is None:
            await interaction.response.send_message("此功能只能在伺服器內使用。", ephemeral=True)
            return
        target = channel or (
            interaction.channel if isinstance(interaction.channel, discord.TextChannel) else None
        )
        if target is None:
            await interaction.response.send_message("請指定一個文字頻道。", ephemeral=True)
            return
        self.database.set_state(self._channel_key(interaction.guild_id), str(target.id))
        await self._enable(interaction.guild_id, target.id)
        await interaction.response.send_message(
            f"之後同步歌詞固定顯示在 {target.mention}。", ephemeral=True
        )

    @lyrics.command(name="auto", description="取消固定歌詞頻道，改為跟隨播放指令所在頻道")
    async def lyrics_auto(self, interaction: discord.Interaction) -> None:
        if interaction.guild_id is None or interaction.channel_id is None:
            await interaction.response.send_message("此功能只能在伺服器內使用。", ephemeral=True)
            return
        self.database.set_state(self._channel_key(interaction.guild_id), "0")
        await self._enable(interaction.guild_id, interaction.channel_id, ignore_configured_channel=True)
        await interaction.response.send_message(
            "已取消固定歌詞頻道；之後會跟隨 `/play` 所在的文字頻道。",
            ephemeral=True,
        )

    @lyrics.command(name="mode", description="設定歌詞面板顯示方式")
    @app_commands.describe(mode="panel：固定編輯同一則；bottom：有人聊天後自動搬回最底")
    @app_commands.choices(
        mode=[
            app_commands.Choice(name="Panel－固定同一則訊息", value="panel"),
            app_commands.Choice(name="Bottom－保持在頻道最下面", value="bottom"),
        ]
    )
    async def lyrics_mode(self, interaction: discord.Interaction, mode: str) -> None:
        if interaction.guild_id is None:
            await interaction.response.send_message("此功能只能在伺服器內使用。", ephemeral=True)
            return
        if mode not in LYRICS_MODES:
            await interaction.response.send_message("不支援的歌詞顯示模式。", ephemeral=True)
            return
        self.database.set_state(self._mode_key(interaction.guild_id), mode)
        session = self._sessions.get(interaction.guild_id)
        if session is not None:
            session.mode = mode
            session.needs_bump = mode == "bottom"
        explanation = (
            "會編輯同一則歌詞訊息。"
            if mode == "panel"
            else "有人在歌詞頻道發訊息後，歌詞面板會自動搬回最下面。"
        )
        await interaction.response.send_message(
            f"歌詞顯示模式已設為 **{mode}**；{explanation}", ephemeral=True
        )

    @lyrics.command(name="offset", description="校正同步歌詞時間；正值會讓歌詞晚一點出現")
    @app_commands.describe(seconds="秒數，例如 1.5；可設定 -5.0 到 10.0 秒")
    async def lyrics_offset(self, interaction: discord.Interaction, seconds: float) -> None:
        if interaction.guild_id is None:
            await interaction.response.send_message("此功能只能在伺服器內使用。", ephemeral=True)
            return
        value = round(float(seconds), 2)
        if not MIN_LYRICS_OFFSET_SECONDS <= value <= MAX_LYRICS_OFFSET_SECONDS:
            await interaction.response.send_message(
                f"歌詞延遲需介於 {MIN_LYRICS_OFFSET_SECONDS:.1f} 到 {MAX_LYRICS_OFFSET_SECONDS:.1f} 秒。",
                ephemeral=True,
            )
            return
        self.database.set_state(self._offset_key(interaction.guild_id), str(value))
        session = self._sessions.get(interaction.guild_id)
        if session is not None:
            session.offset_seconds = value
            session.render_key = None
        direction = "延後" if value > 0 else "提前" if value < 0 else "不偏移"
        await interaction.response.send_message(
            f"歌詞同步已設為 **{value:+.2f} 秒**（{direction}）。",
            ephemeral=True,
        )

    @lyrics.command(name="status", description="查看目前歌詞顯示設定")
    async def lyrics_status(self, interaction: discord.Interaction) -> None:
        if interaction.guild_id is None:
            await interaction.response.send_message("此功能只能在伺服器內使用。", ephemeral=True)
            return
        configured_channel = self._configured_channel(interaction.guild_id)
        mode = self._configured_mode(interaction.guild_id)
        offset = self._configured_offset(interaction.guild_id)
        channel_text = f"<#{configured_channel}>" if configured_channel else "跟隨 `/play` 所在頻道"
        await interaction.response.send_message(
            f"歌詞頻道：{channel_text}\n顯示模式：**{mode}**\n同步偏移：**{offset:+.2f} 秒**",
            ephemeral=True,
        )

    async def _enable(
        self,
        guild_id: int,
        fallback_channel_id: int,
        *,
        ignore_configured_channel: bool = False,
    ) -> None:
        configured = None if ignore_configured_channel else self._configured_channel(guild_id)
        channel_id = configured or fallback_channel_id
        mode = self._configured_mode(guild_id)
        offset = self._configured_offset(guild_id)
        session = self._sessions.get(guild_id)
        if session is None:
            self._sessions[guild_id] = LiveLyricsSession(
                channel_id=channel_id,
                mode=mode,
                offset_seconds=offset,
                last_tick=time.monotonic(),
            )
            return
        if session.channel_id != channel_id:
            old_message = session.message
            session.channel_id = channel_id
            session.message = None
            session.render_key = None
            if old_message is not None:
                try:
                    await old_message.delete()
                except discord.HTTPException:
                    pass
        session.mode = mode
        session.offset_seconds = offset
        session.last_tick = time.monotonic()
        if mode == "bottom":
            session.needs_bump = True

    @tasks.loop(seconds=0.75)
    async def _lyrics_tick(self) -> None:
        now = time.monotonic()
        for guild_id, session in list(self._sessions.items()):
            try:
                await self._tick_session(guild_id, session, now)
            except Exception:
                LOGGER.exception("Live lyrics update failed in guild %s", guild_id)

    @_lyrics_tick.before_loop
    async def _before_lyrics_tick(self) -> None:
        await self.bot.wait_until_ready()

    async def _tick_session(self, guild_id: int, session: LiveLyricsSession, now: float) -> None:
        current, _ = self.music.queue_view(guild_id)
        state = self.music.state_for(guild_id)
        voice = state.voice

        if current is None:
            session.last_tick = now
            if session.track_key is not None:
                session.track_key = None
                session.lyrics = None
                session.elapsed_seconds = 0.0
                session.lookup_error = None
                await self._render(session, None, status="idle", force_bump=session.mode == "bottom")
            return

        track = current.track
        if track.id == -2:
            session.last_tick = now
            return

        track_key = f"{track.id}:{track.title}:{track.original_name}:{track.audio_path or ''}"
        if track_key != session.track_key:
            session.track_key = track_key
            session.elapsed_seconds = 0.0
            session.render_key = None
            session.lookup_error = None
            lookup_started = time.monotonic()
            was_playing = bool(voice and voice.is_playing() and not voice.is_paused())
            try:
                session.lyrics = await self.provider.fetch_for_track(track)
            except LyricsLookupError as error:
                session.lyrics = None
                session.lookup_error = str(error)
            lookup_finished = time.monotonic()
            if was_playing and voice and voice.is_playing() and not voice.is_paused():
                session.elapsed_seconds += max(0.0, lookup_finished - lookup_started)
            session.last_tick = lookup_finished
            line_index = self._current_line(session)
            status = "paused" if voice and voice.is_paused() else "playing"
            await self._render(
                session,
                track.title,
                status=status,
                line_index=line_index,
                force_bump=session.mode == "bottom",
            )
            return

        delta = max(0.0, now - session.last_tick)
        session.last_tick = now
        if voice and voice.is_playing() and not voice.is_paused():
            session.elapsed_seconds += delta

        line_index = self._current_line(session)
        status = "paused" if voice and voice.is_paused() else "playing"
        render_key = self._render_key(session, line_index, status)
        if render_key != session.render_key or session.needs_bump:
            await self._render(
                session,
                track.title,
                status=status,
                line_index=line_index,
                force_bump=session.needs_bump,
            )

    @staticmethod
    def _lyric_position(session: LiveLyricsSession) -> float:
        # Positive offset means the lyrics should appear later than the raw LRC timestamp.
        return max(0.0, session.elapsed_seconds - session.offset_seconds)

    def _current_line(self, session: LiveLyricsSession) -> int | None:
        lyrics = session.lyrics
        if lyrics is None:
            return None
        return current_line_index(lyrics.lines, self._lyric_position(session))

    def _render_key(
        self,
        session: LiveLyricsSession,
        line_index: int | None,
        status: str,
    ) -> tuple[object, ...]:
        lyrics = session.lyrics
        time_bucket = int(session.elapsed_seconds) if status == "playing" else None
        return (
            session.track_key,
            line_index,
            status,
            lyrics.source if lyrics else None,
            time_bucket,
            round(session.offset_seconds, 2),
            session.lookup_error,
        )

    async def _render(
        self,
        session: LiveLyricsSession,
        track_title: str | None,
        *,
        status: str,
        line_index: int | None = None,
        force_bump: bool = False,
    ) -> None:
        channel = self.bot.get_channel(session.channel_id)
        if not isinstance(channel, discord.abc.Messageable):
            return

        embed = discord.Embed(title="🎵 同步歌詞")
        if status == "idle" or track_title is None:
            embed.description = "目前沒有播放歌曲。"
            render_key = ("idle",)
        elif session.lookup_error:
            embed.title = f"🎵 {track_title}"
            embed.description = "歌詞服務暫時無法連線，下一首歌會再自動嘗試。"
            render_key = (session.track_key, "error", session.lookup_error)
        elif session.lyrics is None:
            embed.title = f"🎵 {track_title}"
            embed.description = "LRCLIB 與 YouTube 字幕都找不到可用的同步歌詞。"
            render_key = (session.track_key, "missing")
        elif session.lyrics.instrumental:
            embed.title = f"🎵 {track_title}"
            embed.description = "這首曲目在歌詞資料庫中標記為純音樂。"
            render_key = (session.track_key, "instrumental")
        elif not session.lyrics.lines:
            embed.title = f"🎵 {track_title}"
            embed.description = "有找到純文字歌詞，但目前沒有可用的同步時間軸。"
            embed.set_footer(text=f"來源：{session.lyrics.source}")
            render_key = (session.track_key, "plain-only", session.lyrics.source)
        else:
            embed.title = f"🎵 {session.lyrics.track_name}"
            if session.lyrics.artist_name:
                embed.set_author(name=session.lyrics.artist_name)
            embed.description = self._lyrics_window(session.lyrics, line_index)
            if status == "paused":
                embed.set_footer(text=f"⏸ 已暫停 • {session.lyrics.source}")
            else:
                duration = session.lyrics.duration
                if duration and duration > 0:
                    embed.set_footer(
                        text=(
                            f"▶ {self._format_time(session.elapsed_seconds)} / "
                            f"{self._format_time(duration)} • {session.lyrics.source}"
                        )
                    )
                else:
                    embed.set_footer(
                        text=f"▶ {self._format_time(session.elapsed_seconds)} • {session.lyrics.source}"
                    )
            render_key = self._render_key(session, line_index, status)

        try:
            if force_bump and session.message is not None:
                try:
                    await session.message.delete()
                except discord.HTTPException:
                    pass
                session.message = None
            if session.message is None:
                session.message = await channel.send(embed=embed)
            else:
                await session.message.edit(content=None, embed=embed)
            session.render_key = render_key
            session.needs_bump = False
        except discord.NotFound:
            session.message = None
            try:
                session.message = await channel.send(embed=embed)
                session.render_key = render_key
                session.needs_bump = False
            except discord.HTTPException:
                LOGGER.warning("Could not recreate live lyrics message in channel %s", session.channel_id)
        except discord.HTTPException:
            LOGGER.warning("Could not update live lyrics message in channel %s", session.channel_id)

    def _configured_channel(self, guild_id: int) -> int | None:
        raw = self.database.get_state(self._channel_key(guild_id))
        if not raw:
            return None
        try:
            channel_id = int(raw)
        except ValueError:
            return None
        return channel_id if channel_id > 0 else None

    def _configured_mode(self, guild_id: int) -> str:
        value = self.database.get_state(self._mode_key(guild_id)) or "panel"
        return value if value in LYRICS_MODES else "panel"

    def _configured_offset(self, guild_id: int) -> float:
        raw = self.database.get_state(self._offset_key(guild_id))
        if not raw:
            return DEFAULT_LYRICS_OFFSET_SECONDS
        try:
            value = float(raw)
        except ValueError:
            return DEFAULT_LYRICS_OFFSET_SECONDS
        return max(MIN_LYRICS_OFFSET_SECONDS, min(MAX_LYRICS_OFFSET_SECONDS, value))

    @staticmethod
    def _channel_key(guild_id: int) -> str:
        return f"{LYRICS_CHANNEL_STATE_PREFIX}{guild_id}"

    @staticmethod
    def _mode_key(guild_id: int) -> str:
        return f"{LYRICS_MODE_STATE_PREFIX}{guild_id}"

    @staticmethod
    def _offset_key(guild_id: int) -> str:
        return f"{LYRICS_OFFSET_STATE_PREFIX}{guild_id}"

    @staticmethod
    def _lyrics_window(lyrics: LyricsResult, line_index: int | None) -> str:
        lines = lyrics.lines
        if line_index is None:
            first = LyricsCommands._safe_line(lines[0].text)
            return f"*即將開始…*\n\n{first}"

        previous = LyricsCommands._safe_line(lines[line_index - 1].text) if line_index > 0 else ""
        current = LyricsCommands._safe_line(lines[line_index].text) or "♪"
        following = LyricsCommands._safe_line(lines[line_index + 1].text) if line_index + 1 < len(lines) else ""
        parts: list[str] = []
        if previous:
            parts.append(f"*{previous}*")
        parts.append(f"**▶ {current}**")
        if following:
            parts.append(following)
        return "\n\n".join(parts)

    @staticmethod
    def _safe_line(value: str) -> str:
        value = " ".join(value.split())
        return value[:350]

    @staticmethod
    def _format_time(seconds: float) -> str:
        whole = max(0, int(seconds))
        return f"{whole // 60}:{whole % 60:02d}"


def _command_path(data: dict[str, object]) -> tuple[str, ...]:
    """Return paths such as ('play',) or ('music', 'now') from Discord interaction data."""
    root = data.get("name")
    if not isinstance(root, str) or not root:
        return ()
    path = [root]
    options = data.get("options")
    while isinstance(options, list):
        nested = next(
            (
                item
                for item in options
                if isinstance(item, dict)
                and item.get("type") in {1, 2}
                and isinstance(item.get("name"), str)
            ),
            None,
        )
        if nested is None:
            break
        path.append(str(nested["name"]))
        options = nested.get("options")
    return tuple(path)
