from __future__ import annotations

import asyncio
import logging
import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable
from uuid import uuid4

import discord
from discord.ext import voice_recv

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

from discord_ai_assistant.models import QueuedTrack, Track
from discord_ai_assistant.music.queue import GuildQueue

LOGGER = logging.getLogger(__name__)
DEFAULT_VOLUME = 0.2
MAX_VOLUME_PERCENT = 200
NEXT_TRACK_RETRY_ATTEMPTS = 20
NEXT_TRACK_RETRY_DELAY_SECONDS = 0.25
YOUTUBE_STREAM_RETRY_ATTEMPTS = 2

# YouTube 提取設定
YDL_OPTIONS = {
    'format': 'bestaudio/best',
    'noplaylist': True,
    'quiet': True,
    'no_warnings': True,
    'noprogress': True,
}


class _QuietYtDlpLogger:
    """Avoid duplicating yt-dlp download failures in the bot console."""

    def debug(self, message: str) -> None:
        LOGGER.debug("yt-dlp: %s", message)

    def warning(self, message: str) -> None:
        LOGGER.debug("yt-dlp: %s", message)

    def error(self, message: str) -> None:
        LOGGER.debug("yt-dlp: %s", message)

@dataclass(slots=True)
class GuildPlayback:
    queue: GuildQueue = field(default_factory=GuildQueue)
    voice: discord.VoiceClient | None = None
    source: discord.PCMVolumeTransformer | None = None
    volume: float = DEFAULT_VOLUME
    repeat_one: bool = False
    shuffle: bool = False
    auto_recommend: bool = False
    skip_requested: bool = False
    empty_since: float | None = None
    start_retry_task: asyncio.Task[None] | None = None
    failed_track_attempts: dict[QueuedTrack, int] = field(default_factory=dict)
    refresh_voice_connection: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class MusicManager:
    def __init__(
        self,
        library_root: Path,
        recommendation_provider: Callable[[int | None], Track | None] | None = None,
        *,
        youtube_cookies_from_browser: str | None = None,
        youtube_cookies_file: Path | None = None,
        youtube_po_token: str | None = None,
    ) -> None:
        self.library_root = library_root
        self.stream_cache_path = library_root.parent / "data" / "stream-cache"
        self.recommendation_provider = recommendation_provider
        self.youtube_cookies_from_browser = youtube_cookies_from_browser
        self.youtube_cookies_file = youtube_cookies_file
        self.youtube_po_token = youtube_po_token
        self._guilds: dict[int, GuildPlayback] = {}
        self._cache_cleanup_tasks: dict[Path, asyncio.Task[None]] = {}
        self._node_runtime_available = shutil.which("node") is not None
        if not self._node_runtime_available:
            LOGGER.warning("Node.js was not found; yt-dlp will run without the Node JavaScript runtime.")

    def state_for(self, guild_id: int) -> GuildPlayback:
        return self._guilds.setdefault(guild_id, GuildPlayback())

    async def connect(self, guild_id: int, channel: discord.VoiceChannel | discord.StageChannel) -> None:
        state = self.state_for(guild_id)
        if state.refresh_voice_connection and state.voice and state.voice.is_connected():
            if state.voice.is_playing() or state.voice.is_paused():
                LOGGER.info("Deferring voice refresh in guild %s until playback is idle", guild_id)
            else:
                if not await self._refresh_voice_connection(guild_id):
                    raise RuntimeError("Discord 語音連線重新建立失敗，請稍後再試。")
        if state.voice and state.voice.is_connected():
            if state.voice.channel != channel:
                await state.voice.move_to(channel)
            return
        state.voice = await channel.connect(cls=voice_recv.VoiceRecvClient, self_deaf=False)
        state.refresh_voice_connection = False

    def mark_voice_connections_for_refresh(self) -> None:
        """Reconnect before future playback after the main Discord gateway reconnects."""
        for state in self._guilds.values():
            if state.voice and state.voice.is_connected():
                state.refresh_voice_connection = True

    async def disconnect(self, guild_id: int) -> None:
        state = self.state_for(guild_id)
        self._cancel_start_retry(state)
        self._cleanup_tracks([state.queue.current, *state.queue.snapshot()])
        state.queue.clear()
        state.failed_track_attempts.clear()
        if state.voice and state.voice.is_connected():
            await state.voice.disconnect(force=True)
        state.voice = None
        state.source = None
        state.empty_since = None
        state.refresh_voice_connection = False

    async def enqueue(self, guild_id: int, track: Track, requested_by: int, next_up: bool = False) -> bool:
        state = self.state_for(guild_id)
        prepared_track = await self._prepare_track(track)
        if not prepared_track:
            return False
        item = QueuedTrack(track=prepared_track, requested_by=requested_by)
        if next_up:
            state.queue.append_next(item)
        else:
            state.queue.append(item)
        await self._start_if_idle(guild_id)
        return self._is_current_or_queued(state, item)

    async def play_now(self, guild_id: int, track: Track, requested_by: int) -> bool:
        state = self.state_for(guild_id)
        prepared_track = await self._prepare_track(track)
        if not prepared_track:
            return False
        item = QueuedTrack(track=prepared_track, requested_by=requested_by)
        state.queue.append_next(item)
        if state.voice and (state.voice.is_playing() or state.voice.is_paused()):
            state.skip_requested = True
            self._stop_playback(state.voice)
            return True
        else:
            await self._start_if_idle(guild_id)
        return self._is_current_or_queued(state, item)

    async def skip(self, guild_id: int) -> bool:
        state = self.state_for(guild_id)
        if not state.voice or not (state.voice.is_playing() or state.voice.is_paused()):
            return False
        state.skip_requested = True
        self._stop_playback(state.voice)
        return True

    async def previous(self, guild_id: int) -> bool:
        state = self.state_for(guild_id)
        if not state.queue.queue_previous():
            return False
        if state.voice and (state.voice.is_playing() or state.voice.is_paused()):
            state.skip_requested = True
            self._stop_playback(state.voice)
            return True
        state.queue.current = None
        return await self._start_if_idle(guild_id)

    async def stop(self, guild_id: int) -> None:
        state = self.state_for(guild_id)
        self._cancel_start_retry(state)
        self._cleanup_tracks([state.queue.current, *state.queue.snapshot()])
        state.queue.clear()
        state.failed_track_attempts.clear()
        if state.voice and (state.voice.is_playing() or state.voice.is_paused()):
            state.skip_requested = True
            self._stop_playback(state.voice)

    async def pause(self, guild_id: int) -> bool:
        state = self.state_for(guild_id)
        if not state.voice or not state.voice.is_playing():
            return False
        state.voice.pause()
        return True

    async def resume(self, guild_id: int) -> bool:
        state = self.state_for(guild_id)
        if not state.voice or not state.voice.is_paused():
            return False
        state.voice.resume()
        return True

    def queue_view(self, guild_id: int) -> tuple[QueuedTrack | None, list[QueuedTrack]]:
        state = self.state_for(guild_id)
        return state.queue.current, state.queue.snapshot()

    def set_volume(self, guild_id: int, percent: int) -> int:
        if not 0 <= percent <= MAX_VOLUME_PERCENT:
            raise ValueError(f"音量必須介於 0 到 {MAX_VOLUME_PERCENT}。")
        state = self.state_for(guild_id)
        state.volume = percent / 100
        if state.source:
            state.source.volume = state.volume
        return percent

    def volume_percent(self, guild_id: int) -> int:
        return round(self.state_for(guild_id).volume * 100)

    def set_repeat_one(self, guild_id: int, enabled: bool) -> bool:
        self.state_for(guild_id).repeat_one = enabled
        return enabled

    def set_shuffle(self, guild_id: int, enabled: bool) -> bool:
        self.state_for(guild_id).shuffle = enabled
        return enabled

    def set_auto_recommend(self, guild_id: int, enabled: bool) -> bool:
        self.state_for(guild_id).auto_recommend = enabled
        return enabled

    def playback_modes(self, guild_id: int) -> tuple[bool, bool, bool]:
        state = self.state_for(guild_id)
        return state.repeat_one, state.shuffle, state.auto_recommend

    async def enqueue_tts(self, guild_id: int, audio_path: Path, requested_by: int) -> None:
        track = Track(
            id=-2,
            title="墨雪語音回覆",
            original_name=audio_path.name,
            stored_name=audio_path.name,
            uploaded_by=requested_by,
            audio_path=str(audio_path),
            delete_after_play=True,
        )
        await self.enqueue(guild_id, track, requested_by)

    def active_guild_ids(self) -> list[int]:
        return list(self._guilds)

    def start_listening(self, guild_id: int, sink: voice_recv.AudioSink) -> bool:
        voice = self.state_for(guild_id).voice
        if not isinstance(voice, voice_recv.VoiceRecvClient):
            return False
        if voice.is_listening():
            voice.stop_listening()
        voice.listen(sink)
        return True

    def stop_listening(self, guild_id: int) -> bool:
        voice = self.state_for(guild_id).voice
        if not isinstance(voice, voice_recv.VoiceRecvClient) or not voice.is_listening():
            return False
        voice.stop_listening()
        return True

    def should_disconnect_if_empty(self, guild_id: int, timeout_seconds: int) -> bool:
        state = self.state_for(guild_id)
        voice = state.voice
        channel = voice.channel if voice and voice.is_connected() else None
        members = getattr(channel, "members", []) if channel else []
        has_human = any(not member.bot for member in members)
        if has_human:
            state.empty_since = None
            return False
        now = time.monotonic()
        if state.empty_since is None:
            state.empty_since = now
            return False
        return now - state.empty_since >= timeout_seconds

    async def _start_if_idle(self, guild_id: int) -> bool:
        state = self.state_for(guild_id)
        async with state.lock:
            voice = state.voice
            if not voice or not voice.is_connected() or voice.is_playing() or voice.is_paused():
                return False
            if state.refresh_voice_connection:
                if not await self._refresh_voice_connection(guild_id):
                    return False
                voice = state.voice
                if not voice or not voice.is_connected():
                    return False

            while item := state.queue.advance(shuffle=state.shuffle):
                track_path = item.track.path(self.library_root)
                youtube_url = self._youtube_url(item.track)

                if item.track.audio_path:
                    source_audio = discord.FFmpegPCMAudio(
                        item.track.audio_path, before_options="-nostdin", options="-vn"
                    )
                elif item.track.id != -1 and track_path.is_file():
                    # 播放本地文件
                    source_audio = discord.FFmpegPCMAudio(
                        str(track_path), before_options="-nostdin", options="-vn"
                    )
                elif youtube_url:
                    prepared_track = await self._prepare_track(item.track)
                    if not prepared_track or not prepared_track.audio_path:
                        LOGGER.warning("Could not cache YouTube audio for '%s'", item.track.title)
                        continue
                    item = QueuedTrack(track=prepared_track, requested_by=item.requested_by)
                    state.queue.current = item
                    source_audio = discord.FFmpegPCMAudio(
                        prepared_track.audio_path,
                        before_options="-nostdin",
                        options="-vn",
                    )
                elif item.track.stream_url:
                    source_audio = discord.FFmpegPCMAudio(
                        item.track.stream_url,
                        before_options="-nostdin",
                        options="-vn",
                    )
                else:
                    LOGGER.warning("Unsupported track source: %s", item.track.id)
                    continue

                source = discord.PCMVolumeTransformer(
                    source_audio,
                    volume=state.volume,
                )
                state.source = source
                try:
                    voice.play(source, after=lambda error: self._after_track(guild_id, error))
                except (discord.ClientException, OSError) as error:
                    LOGGER.warning("Could not start track '%s'; will retry: %s", item.track.title, error)
                    state.queue.current = None
                    state.source = None
                    state.queue.append_next(item)
                    self._schedule_start_retry(guild_id)
                    return False
                LOGGER.info("Started track '%s' in guild %s", item.track.title, guild_id)
                return True
        return False

    async def _prepare_track(self, track: Track) -> Track | None:
        if track.audio_path or not self._youtube_url(track):
            return track
        cached_path = await self._download_youtube_audio(self._youtube_url(track) or "")
        if not cached_path:
            return None
        return replace(track, audio_path=str(cached_path), delete_after_play=True)

    async def _download_youtube_audio(self, url: str) -> Path | None:
        if yt_dlp is None:
            LOGGER.error("yt-dlp is not installed; cannot cache YouTube audio")
            return None
        cache_job_path = self.stream_cache_path / uuid4().hex

        def download() -> Path | None:
            cache_job_path.mkdir(parents=True, exist_ok=False)
            try:
                return self._download_youtube_audio_with_options(url, cache_job_path, use_credentials=True)
            except Exception as error:
                if not self._is_cookie_access_error(error):
                    raise
                LOGGER.warning("Browser cookies are unavailable; retrying YouTube download without them.")
                self._clear_cache_job_files(cache_job_path)
                return self._download_youtube_audio_with_options(url, cache_job_path, use_credentials=False)

        try:
            loop = asyncio.get_running_loop()
            cached_path = await loop.run_in_executor(None, download)
        except Exception as error:
            message = str(error).replace("\n", " ")
            if "HTTP Error 403" in message or "DECRYPTION_FAILED_OR_BAD_RECORD_MAC" in message:
                LOGGER.warning("YouTube refused the temporary audio download: %s", message[:220])
            else:
                LOGGER.exception("Could not cache YouTube audio")
            self._remove_empty_cache_job(cache_job_path)
            return None
        if not cached_path:
            self._remove_empty_cache_job(cache_job_path)
            return None
        LOGGER.info("Cached YouTube audio at %s", cached_path)
        return cached_path

    def _download_youtube_audio_with_options(
        self, url: str, cache_job_path: Path, *, use_credentials: bool
    ) -> Path | None:
            options = {
                **YDL_OPTIONS,
                "outtmpl": str(cache_job_path / "%(id)s.%(ext)s"),
                "logger": _QuietYtDlpLogger(),
            }
            if self._node_runtime_available:
                options["js_runtimes"] = {"node": {}}
            if use_credentials and self.youtube_cookies_from_browser:
                options["cookiesfrombrowser"] = (self.youtube_cookies_from_browser,)
            elif use_credentials and self.youtube_cookies_file:
                options["cookiefile"] = str(self.youtube_cookies_file)
            if use_credentials and self.youtube_po_token:
                options["extractor_args"] = {
                    "youtube": {
                        "player_client": ["mweb"],
                        "po_token": [f"mweb.gvs+{self.youtube_po_token}"],
                    }
                }
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(url, download=True)
            requested_downloads = info.get("requested_downloads") if isinstance(info, dict) else None
            if isinstance(requested_downloads, list):
                for download_info in requested_downloads:
                    file_path = download_info.get("filepath") if isinstance(download_info, dict) else None
                    if isinstance(file_path, str) and Path(file_path).is_file():
                        return Path(file_path)
            return next((path for path in cache_job_path.iterdir() if path.is_file()), None)

    @staticmethod
    def _is_cookie_access_error(error: Exception) -> bool:
        message = str(error).lower()
        return "cookie database" in message or "failed to load cookies" in message

    @staticmethod
    def _clear_cache_job_files(cache_job_path: Path) -> None:
        for path in cache_job_path.iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)

    @staticmethod
    def _youtube_url(track: Track) -> str | None:
        for value in (track.original_name, track.stored_name):
            if "youtube.com" in value or "youtu.be" in value:
                return value
        return None

    def _after_track(self, guild_id: int, error: Exception | None) -> None:
        state = self.state_for(guild_id)
        ffmpeg_return_code = self._ffmpeg_return_code(state.source)
        if error is None and not state.skip_requested and ffmpeg_return_code not in (None, 0):
            error = RuntimeError(f"FFmpeg exited with code {ffmpeg_return_code}")
        if error:
            LOGGER.error("Voice playback failed in guild %s: %s", guild_id, error)
        if _MAIN_LOOP is None:
            LOGGER.error("Playback callback arrived before the event loop was configured.")
            return
        # discord.py invokes this callback from its audio thread.
        if _MAIN_LOOP.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._advance_after_callback(guild_id, error), _MAIN_LOOP)
        except RuntimeError:
            LOGGER.debug("Ignored playback callback while the bot was shutting down.")

    async def _advance_after_callback(self, guild_id: int, error: Exception | None) -> None:
        state = self.state_for(guild_id)
        finished = state.queue.current
        repeat_current = bool(finished and state.repeat_one and not state.skip_requested and error is None)
        if repeat_current and finished:
            state.queue.append_next(finished)
        elif finished and error and not state.skip_requested and self._youtube_url(finished.track):
            retry_count = state.failed_track_attempts.get(finished, 0) + 1
            state.failed_track_attempts[finished] = retry_count
            if retry_count <= YOUTUBE_STREAM_RETRY_ATTEMPTS:
                state.queue.append_next(finished)
                LOGGER.warning(
                    "Retrying YouTube track '%s' in guild %s (%s/%s)",
                    finished.track.title,
                    guild_id,
                    retry_count,
                    YOUTUBE_STREAM_RETRY_ATTEMPTS,
                )
            else:
                state.failed_track_attempts.pop(finished, None)
                self._cleanup_tracks([finished])
        elif finished:
            state.failed_track_attempts.pop(finished, None)
            self._cleanup_tracks([finished])
        if not state.queue.snapshot() and error is None and state.auto_recommend and self.recommendation_provider:
            exclude_id = finished.track.id if finished and finished.track.id > 0 else None
            recommendation = self.recommendation_provider(exclude_id)
            if recommendation:
                state.queue.append(QueuedTrack(recommendation, requested_by=0))
                LOGGER.info("Queued automatic recommendation '%s' in guild %s", recommendation.title, guild_id)
        state.skip_requested = False
        state.queue.current = None
        state.source = None
        for attempt in range(NEXT_TRACK_RETRY_ATTEMPTS):
            if await self._start_if_idle(guild_id):
                return
            if not state.queue.snapshot() or not state.voice or not state.voice.is_connected() or state.voice.is_paused():
                return
            await asyncio.sleep(NEXT_TRACK_RETRY_DELAY_SECONDS)
        LOGGER.warning("Could not start the next queued track in guild %s after %s retries", guild_id, NEXT_TRACK_RETRY_ATTEMPTS)

    def _is_current_or_queued(self, state: GuildPlayback, item: QueuedTrack) -> bool:
        return state.queue.current is item or any(queued is item for queued in state.queue.snapshot())

    def _schedule_start_retry(self, guild_id: int) -> None:
        state = self.state_for(guild_id)
        if state.start_retry_task and not state.start_retry_task.done():
            return
        state.start_retry_task = asyncio.create_task(self._retry_start(guild_id), name=f"music-start-retry-{guild_id}")

    async def _retry_start(self, guild_id: int) -> None:
        state = self.state_for(guild_id)
        this_task = asyncio.current_task()
        try:
            for _ in range(NEXT_TRACK_RETRY_ATTEMPTS):
                await asyncio.sleep(NEXT_TRACK_RETRY_DELAY_SECONDS)
                if await self._start_if_idle(guild_id):
                    return
                voice = state.voice
                if not state.queue.snapshot() or not voice or not voice.is_connected() or voice.is_playing() or voice.is_paused():
                    return
            LOGGER.warning("Could not retry queued playback in guild %s", guild_id)
        finally:
            if state.start_retry_task is this_task:
                state.start_retry_task = None

    async def _refresh_voice_connection(self, guild_id: int) -> bool:
        state = self.state_for(guild_id)
        voice = state.voice
        channel = voice.channel if voice and voice.is_connected() else None
        if not isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
            return False
        if voice.is_playing() or voice.is_paused():
            return False
        LOGGER.info("Refreshing Discord voice connection in guild %s after gateway reconnect", guild_id)
        try:
            await voice.disconnect(force=True)
            state.voice = await channel.connect(cls=voice_recv.VoiceRecvClient, self_deaf=False)
        except (discord.ClientException, OSError):
            LOGGER.exception("Could not refresh Discord voice connection in guild %s", guild_id)
            state.voice = None
            return False
        state.source = None
        state.refresh_voice_connection = False
        return True

    @staticmethod
    def _cancel_start_retry(state: GuildPlayback) -> None:
        if state.start_retry_task and not state.start_retry_task.done():
            state.start_retry_task.cancel()
        state.start_retry_task = None

    @staticmethod
    def _ffmpeg_return_code(source: discord.PCMVolumeTransformer | None) -> int | None:
        original = getattr(source, "original", None)
        process = getattr(original, "_process", None)
        poll = getattr(process, "poll", None)
        if not callable(poll):
            return None
        try:
            return poll()
        except OSError:
            return None

    def _cleanup_tracks(self, items: list[QueuedTrack | None]) -> None:
        for item in items:
            if not item or not item.track.delete_after_play or not item.track.audio_path:
                continue
            try:
                audio_path = Path(item.track.audio_path)
                audio_path.unlink(missing_ok=True)
                self._remove_empty_cache_job(audio_path.parent)
            except PermissionError:
                self._schedule_cache_cleanup(audio_path)
            except OSError:
                LOGGER.warning("Could not remove temporary audio file %s", item.track.audio_path)

    def _remove_empty_cache_job(self, path: Path) -> None:
        try:
            if path.parent == self.stream_cache_path:
                path.rmdir()
        except OSError:
            pass

    def _schedule_cache_cleanup(self, audio_path: Path) -> None:
        if audio_path.parent.parent != self.stream_cache_path:
            LOGGER.warning("Could not remove temporary audio file %s", audio_path)
            return
        task = self._cache_cleanup_tasks.get(audio_path)
        if task and not task.done():
            return
        self._cache_cleanup_tasks[audio_path] = asyncio.create_task(
            self._retry_cache_cleanup(audio_path), name=f"cache-cleanup-{audio_path.parent.name}"
        )

    async def _retry_cache_cleanup(self, audio_path: Path) -> None:
        this_task = asyncio.current_task()
        try:
            for _ in range(20):
                await asyncio.sleep(0.5)
                try:
                    audio_path.unlink(missing_ok=True)
                    self._remove_empty_cache_job(audio_path.parent)
                    LOGGER.info("Removed delayed temporary audio file %s", audio_path)
                    return
                except PermissionError:
                    continue
                except OSError:
                    break
            LOGGER.warning("Could not remove temporary audio file after retries: %s", audio_path)
        finally:
            if self._cache_cleanup_tasks.get(audio_path) is this_task:
                self._cache_cleanup_tasks.pop(audio_path, None)

    @staticmethod
    def _stop_playback(voice: discord.VoiceClient) -> None:
        if isinstance(voice, voice_recv.VoiceRecvClient):
            voice.stop_playing()
        else:
            voice.stop()


_MAIN_LOOP: asyncio.AbstractEventLoop | None = None


def configure_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _MAIN_LOOP
    _MAIN_LOOP = loop
