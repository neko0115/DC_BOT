from __future__ import annotations

import logging
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from discord.ext import commands, tasks

from discord_ai_assistant.music.evented_player import EventedEnhancedMusicManager

LOGGER = logging.getLogger(__name__)
WATCHDOG_INTERVAL_SECONDS = 5.0
TTS_STALL_GRACE_SECONDS = 12.0
RECONNECT_ESCALATION_WINDOW_SECONDS = 60.0


@dataclass(slots=True)
class _TTSObservation:
    track_identity: int
    first_seen_at: float
    expected_duration_seconds: float | None


class VoicePlaybackWatchdog(commands.Cog):
    """Recover Discord voice playback that is logically playing after FFmpeg/TTS has stalled."""

    def __init__(
        self,
        bot: commands.Bot,
        music: EventedEnhancedMusicManager,
        *,
        interval_seconds: float = WATCHDOG_INTERVAL_SECONDS,
        tts_stall_grace_seconds: float = TTS_STALL_GRACE_SECONDS,
    ) -> None:
        self.bot = bot
        self.music = music
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.tts_stall_grace_seconds = max(3.0, float(tts_stall_grace_seconds))
        self._tts_observations: dict[int, _TTSObservation] = {}
        self._last_recovery_at: dict[int, float] = {}

    async def cog_load(self) -> None:
        self._watch.change_interval(seconds=self.interval_seconds)
        self._watch.start()

    def cog_unload(self) -> None:
        self._watch.cancel()
        self._tts_observations.clear()
        self._last_recovery_at.clear()

    @tasks.loop(seconds=WATCHDOG_INTERVAL_SECONDS)
    async def _watch(self) -> None:
        await self.check_once()

    @_watch.before_loop
    async def _before_watch(self) -> None:
        await self.bot.wait_until_ready()

    async def check_once(self, *, now: float | None = None) -> None:
        current_time = time.monotonic() if now is None else float(now)
        for guild_id in self.music.active_guild_ids():
            state = self.music.state_for(guild_id)
            voice = state.voice
            if not voice or not voice.is_connected():
                self._tts_observations.pop(guild_id, None)
                continue

            if voice.is_paused():
                continue

            if not voice.is_playing():
                self._tts_observations.pop(guild_id, None)
                # A lost/late Discord after-callback can leave upcoming audio queued forever.
                # Starting only when the voice client is truly idle is safe and idempotent.
                if state.queue.snapshot():
                    started = await self.music._start_if_idle(guild_id)
                    if started:
                        LOGGER.info(
                            "Voice playback watchdog restarted an idle queued track in guild %s",
                            guild_id,
                        )
                continue

            reason = self._stalled_reason(guild_id, current_time)
            if not reason:
                continue

            previous = self._last_recovery_at.get(guild_id)
            escalate = bool(
                previous is not None
                and current_time - previous <= RECONNECT_ESCALATION_WINDOW_SECONDS
            )
            self._last_recovery_at[guild_id] = current_time
            self._tts_observations.pop(guild_id, None)
            if escalate:
                # The normal playback callback will reconnect before starting the next
                # queued item. Escalate only after repeated stalls so one bad FFmpeg
                # process does not disrupt voice receive/recognition unnecessarily.
                state.refresh_voice_connection = True

            LOGGER.warning(
                "Voice playback watchdog recovering guild %s: %s%s",
                guild_id,
                reason,
                " (voice reconnect armed)" if escalate else "",
            )
            self.music._stop_playback(voice)

    def _stalled_reason(self, guild_id: int, now: float) -> str | None:
        state = self.music.state_for(guild_id)
        current = state.queue.current
        if current is None:
            return "voice reports playing without a current track"
        if state.source is None:
            return "voice reports playing without a PCM source"

        ffmpeg_return_code = self.music._ffmpeg_return_code(state.source)
        if ffmpeg_return_code is not None:
            return f"FFmpeg already exited with code {ffmpeg_return_code}"

        if current.track.id != -2:
            self._tts_observations.pop(guild_id, None)
            return None

        identity = id(current)
        observation = self._tts_observations.get(guild_id)
        if observation is None or observation.track_identity != identity:
            expected = current.track.duration
            if expected is None and current.track.audio_path:
                expected = self._wav_duration_seconds(Path(current.track.audio_path))
            observation = _TTSObservation(identity, now, expected)
            self._tts_observations[guild_id] = observation
            return None

        expected = observation.expected_duration_seconds
        if expected is None:
            return None
        elapsed = now - observation.first_seen_at
        deadline = expected + self.tts_stall_grace_seconds
        if elapsed > deadline:
            return (
                f"TTS playback exceeded expected {expected:.1f}s + "
                f"{self.tts_stall_grace_seconds:.1f}s grace"
            )
        return None

    @staticmethod
    def _wav_duration_seconds(path: Path) -> float | None:
        try:
            with wave.open(str(path), "rb") as audio:
                rate = audio.getframerate()
                frames = audio.getnframes()
        except (OSError, EOFError, wave.Error):
            return None
        if rate <= 0:
            return None
        return frames / rate
