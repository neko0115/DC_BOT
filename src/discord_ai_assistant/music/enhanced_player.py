from __future__ import annotations

import logging
import sys
import threading
import time
from array import array
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import discord

from discord_ai_assistant.models import QueuedTrack, Track
from discord_ai_assistant.music.player import MusicManager

LOGGER = logging.getLogger(__name__)
MUSIC_TARGET_LUFS = -18.0
MUSIC_TRUE_PEAK_DB = -1.5
MUSIC_LRA = 11.0
DEFAULT_DUCKING_RATIO = 0.30
# discord.py expects one 20 ms PCM frame for 48 kHz / stereo / signed 16-bit audio.
PCM_FRAME_BYTES = 48_000 * 2 * 2 // 50
# Do not duck the song for effectively silent TTS frames. This is intentionally low;
# it only filters digital silence / near-silence, not softly spoken syllables.
SPEECH_DUCKING_PEAK_THRESHOLD = 32


@dataclass(slots=True)
class _SpeechClip:
    source: discord.AudioSource
    path: Path | None = None

    def cleanup(self) -> None:
        try:
            self.source.cleanup()
        finally:
            if self.path is None:
                return
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("Could not remove temporary TTS file %s", self.path)


class DuckingAudioSource(discord.AudioSource):
    """Mix speech over music while lowering music only for audible spoken frames.

    Discord pulls 20 ms PCM frames from this source on its audio thread. Speech clips
    can be queued safely from the asyncio thread without stopping or seeking the song.
    A broken speech decoder is isolated from the music path so one TTS failure cannot
    terminate the shared Discord AudioPlayer thread.
    """

    def __init__(
        self,
        music_source: discord.AudioSource,
        *,
        volume: float,
        ducking_ratio: float = DEFAULT_DUCKING_RATIO,
    ) -> None:
        if music_source.is_opus():
            raise ValueError("DuckingAudioSource requires PCM input")
        self.original = music_source
        self._volume = max(0.0, float(volume))
        self.ducking_ratio = max(0.0, min(float(ducking_ratio), 1.0))
        self._pending: deque[_SpeechClip] = deque()
        self._active_speech: _SpeechClip | None = None
        self._lock = threading.Lock()
        self._music_finished = False
        self._closed = False
        self._last_error: Exception | None = None
        self._last_read_at = time.monotonic()
        self._frames_read = 0

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        self._volume = max(0.0, float(value))

    @property
    def last_error(self) -> Exception | None:
        return self._last_error

    @property
    def music_finished(self) -> bool:
        return self._music_finished

    @property
    def last_read_at(self) -> float:
        return self._last_read_at

    @property
    def frames_read(self) -> int:
        return self._frames_read

    def enqueue_speech(self, source: discord.AudioSource, path: Path | None = None) -> None:
        if source.is_opus():
            raise ValueError("Speech overlay requires PCM input")
        clip = _SpeechClip(source=source, path=path)
        with self._lock:
            if self._closed:
                clip.cleanup()
                return
            self._pending.append(clip)

    def has_speech(self) -> bool:
        with self._lock:
            return self._active_speech is not None or bool(self._pending)

    def is_opus(self) -> bool:
        return False

    def read(self) -> bytes:
        if self._closed:
            return b""

        self._last_read_at = time.monotonic()
        self._frames_read += 1

        try:
            music_data = self._read_music_frame()
            speech_data = self._read_speech_frame()

            if not music_data and not speech_data:
                return b""
            if not speech_data:
                return _scale_pcm(music_data, self._volume)
            if not music_data:
                return _scale_pcm(speech_data, self._volume)

            # Gemini/Kokoro/SAPI files can contain leading/trailing silent PCM. Do not
            # make the song appear to vanish merely because a speech decoder is emitting
            # silent frames.
            if _pcm_peak(speech_data) < SPEECH_DUCKING_PEAK_THRESHOLD:
                return _mix_pcm(
                    music_data,
                    speech_data,
                    music_gain=self._volume,
                    speech_gain=self._volume,
                )

            return _mix_pcm(
                music_data,
                speech_data,
                music_gain=self._volume * self.ducking_ratio,
                speech_gain=self._volume,
            )
        except Exception as error:
            # Never let an unclassified mixer bug tear down the audio thread silently.
            # The manager promotes this stored error into the normal playback retry path.
            self._last_error = error
            LOGGER.exception("Audio mixer failed while producing a Discord PCM frame")
            return b""

    def cleanup(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            active = self._active_speech
            self._active_speech = None
            pending = list(self._pending)
            self._pending.clear()
        try:
            self.original.cleanup()
        finally:
            if active:
                active.cleanup()
            for clip in pending:
                clip.cleanup()

    def _read_music_frame(self) -> bytes:
        if self._music_finished:
            return b""
        try:
            data = self.original.read()
        except Exception as error:
            self._last_error = error
            self._music_finished = True
            LOGGER.exception("Music source failed inside the audio mixer")
            return b""
        if not data:
            self._music_finished = True
            return b""
        return _fit_pcm_frame(data)

    def _read_speech_frame(self) -> bytes:
        while True:
            clip = self._current_speech()
            if clip is None:
                return b""
            try:
                data = clip.source.read()
            except Exception:
                # Speech is an overlay. A malformed/failed TTS clip must not stop the
                # underlying song or poison future speech clips.
                LOGGER.exception("Speech overlay source failed; dropping this TTS clip")
                self._finish_current_speech(clip)
                continue
            if data:
                return _fit_pcm_frame(data)
            self._finish_current_speech(clip)

    def _current_speech(self) -> _SpeechClip | None:
        with self._lock:
            if self._active_speech is None and self._pending:
                self._active_speech = self._pending.popleft()
            return self._active_speech

    def _finish_current_speech(self, clip: _SpeechClip) -> None:
        with self._lock:
            if self._active_speech is clip:
                self._active_speech = None
        clip.cleanup()


class EnhancedMusicManager(MusicManager):
    """Music manager with loudness-normalized music and non-blocking TTS overlays."""

    async def enqueue_tts(self, guild_id: int, audio_path: Path, requested_by: int) -> None:
        state = self.state_for(guild_id)
        voice = state.voice
        source = state.source

        # When music is already active, inject speech into the current PCM stream.
        # The song keeps progressing underneath the speech, so lyrics/timing stay aligned.
        if voice and voice.is_playing() and isinstance(source, DuckingAudioSource):
            speech_source = discord.FFmpegPCMAudio(
                str(audio_path),
                before_options="-nostdin",
                options="-vn -ar 48000 -ac 2",
            )
            source.enqueue_speech(speech_source, path=audio_path)
            LOGGER.info("Queued TTS overlay in guild %s while music continues", guild_id)
            return

        # No active music mixer: preserve the existing queue semantics, but put speech
        # next in line so a paused/finishing source does not strand a voice response.
        track = Track(
            id=-2,
            title="墨雪語音回覆",
            original_name=audio_path.name,
            stored_name=audio_path.name,
            uploaded_by=requested_by,
            audio_path=str(audio_path),
            delete_after_play=True,
        )
        await self.enqueue(guild_id, track, requested_by, next_up=True)

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
                is_tts = item.track.id == -2
                music_options = (
                    "-vn "
                    f"-af loudnorm=I={MUSIC_TARGET_LUFS}:TP={MUSIC_TRUE_PEAK_DB}:LRA={MUSIC_LRA} "
                    "-ar 48000 -ac 2"
                )
                options = "-vn -ar 48000 -ac 2" if is_tts else music_options

                if item.track.audio_path:
                    source_audio = discord.FFmpegPCMAudio(
                        item.track.audio_path,
                        before_options="-nostdin",
                        options=options,
                    )
                elif item.track.id != -1 and track_path.is_file():
                    source_audio = discord.FFmpegPCMAudio(
                        str(track_path),
                        before_options="-nostdin",
                        options=options,
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
                        options=options,
                    )
                elif item.track.stream_url:
                    source_audio = discord.FFmpegPCMAudio(
                        item.track.stream_url,
                        before_options="-nostdin",
                        options=options,
                    )
                else:
                    LOGGER.warning("Unsupported track source: %s", item.track.id)
                    continue

                if is_tts:
                    source: discord.AudioSource = discord.PCMVolumeTransformer(
                        source_audio,
                        volume=state.volume,
                    )
                else:
                    source = DuckingAudioSource(source_audio, volume=state.volume)

                state.source = source  # type: ignore[assignment]
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

    def _after_track(self, guild_id: int, error: Exception | None) -> None:
        # DuckingAudioSource deliberately converts an exception on the Discord audio
        # thread into EOF so the callback is guaranteed to run. Restore that exception
        # here so MusicManager retries a failed YouTube source instead of treating the
        # dropout as a normal end-of-song event.
        source = self.state_for(guild_id).source
        if error is None and isinstance(source, DuckingAudioSource) and source.last_error is not None:
            error = source.last_error
        super()._after_track(guild_id, error)

    def set_volume(self, guild_id: int, percent: int) -> int:
        value = super().set_volume(guild_id, percent)
        source = self.state_for(guild_id).source
        if isinstance(source, DuckingAudioSource):
            source.volume = percent / 100
        return value


def _fit_pcm_frame(data: bytes) -> bytes:
    """Return exactly one Discord PCM frame for any non-empty decoder output."""
    if not data:
        return b""
    # Signed 16-bit PCM cannot contain a half sample. Discard a trailing partial byte
    # before padding so array('h') never raises on an odd-length decoder tail.
    if len(data) % 2:
        data = data[:-1]
    if not data:
        return b""
    if len(data) >= PCM_FRAME_BYTES:
        return data[:PCM_FRAME_BYTES]
    return data + (b"\x00" * (PCM_FRAME_BYTES - len(data)))


def _scale_pcm(data: bytes, gain: float) -> bytes:
    if not data or gain == 1.0:
        return data
    samples = _pcm_array(data)
    for index, sample in enumerate(samples):
        samples[index] = _clamp_sample(round(sample * gain))
    return _pcm_bytes(samples)


def _mix_pcm(music: bytes, speech: bytes, *, music_gain: float, speech_gain: float) -> bytes:
    music_samples = _pcm_array(music)
    speech_samples = _pcm_array(speech)
    length = max(len(music_samples), len(speech_samples))
    if len(music_samples) < length:
        music_samples.extend([0] * (length - len(music_samples)))
    if len(speech_samples) < length:
        speech_samples.extend([0] * (length - len(speech_samples)))
    output = array("h", [0]) * length
    for index in range(length):
        mixed = round(music_samples[index] * music_gain + speech_samples[index] * speech_gain)
        output[index] = _clamp_sample(mixed)
    return _pcm_bytes(output)


def _pcm_peak(data: bytes) -> int:
    if not data:
        return 0
    samples = _pcm_array(data)
    return max((abs(int(sample)) for sample in samples), default=0)


def _pcm_array(data: bytes) -> array:
    samples = array("h")
    samples.frombytes(data)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def _pcm_bytes(samples: array) -> bytes:
    if sys.byteorder != "little":
        copied = array("h", samples)
        copied.byteswap()
        return copied.tobytes()
    return samples.tobytes()


def _clamp_sample(value: int) -> int:
    return max(-32768, min(32767, value))
