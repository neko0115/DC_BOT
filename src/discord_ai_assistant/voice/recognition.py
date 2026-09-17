from __future__ import annotations

import asyncio
import audioop
import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import Callable

import discord
import numpy as np
from discord.ext import voice_recv
from faster_whisper import WhisperModel

LOGGER = logging.getLogger(__name__)
DISCORD_SAMPLE_RATE = 48_000
WHISPER_SAMPLE_RATE = 16_000


@dataclass(slots=True)
class AudioSegment:
    speaker_id: int
    started_at: float
    last_audio_at: float
    pcm_frames: bytearray = field(default_factory=bytearray)


@dataclass(frozen=True, slots=True)
class SpeakerReception:
    speaker_id: int
    packet_count: int
    pcm_bytes: int
    last_packet_at: float


@dataclass(frozen=True, slots=True)
class RecognitionSettings:
    model_name: str
    language: str
    beam_size: int
    silence_seconds: float
    max_segment_seconds: float
    initial_prompt: str


class PerSpeakerSegmentBuffer:
    """Separates speakers and forces long uninterrupted speech into bounded segments."""

    def __init__(self, silence_seconds: float = 1.2, max_segment_seconds: float = 15.0) -> None:
        self.silence_seconds = silence_seconds
        self.max_segment_seconds = max_segment_seconds
        self._segments: dict[int, AudioSegment] = {}
        self._lock = RLock()

    def add(self, speaker_id: int, pcm: bytes, now: float | None = None) -> list[AudioSegment]:
        current_time = monotonic() if now is None else now
        with self._lock:
            ready = self._flush_idle_unlocked(current_time)
            segment = self._segments.get(speaker_id)
            if segment is None or current_time - segment.started_at >= self.max_segment_seconds:
                if segment:
                    ready.append(segment)
                segment = AudioSegment(speaker_id, current_time, current_time)
                self._segments[speaker_id] = segment
            segment.pcm_frames.extend(pcm)
            segment.last_audio_at = current_time
            return ready

    def flush_idle(self, now: float | None = None) -> list[AudioSegment]:
        current_time = monotonic() if now is None else now
        with self._lock:
            return self._flush_idle_unlocked(current_time)

    def _flush_idle_unlocked(self, current_time: float) -> list[AudioSegment]:
        ready: list[AudioSegment] = []
        for speaker_id, segment in list(self._segments.items()):
            if current_time - segment.last_audio_at >= self.silence_seconds:
                ready.append(segment)
                del self._segments[speaker_id]
        return ready


class RecognitionTestPublisher:
    """Posts confirmed recognition results to the configured test text channel."""

    def __init__(self, channel_name: str) -> None:
        self.channel_name = channel_name

    async def publish(self, guild: discord.Guild, speaker_id: int, text: str) -> bool:
        if not self.channel_name or not text.strip():
            return False
        channel = discord.utils.get(guild.text_channels, name=self.channel_name)
        if not channel:
            return False
        member = guild.get_member(speaker_id)
        speaker_name = member.display_name if member else f"使用者 {speaker_id}"
        await channel.send(
            f"語音辨識測試 | {speaker_name}：{text.strip()[:1800]}",
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return True


class WhisperTranscriber:
    """Lazy, serialized CPU transcription so simultaneous speakers do not overload the host."""

    def __init__(
        self,
        model_name: str,
        language: str,
        model_directory: Path,
        beam_size: int,
        initial_prompt: str,
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.model_directory = model_directory
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self._model: WhisperModel | None = None
        self._lock = asyncio.Lock()

    async def transcribe(self, segment: AudioSegment) -> str:
        async with self._lock:
            return await asyncio.to_thread(self._transcribe_sync, segment)

    def _transcribe_sync(self, segment: AudioSegment) -> str:
        model = self._load_model()
        audio = self._to_whisper_audio(bytes(segment.pcm_frames))
        if audio.size == 0:
            return ""
        parts, _ = model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt or None,
        )
        return "".join(part.text for part in parts).strip()

    def _load_model(self) -> WhisperModel:
        if self._model is None:
            self.model_directory.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Loading Whisper model '%s'; the first use may download it", self.model_name)
            self._model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type="int8",
                download_root=str(self.model_directory),
            )
        return self._model

    @staticmethod
    def _to_whisper_audio(pcm: bytes) -> np.ndarray:
        if not pcm:
            return np.empty(0, dtype=np.float32)
        mono = audioop.tomono(pcm, 2, 0.5, 0.5)
        resampled, _ = audioop.ratecv(mono, 2, 1, DISCORD_SAMPLE_RATE, WHISPER_SAMPLE_RATE, None)
        return np.frombuffer(resampled, dtype=np.int16).astype(np.float32) / 32768.0


class VoiceRecognitionSink(voice_recv.AudioSink):
    """Receives decoded PCM on the voice thread and schedules finished segments safely."""

    def __init__(
        self,
        buffer: PerSpeakerSegmentBuffer,
        submit: Callable[[AudioSegment], None],
        on_audio: Callable[[int, int], None] | None = None,
    ) -> None:
        super().__init__()
        self.buffer = buffer
        self.submit = submit
        self.on_audio = on_audio

    def wants_opus(self) -> bool:
        return False

    def write(self, user: discord.User | None, data: voice_recv.VoiceData) -> None:
        if user is None or not data.pcm:
            return
        if self.on_audio:
            self.on_audio(user.id, len(data.pcm))
        for segment in self.buffer.add(user.id, data.pcm):
            self.submit(segment)

    def cleanup(self) -> None:
        # AudioSink requires this hook. This sink owns no external resources.
        return None


class VoiceRecognitionSession:
    def __init__(
        self,
        guild: discord.Guild,
        loop: asyncio.AbstractEventLoop,
        transcriber: WhisperTranscriber,
        publisher: RecognitionTestPublisher,
        silence_seconds: float,
        max_segment_seconds: float,
    ) -> None:
        self.guild = guild
        self.loop = loop
        self.transcriber = transcriber
        self.publisher = publisher
        self.buffer = PerSpeakerSegmentBuffer(silence_seconds, max_segment_seconds)
        self._speaker_reception: dict[int, SpeakerReception] = {}
        self._reception_lock = RLock()
        self.sink = VoiceRecognitionSink(self.buffer, self.submit, self.record_audio)
        self._stopped = False

    def record_audio(self, speaker_id: int, pcm_bytes: int) -> None:
        received_at = monotonic()
        with self._reception_lock:
            previous = self._speaker_reception.get(speaker_id)
            self._speaker_reception[speaker_id] = SpeakerReception(
                speaker_id=speaker_id,
                packet_count=(previous.packet_count if previous else 0) + 1,
                pcm_bytes=(previous.pcm_bytes if previous else 0) + pcm_bytes,
                last_packet_at=received_at,
            )

    def reception_snapshot(self) -> list[SpeakerReception]:
        with self._reception_lock:
            return sorted(self._speaker_reception.values(), key=lambda item: item.last_packet_at, reverse=True)

    def submit(self, segment: AudioSegment) -> None:
        if not self._stopped:
            self.loop.call_soon_threadsafe(lambda: asyncio.create_task(self._process(segment)))

    async def flush_idle(self) -> None:
        for segment in self.buffer.flush_idle():
            self.submit(segment)

    async def _process(self, segment: AudioSegment) -> None:
        try:
            text = await self.transcriber.transcribe(segment)
            if text and not self._stopped:
                await self.publisher.publish(self.guild, segment.speaker_id, text)
        except Exception:
            LOGGER.exception("Voice recognition failed for speaker %s in guild %s", segment.speaker_id, self.guild.id)

    def stop(self) -> None:
        self._stopped = True


class VoiceRecognitionController:
    def __init__(self, default_settings: RecognitionSettings, model_directory: Path, channel_name: str) -> None:
        self.default_settings = default_settings
        self.model_directory = model_directory
        self.publisher = RecognitionTestPublisher(channel_name)
        self._sessions: dict[int, VoiceRecognitionSession] = {}
        self._transcribers: dict[RecognitionSettings, WhisperTranscriber] = {}

    def start(
        self,
        guild: discord.Guild,
        loop: asyncio.AbstractEventLoop,
        settings: RecognitionSettings | None = None,
    ) -> VoiceRecognitionSession:
        existing = self._sessions.pop(guild.id, None)
        if existing:
            existing.stop()
        effective_settings = settings or self.default_settings
        transcriber = self._transcribers.get(effective_settings)
        if transcriber is None:
            transcriber = WhisperTranscriber(
                effective_settings.model_name,
                effective_settings.language,
                self.model_directory,
                effective_settings.beam_size,
                effective_settings.initial_prompt,
            )
            self._transcribers[effective_settings] = transcriber
        session = VoiceRecognitionSession(
            guild,
            loop,
            transcriber,
            self.publisher,
            effective_settings.silence_seconds,
            effective_settings.max_segment_seconds,
        )
        self._sessions[guild.id] = session
        return session

    def stop(self, guild_id: int) -> bool:
        session = self._sessions.pop(guild_id, None)
        if not session:
            return False
        session.stop()
        return True

    def session(self, guild_id: int) -> VoiceRecognitionSession | None:
        return self._sessions.get(guild_id)

    async def flush_all(self) -> None:
        for session in list(self._sessions.values()):
            await session.flush_idle()
