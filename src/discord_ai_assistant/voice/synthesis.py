from __future__ import annotations

import asyncio
import base64
import logging
import os
import shutil
import uuid
import wave
from pathlib import Path
from typing import Protocol

LOGGER = logging.getLogger(__name__)
TTS_TARGET_LUFS = -16.0
TTS_TRUE_PEAK_DB = -1.5
TTS_LRA = 7.0
TTS_MAX_CHARACTERS = 400
GEMINI_PCM_RATE = 24000
KOKORO_PCM_RATE = 24000
DEFAULT_GEMINI_STYLE = (
    "Speak in natural Taiwan Mandarin with a youthful, soft female voice. "
    "Sound warm, relaxed, slightly playful, and conversational. "
    "Use natural pauses and subtle emotional variation. Avoid announcer-like delivery."
)


class SpeechSynthesizer(Protocol):
    async def synthesize(self, text: str) -> Path: ...


def _validated_text(text: str) -> str:
    value = text.strip()
    if not value or len(value) > TTS_MAX_CHARACTERS:
        raise ValueError(f"朗讀內容需介於 1 到 {TTS_MAX_CHARACTERS} 個字元。")
    return value


def _env_enabled(name: str, default: bool) -> bool:
    fallback = "true" if default else "false"
    return os.getenv(name, fallback).strip().lower() in {"1", "true", "yes", "on"}


class _NormalizedSpeechSynthesizer:
    def __init__(self, output_directory: Path) -> None:
        self.output_directory = output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def _paths(self, provider: str) -> tuple[Path, Path]:
        speech_id = uuid.uuid4().hex
        raw_path = self.output_directory / f"speech-{speech_id}-{provider}-raw.wav"
        output_path = self.output_directory / f"speech-{speech_id}.wav"
        return raw_path, output_path

    async def _normalize(self, raw_path: Path, output_path: Path) -> Path:
        """Normalize speech loudness while keeping a safe fallback if FFmpeg fails."""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            LOGGER.warning("FFmpeg not found; TTS loudness normalization is disabled.")
            raw_path.replace(output_path)
            return output_path

        audio_filter = f"loudnorm=I={TTS_TARGET_LUFS}:TP={TTS_TRUE_PEAK_DB}:LRA={TTS_LRA}"
        process = await asyncio.create_subprocess_exec(
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(raw_path),
            "-af",
            audio_filter,
            "-ar",
            "48000",
            "-ac",
            "2",
            str(output_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode == 0 and output_path.is_file():
            raw_path.unlink(missing_ok=True)
            return output_path

        detail = stderr.decode("utf-8", errors="replace").strip()
        LOGGER.warning("TTS loudness normalization failed; using raw output: %s", detail or process.returncode)
        output_path.unlink(missing_ok=True)
        raw_path.replace(output_path)
        return output_path


class GeminiSpeechSynthesizer(_NormalizedSpeechSynthesizer):
    """Generate natural speech with the Gemini Interactions API and save normalized WAV."""

    def __init__(
        self,
        output_directory: Path,
        *,
        api_key: str,
        model: str = "gemini-3.1-flash-tts-preview",
        voice: str = "Leda",
        style: str = DEFAULT_GEMINI_STYLE,
        timeout_seconds: float = 45.0,
    ) -> None:
        super().__init__(output_directory)
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.style = style.strip()
        self.timeout_seconds = max(5.0, float(timeout_seconds))
        self._client = None

    async def synthesize(self, text: str) -> Path:
        value = _validated_text(text)
        raw_path, output_path = self._paths("gemini")
        try:
            pcm = await asyncio.wait_for(
                asyncio.to_thread(self._generate_pcm, value),
                timeout=self.timeout_seconds,
            )
            self._write_pcm_wave(raw_path, pcm)
            LOGGER.info("Gemini TTS generated %s bytes with voice %s", len(pcm), self.voice)
            return await self._normalize(raw_path, output_path)
        except Exception:
            raw_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            raise

    def _generate_pcm(self, text: str) -> bytes:
        from google import genai

        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)

        prompt = text
        if self.style:
            prompt = f"{self.style}\n\nRead only this message aloud:\n{text}"

        interaction = self._client.interactions.create(
            model=self.model,
            input=prompt,
            # Pin the wire format instead of relying on the API's default audio MIME.
            # The writer below intentionally expects raw 24 kHz / mono / 16-bit PCM.
            response_format={"type": "audio", "mime_type": "audio/l16"},
            generation_config={
                "speech_config": [
                    {"voice": self.voice},
                ]
            },
        )
        output_audio = getattr(interaction, "output_audio", None)
        data = getattr(output_audio, "data", None)
        if isinstance(data, bytes):
            pcm = data
        elif isinstance(data, str):
            try:
                pcm = base64.b64decode(data, validate=True)
            except ValueError as error:
                raise RuntimeError("Gemini TTS 回傳了無法解析的音訊資料。") from error
        else:
            raise RuntimeError("Gemini TTS 沒有回傳可用的 output_audio。")
        if not pcm:
            raise RuntimeError("Gemini TTS 回傳了空白音訊。")
        return pcm

    @staticmethod
    def _write_pcm_wave(path: Path, pcm: bytes) -> None:
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(GEMINI_PCM_RATE)
            output.writeframes(pcm)


class KokoroSpeechSynthesizer(_NormalizedSpeechSynthesizer):
    """Optional fully-local Kokoro fallback. Imports the heavy stack only when it is used."""

    def __init__(
        self,
        output_directory: Path,
        *,
        voice: str = "zf_xiaoni",
        speed: float = 1.0,
    ) -> None:
        super().__init__(output_directory)
        self.voice = voice
        self.speed = max(0.5, min(float(speed), 2.0))
        self._pipeline = None

    async def synthesize(self, text: str) -> Path:
        value = _validated_text(text)
        raw_path, output_path = self._paths("kokoro")
        try:
            await asyncio.to_thread(self._generate_wave, value, raw_path)
            LOGGER.info("Kokoro TTS generated speech with voice %s", self.voice)
            return await self._normalize(raw_path, output_path)
        except Exception:
            raw_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            raise

    def _generate_wave(self, text: str, output_path: Path) -> None:
        try:
            import numpy as np
            import soundfile as sf
            from kokoro import KPipeline
        except ImportError as error:
            raise RuntimeError(
                "Kokoro 本地 TTS 尚未安裝；可執行 python -m pip install -e \".[tts-local]\"。"
            ) from error

        if self._pipeline is None:
            self._pipeline = KPipeline(lang_code="z")

        chunks = []
        for result in self._pipeline(text, voice=self.voice, speed=self.speed):
            audio = getattr(result, "audio", None)
            if audio is None and isinstance(result, (tuple, list)) and len(result) >= 3:
                audio = result[2]
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            array = np.asarray(audio, dtype=np.float32).reshape(-1)
            if array.size:
                chunks.append(array)
        if not chunks:
            raise RuntimeError("Kokoro TTS 沒有產生音訊。")
        sf.write(str(output_path), np.concatenate(chunks), KOKORO_PCM_RATE)


class SapiSpeechSynthesizer(_NormalizedSpeechSynthesizer):
    """Creates speech with the Windows SAPI voice as the final no-network fallback."""

    async def synthesize(self, text: str) -> Path:
        value = _validated_text(text)
        raw_path, output_path = self._paths("sapi")
        encoded_text = base64.b64encode(value.encode("utf-8")).decode("ascii")
        escaped_path = str(raw_path).replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$speaker.SetOutputToWaveFile('{escaped_path}'); "
            f"$text = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded_text}')); "
            "$speaker.Speak($text); $speaker.Dispose()"
        )
        process = await asyncio.create_subprocess_exec(
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0 or not raw_path.is_file():
            raw_path.unlink(missing_ok=True)
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Windows 語音合成失敗：{detail or '沒有產生 WAV 檔。'}")

        return await self._normalize(raw_path, output_path)


class FallbackSpeechSynthesizer:
    """Try speech backends in order without letting a quota/network failure silence the bot."""

    def __init__(self, providers: list[tuple[str, SpeechSynthesizer]]) -> None:
        if not providers:
            raise ValueError("At least one TTS provider is required")
        self.providers = providers

    async def synthesize(self, text: str) -> Path:
        value = _validated_text(text)
        errors: list[str] = []
        for name, provider in self.providers:
            try:
                path = await provider.synthesize(value)
            except Exception as error:
                errors.append(f"{name}: {error}")
                LOGGER.warning("TTS provider %s failed; trying fallback: %s", name, error)
                continue
            LOGGER.info("TTS provider selected: %s", name)
            return path
        raise RuntimeError("所有語音合成後端都失敗：" + " | ".join(errors))


def build_speech_synthesizer(
    output_directory: Path,
    *,
    provider: str = "auto",
    gemini_api_key: str | None = None,
    gemini_model: str = "gemini-3.1-flash-tts-preview",
    gemini_voice: str = "Leda",
    gemini_style: str = DEFAULT_GEMINI_STYLE,
    gemini_timeout_seconds: float = 45.0,
    kokoro_enabled: bool = True,
    kokoro_voice: str = "zf_xiaoni",
    kokoro_speed: float = 1.0,
) -> FallbackSpeechSynthesizer:
    """Build the configured TTS chain.

    auto: Gemini -> optional Kokoro -> Windows SAPI
    gemini: Gemini -> Windows SAPI
    kokoro: Kokoro -> Windows SAPI
    sapi: Windows SAPI only
    """
    requested = provider.strip().lower() or "auto"
    if requested not in {"auto", "gemini", "kokoro", "sapi"}:
        raise ValueError("TTS_PROVIDER must be one of: auto, gemini, kokoro, sapi")

    providers: list[tuple[str, SpeechSynthesizer]] = []
    if requested in {"auto", "gemini"} and gemini_api_key:
        providers.append(
            (
                "gemini",
                GeminiSpeechSynthesizer(
                    output_directory,
                    api_key=gemini_api_key,
                    model=gemini_model,
                    voice=gemini_voice,
                    style=gemini_style,
                    timeout_seconds=gemini_timeout_seconds,
                ),
            )
        )
    if requested in {"auto", "kokoro"} and kokoro_enabled:
        providers.append(
            (
                "kokoro",
                KokoroSpeechSynthesizer(
                    output_directory,
                    voice=kokoro_voice,
                    speed=kokoro_speed,
                ),
            )
        )
    providers.append(("sapi", SapiSpeechSynthesizer(output_directory)))
    return FallbackSpeechSynthesizer(providers)


class WindowsSpeechSynthesizer:
    """Backward-compatible facade used by existing commands.

    The old class name is retained so every existing voice call automatically gets
    the new provider chain without touching the command/voice routing code.
    """

    def __init__(self, output_directory: Path) -> None:
        self._delegate = build_speech_synthesizer(
            output_directory,
            provider=os.getenv("TTS_PROVIDER", "auto"),
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip() or None,
            gemini_model=(
                os.getenv("TTS_GEMINI_MODEL", "gemini-3.1-flash-tts-preview").strip()
                or "gemini-3.1-flash-tts-preview"
            ),
            gemini_voice=os.getenv("TTS_GEMINI_VOICE", "Leda").strip() or "Leda",
            gemini_style=os.getenv("TTS_GEMINI_STYLE", DEFAULT_GEMINI_STYLE).strip(),
            gemini_timeout_seconds=float(os.getenv("TTS_GEMINI_TIMEOUT_SECONDS", "45")),
            kokoro_enabled=_env_enabled("TTS_KOKORO_ENABLED", True),
            kokoro_voice=os.getenv("TTS_KOKORO_VOICE", "zf_xiaoni").strip() or "zf_xiaoni",
            kokoro_speed=float(os.getenv("TTS_KOKORO_SPEED", "1.0")),
        )

    @property
    def providers(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self._delegate.providers)

    async def synthesize(self, text: str) -> Path:
        return await self._delegate.synthesize(text)
