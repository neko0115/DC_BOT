from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from discord_ai_assistant.voice.synthesis import (
    FallbackSpeechSynthesizer,
    GeminiSpeechSynthesizer,
    WindowsSpeechSynthesizer,
    _validated_text,
    build_speech_synthesizer,
)


class _FailingProvider:
    def __init__(self, calls: list[str], name: str) -> None:
        self.calls = calls
        self.name = name

    async def synthesize(self, text: str) -> Path:
        self.calls.append(self.name)
        raise RuntimeError("expected failure")


class _WorkingProvider:
    def __init__(self, calls: list[str], name: str, path: Path) -> None:
        self.calls = calls
        self.name = name
        self.path = path

    async def synthesize(self, text: str) -> Path:
        self.calls.append(self.name)
        return self.path


class _FakeGeminiInteractions:
    def __init__(self, pcm: bytes) -> None:
        self.pcm = pcm
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return SimpleNamespace(
            output_audio=SimpleNamespace(data=base64.b64encode(self.pcm).decode("ascii"))
        )


class SynthesisTests(unittest.TestCase):
    def test_validated_text_strips_and_rejects_empty(self) -> None:
        self.assertEqual(_validated_text("  你好  "), "你好")
        with self.assertRaises(ValueError):
            _validated_text("   ")

    def test_gemini_pcm_writer_creates_24khz_mono_16bit_wave(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            pcm = b"\x00\x00\x01\x00" * 16
            GeminiSpeechSynthesizer._write_pcm_wave(path, pcm)
            with wave.open(str(path), "rb") as source:
                self.assertEqual(source.getnchannels(), 1)
                self.assertEqual(source.getsampwidth(), 2)
                self.assertEqual(source.getframerate(), 24000)
                self.assertEqual(source.readframes(source.getnframes()), pcm)

    def test_gemini_request_uses_interactions_audio_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pcm = b"\x00\x00" * 32
            interactions = _FakeGeminiInteractions(pcm)
            synthesizer = GeminiSpeechSynthesizer(
                Path(directory),
                api_key="fake-key",
                voice="Leda",
                style="Speak softly.",
            )
            synthesizer._client = SimpleNamespace(interactions=interactions)
            result = synthesizer._generate_pcm("你好")
            self.assertEqual(result, pcm)
            assert interactions.kwargs is not None
            self.assertEqual(interactions.kwargs["model"], "gemini-3.1-flash-tts-preview")
            self.assertEqual(
                interactions.kwargs["response_format"],
                {"type": "audio", "mime_type": "audio/l16"},
            )
            self.assertIn("你好", str(interactions.kwargs["input"]))
            self.assertEqual(
                interactions.kwargs["generation_config"],
                {"speech_config": [{"voice": "Leda"}]},
            )

    def test_fallback_provider_tries_next_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ok.wav"
            output.write_bytes(b"ok")
            calls: list[str] = []
            synthesizer = FallbackSpeechSynthesizer(
                [
                    ("first", _FailingProvider(calls, "first")),
                    ("second", _WorkingProvider(calls, "second", output)),
                ]
            )
            result = asyncio.run(synthesizer.synthesize("測試"))
            self.assertEqual(result, output)
            self.assertEqual(calls, ["first", "second"])

    def test_auto_chain_prefers_gemini_then_kokoro_then_sapi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            synthesizer = build_speech_synthesizer(
                Path(directory),
                provider="auto",
                gemini_api_key="fake-key",
                kokoro_enabled=True,
            )
            self.assertEqual([name for name, _ in synthesizer.providers], ["gemini", "kokoro", "sapi"])

    def test_sapi_mode_disables_remote_and_local_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            synthesizer = build_speech_synthesizer(
                Path(directory),
                provider="sapi",
                gemini_api_key="fake-key",
                kokoro_enabled=True,
            )
            self.assertEqual([name for name, _ in synthesizer.providers], ["sapi"])

    def test_legacy_facade_uses_environment_chain(self) -> None:
        environment = {
            "GEMINI_API_KEY": "fake-key",
            "TTS_PROVIDER": "auto",
            "TTS_KOKORO_ENABLED": "true",
            "TTS_GEMINI_MODEL": "gemini-3.1-flash-tts-preview",
            "TTS_GEMINI_VOICE": "Leda",
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, environment, clear=True):
            synthesizer = WindowsSpeechSynthesizer(Path(directory))
            self.assertEqual(synthesizer.providers, ("gemini", "kokoro", "sapi"))


if __name__ == "__main__":
    unittest.main()
