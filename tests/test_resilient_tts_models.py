from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from discord_ai_assistant.voice.resilient_synthesis import (
    MultiModelGeminiSpeechSynthesizer,
    ResilientWindowsSpeechSynthesizer,
    load_gemini_tts_models,
)


class _FailingSpeech:
    async def synthesize(self, text: str) -> Path:
        raise RuntimeError("quota")


class _WorkingSpeech:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> Path:
        self.calls.append(text)
        return self.path


class ResilientTTSModelTests(unittest.TestCase):
    def test_default_tts_models_include_31_then_25(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            models = load_gemini_tts_models("gemini-3.1-flash-tts-preview")
        self.assertEqual(
            models,
            ("gemini-3.1-flash-tts-preview", "gemini-2.5-flash-preview-tts"),
        )

    def test_env_tts_model_chain_preserves_legacy_as_last_resort(self) -> None:
        with patch.dict(
            os.environ,
            {"TTS_GEMINI_MODELS": "model-a,model-b"},
            clear=True,
        ):
            models = load_gemini_tts_models("legacy-model")
        self.assertEqual(models, ("model-a", "model-b", "legacy-model"))

    def test_multimodel_tts_uses_second_model_after_first_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "speech.wav"
            output.write_bytes(b"wav")
            working = _WorkingSpeech(output)
            synthesizer = MultiModelGeminiSpeechSynthesizer.__new__(MultiModelGeminiSpeechSynthesizer)
            synthesizer.backends = (
                ("first", _FailingSpeech()),
                ("second", working),
            )

            result = asyncio.run(synthesizer.synthesize("測試"))

        self.assertEqual(result, output)
        self.assertEqual(working.calls, ["測試"])

    def test_outer_provider_names_remain_backward_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            synthesizer = ResilientWindowsSpeechSynthesizer(
                Path(directory),
                primary_api_key="fake-key",
                provider="auto",
                kokoro_enabled=True,
            )
        self.assertEqual(synthesizer.providers, ("gemini", "kokoro", "sapi"))


if __name__ == "__main__":
    unittest.main()
