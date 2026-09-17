from __future__ import annotations

import logging
import os
from pathlib import Path

from discord_ai_assistant.ai.key_pool import FailoverGeminiClient, load_gemini_api_keys
from discord_ai_assistant.voice.synthesis import (
    DEFAULT_GEMINI_STYLE,
    FallbackSpeechSynthesizer,
    GeminiSpeechSynthesizer,
    KokoroSpeechSynthesizer,
    SapiSpeechSynthesizer,
    _validated_text,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_GEMINI_TTS_MODELS = (
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
)


def load_gemini_tts_models(primary_model: str | None = None) -> tuple[str, ...]:
    raw = os.getenv("TTS_GEMINI_MODELS", "").strip()
    if raw:
        models = tuple(part.strip() for part in raw.split(",") if part.strip())
    else:
        models = DEFAULT_GEMINI_TTS_MODELS
    primary = (primary_model or "").strip()
    if primary and primary not in models:
        models = (*models, primary)
    return tuple(dict.fromkeys(models))


class FailoverGeminiSpeechSynthesizer(GeminiSpeechSynthesizer):
    """One Gemini TTS model that shares the same API-key failover rules as chat."""

    def __init__(
        self,
        output_directory: Path,
        *,
        api_keys: tuple[str, ...],
        model: str,
        voice: str,
        style: str,
        timeout_seconds: float,
    ) -> None:
        super().__init__(
            output_directory,
            api_key=api_keys[0],
            model=model,
            voice=voice,
            style=style,
            timeout_seconds=timeout_seconds,
        )
        self._client = FailoverGeminiClient(api_keys)


class MultiModelGeminiSpeechSynthesizer:
    """Try free Gemini TTS models in order before leaving the Gemini provider."""

    def __init__(
        self,
        output_directory: Path,
        *,
        api_keys: tuple[str, ...],
        models: tuple[str, ...],
        voice: str,
        style: str,
        timeout_seconds: float,
    ) -> None:
        if not models:
            raise ValueError("At least one Gemini TTS model is required")
        self.backends = tuple(
            (
                model,
                FailoverGeminiSpeechSynthesizer(
                    output_directory,
                    api_keys=api_keys,
                    model=model,
                    voice=voice,
                    style=style,
                    timeout_seconds=timeout_seconds,
                ),
            )
            for model in models
        )

    async def synthesize(self, text: str) -> Path:
        value = _validated_text(text)
        errors: list[str] = []
        for model, backend in self.backends:
            try:
                result = await backend.synthesize(value)
            except Exception as error:
                errors.append(f"{model}: {error}")
                LOGGER.warning("Gemini TTS model %s failed; trying next model: %s", model, error)
                continue
            LOGGER.info("Gemini TTS model selected: %s", model)
            return result
        raise RuntimeError("所有 Gemini TTS 模型都失敗：" + " | ".join(errors))


class ResilientWindowsSpeechSynthesizer:
    """Drop-in speech facade with model/key rotation before local/provider fallback."""

    def __init__(
        self,
        output_directory: Path,
        *,
        primary_api_key: str | None,
        provider: str = "auto",
        gemini_model: str = "gemini-3.1-flash-tts-preview",
        gemini_voice: str = "Leda",
        gemini_style: str = DEFAULT_GEMINI_STYLE,
        gemini_timeout_seconds: float = 45.0,
        kokoro_enabled: bool = True,
        kokoro_voice: str = "zf_xiaoni",
        kokoro_speed: float = 1.0,
    ) -> None:
        requested = provider.strip().lower() or "auto"
        if requested not in {"auto", "gemini", "kokoro", "sapi"}:
            raise ValueError("TTS_PROVIDER must be one of: auto, gemini, kokoro, sapi")

        providers = []
        api_keys = load_gemini_api_keys(primary_api_key)
        if requested in {"auto", "gemini"} and api_keys:
            providers.append(
                (
                    "gemini",
                    MultiModelGeminiSpeechSynthesizer(
                        output_directory,
                        api_keys=api_keys,
                        models=load_gemini_tts_models(gemini_model),
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
        self._delegate = FallbackSpeechSynthesizer(providers)

    @property
    def providers(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self._delegate.providers)

    async def synthesize(self, text: str) -> Path:
        return await self._delegate.synthesize(text)
