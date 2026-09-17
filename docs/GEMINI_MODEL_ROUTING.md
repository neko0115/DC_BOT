# Gemini model routing policy

This project routes Gemini requests by workload instead of sending every request through one model.

## Why

Google Gemini API limits are project/model specific. Multiple API keys in the same project do not create independent project quota. Key failover therefore remains useful for invalid credentials and genuinely independent projects, while model failover handles model-specific rate/quota/service failures.

Do not hard-code public RPM/TPM/RPD guesses into the bot. Check the project's active limits in Google AI Studio when exact quota values matter.

## Default workload routes

| Workload | Default order |
| --- | --- |
| Chat | `gemini-3.6-flash` → `gemini-3.5-flash` → `gemini-3.5-flash-lite` → `gemma-4-31b-it` |
| Reasoning | `gemini-3.7-flash` → `gemini-3.6-flash` → `gemini-3.5-flash` → `gemini-2.5-pro` → `gemma-4-31b-it` |
| Tools | `gemini-3.7-flash` → `gemini-3.6-flash` → `gemini-3.5-flash` → `gemini-2.5-flash` |
| Vision | `gemini-3.7-flash` → `gemini-3.6-flash` → `gemini-3.5-flash` → `gemini-2.5-flash` |
| Passive social | `gemini-3.5-flash-lite` → `gemini-3.1-flash-lite` → `gemma-4-26b-a4b-it` |
| Memory extraction | `gemini-3.1-flash-lite` → `gemini-3.5-flash-lite` → `gemma-4-26b-a4b-it` |
| Meeting reports | `gemini-3.7-flash` → `gemini-3.6-flash` → `gemini-3.5-flash` → `gemini-2.5-pro` → `gemma-4-31b-it` |
| Free-tier Search | `gemini-2.5-flash` → `gemini-2.5-flash-lite` |

The meeting workload preserves the existing 240-second task timeout and meeting revision recovery behavior.

## Search boundary

Free-tier Google Search is deliberately isolated to the Gemini 2.5 Flash family. Gemini 3.x free token inference does not imply free Search grounding. The 2.5 Flash and Flash-Lite Search allowance is a shared pool, so an explicit shared Search quota error blocks the Search workload instead of repeatedly cycling between models.

Search-only proactive assistance continues to expose only Google Search and no Discord/music/meeting/memory actions.

## Failure behavior

- `401/403` and key-specific failures remain handled by the API-key pool.
- A retryable model `429`, model-unavailable error, timeout, or selected `5xx` can move the fresh request to the next model.
- Daily quota signals may cool a model until the next Pacific midnight.
- Stateful `previous_interaction_id` continuations are pinned to both their originating API key/project and their originating model. A failed continuation is never replayed on another model.
- An unavailable Search pool fails closed. Proactive search stays silent rather than inventing current facts.

## TTS

The normal TTS order remains:

1. configured local GPT-SoVITS worker pool;
2. Gemini TTS provider, internally trying `gemini-3.1-flash-tts-preview` then `gemini-2.5-flash-preview-tts`;
3. optional local Kokoro;
4. Windows SAPI.

The outer provider name remains `gemini` for backward compatibility.

## Configuration

All workload chains can be overridden with the `GEMINI_MODELS_*` environment variables documented in `.env.example`. `GEMINI_MODEL` remains a legacy non-search fallback so an existing installation does not break when this router is introduced.
