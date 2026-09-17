from __future__ import annotations

import asyncio
import unittest

from discord_ai_assistant.ai.gemini import describe_gemini_error


class FakeApiError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GeminiErrorTests(unittest.TestCase):
    def test_invalid_key_message_is_actionable(self) -> None:
        error = FakeApiError("API_KEY_INVALID", 400)

        self.assertIn("沒有認出", describe_gemini_error(error))
        self.assertNotIn("API_KEY_INVALID", describe_gemini_error(error))

    def test_legacy_schema_message_has_upgrade_command(self) -> None:
        error = FakeApiError("The legacy Interactions API schema is no longer supported", 400)

        self.assertIn("pip install -U -e .", describe_gemini_error(error))

    def test_rate_limit_message_is_actionable(self) -> None:
        self.assertIn("額度", describe_gemini_error(FakeApiError("quota", 429)))

    def test_timeout_message_is_actionable(self) -> None:
        self.assertIn("Gemini", describe_gemini_error(asyncio.TimeoutError()))
