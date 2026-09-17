from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from discord_ai_assistant.config import load_settings


class SettingsTests(unittest.TestCase):
    def test_dj_role_id_is_loaded_when_configured(self) -> None:
        environment = {
            "DISCORD_TOKEN": "test-token",
            "DISCORD_GUILD_ID": "123",
            "DJ_ROLE_NAME": "DJ",
            "DJ_ROLE_ID": "456",
            "MAX_UPLOAD_MB": "25",
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, environment, clear=True):
            settings = load_settings(Path(directory))

        self.assertEqual(settings.dj_role_id, 456)
        self.assertEqual(settings.dj_role_name, "DJ")
        self.assertEqual(settings.persona_channel_name, "墨雪的貓窩")
        self.assertEqual(settings.introduction_channel_name, "墨雪的自我介紹")
        self.assertEqual(settings.persona_timezone, "Asia/Taipei")
        self.assertFalse(settings.voice_recognition_enabled)
        self.assertEqual(settings.voice_recognition_model, "small")
        self.assertEqual(settings.voice_recognition_language, "zh")
        self.assertEqual(settings.voice_recognition_beam_size, 5)
        self.assertEqual(settings.voice_recognition_silence_seconds, 1.8)
        self.assertEqual(settings.voice_recognition_max_segment_seconds, 20)
        self.assertIsNone(settings.youtube_cookies_from_browser)
        self.assertIsNone(settings.youtube_cookies_file)
        self.assertIsNone(settings.youtube_po_token)

    def test_youtube_browser_cookie_setting_is_loaded(self) -> None:
        environment = {
            "DISCORD_TOKEN": "test-token",
            "YOUTUBE_COOKIES_FROM_BROWSER": "chrome",
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, environment, clear=True):
            settings = load_settings(Path(directory))

        self.assertEqual(settings.youtube_cookies_from_browser, "chrome")
