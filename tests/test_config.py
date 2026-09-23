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
        self.assertTrue(settings.tool_gateway_enabled)
        self.assertTrue(settings.tool_gateway_embedded)
        self.assertEqual(settings.tool_gateway_host, "127.0.0.1")
        self.assertEqual(settings.tool_gateway_port, 8765)
        self.assertEqual(settings.tool_gateway_refresh_seconds, 30)
        self.assertEqual(settings.tool_gateway_timeout_seconds, 60)
        self.assertIsNone(settings.tool_gateway_token)
        self.assertIsNone(settings.heartbeat_url)
        self.assertIsNone(settings.heartbeat_token)
        self.assertEqual(settings.heartbeat_interval_seconds, 60)
        self.assertEqual(settings.heartbeat_timeout_seconds, 8)

    def test_heartbeat_settings_can_be_enabled(self) -> None:
        environment = {
            "DISCORD_TOKEN": "test-token",
            "MOXUE_HEARTBEAT_URL": "https://moxueneko.com/api/heartbeat",
            "MOXUE_HEARTBEAT_TOKEN": "heartbeat-secret",
            "MOXUE_HEARTBEAT_INTERVAL_SECONDS": "45",
            "MOXUE_HEARTBEAT_TIMEOUT_SECONDS": "6",
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, environment, clear=True):
            settings = load_settings(Path(directory))

        self.assertEqual(settings.heartbeat_url, "https://moxueneko.com/api/heartbeat")
        self.assertEqual(settings.heartbeat_token, "heartbeat-secret")
        self.assertEqual(settings.heartbeat_interval_seconds, 45)
        self.assertEqual(settings.heartbeat_timeout_seconds, 6)

    def test_heartbeat_url_requires_token(self) -> None:
        environment = {
            "DISCORD_TOKEN": "test-token",
            "MOXUE_HEARTBEAT_URL": "https://moxueneko.com/api/heartbeat",
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(RuntimeError, "MOXUE_HEARTBEAT_TOKEN"):
                load_settings(Path(directory))

    def test_youtube_browser_cookie_setting_is_loaded(self) -> None:
        environment = {
            "DISCORD_TOKEN": "test-token",
            "YOUTUBE_COOKIES_FROM_BROWSER": "chrome",
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, environment, clear=True):
            settings = load_settings(Path(directory))

        self.assertEqual(settings.youtube_cookies_from_browser, "chrome")

    def test_tool_gateway_values_can_be_overridden(self) -> None:
        environment = {
            "DISCORD_TOKEN": "test-token",
            "TOOL_GATEWAY_ENABLED": "true",
            "TOOL_GATEWAY_EMBEDDED": "false",
            "TOOL_GATEWAY_HOST": "localhost",
            "TOOL_GATEWAY_PORT": "9001",
            "TOOL_GATEWAY_REFRESH_SECONDS": "12",
            "TOOL_GATEWAY_TIMEOUT_SECONDS": "45",
            "TOOL_GATEWAY_TOKEN": "secret",
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, environment, clear=True):
            settings = load_settings(Path(directory))

        self.assertTrue(settings.tool_gateway_enabled)
        self.assertFalse(settings.tool_gateway_embedded)
        self.assertEqual(settings.tool_gateway_host, "localhost")
        self.assertEqual(settings.tool_gateway_port, 9001)
        self.assertEqual(settings.tool_gateway_refresh_seconds, 12)
        self.assertEqual(settings.tool_gateway_timeout_seconds, 45)
        self.assertEqual(settings.tool_gateway_token, "secret")
