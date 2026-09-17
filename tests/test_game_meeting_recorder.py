from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from discord_ai_assistant.tool_gateway.registry import ToolRegistry


class GameMeetingRecorderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.registry = ToolRegistry(cls.project_root / "tools")
        cls.registry.reload()
        cls.tool = cls.registry.get("game_meeting_recorder")

    def test_tool_loads_without_optional_capture_dependencies(self) -> None:
        self.assertNotIn("game_meeting_recorder", self.registry.errors)
        self.assertIsNotNone(self.tool)
        status = asyncio.run(self.registry.invoke("game_meeting_recorder", "status", {}, {}))
        self.assertIsInstance(status, dict)
        self.assertFalse(status["recording"])
        self.assertIn("dependencies", status)
        self.assertIn("data/game-meeting-recorder", status["config_path"].replace("\\", "/"))

    def test_manifest_exposes_expected_actions(self) -> None:
        assert self.tool is not None
        actions = {item["name"] for item in self.tool.public_payload()["actions"]}
        self.assertTrue(
            {
                "status",
                "list_audio_devices",
                "configure_audio",
                "configure_chat",
                "configure_output",
                "test_chat_capture",
                "start_session",
                "stop_session",
                "get_transcript",
            }.issubset(actions)
        )

    def test_transcript_format_preserves_source_time_and_confidence(self) -> None:
        assert self.tool is not None
        recorder = self.tool.module.RECORDER
        transcript = recorder._format_transcript(
            [
                {
                    "timestamp": "2026-08-19T23:15:32+08:00",
                    "source": "voice",
                    "text": "下一階段先處理北部。",
                    "confidence": 0.88,
                },
                {
                    "timestamp": "2026-08-19T23:16:04+08:00",
                    "source": "chat",
                    "text": "等第二隊回來再打",
                    "confidence": 0.94,
                },
            ]
        )
        self.assertIn("[23:15:32] [VOICE 0.88]", transcript)
        self.assertIn("[23:16:04] [CHAT 0.94]", transcript)
        self.assertIn("等第二隊回來再打", transcript)

    def test_invalid_chat_roi_is_rejected(self) -> None:
        assert self.tool is not None
        recorder = self.tool.module.RECORDER
        config = recorder._load_config()
        config["chat"]["width"] = 5
        with self.assertRaises(ValueError):
            recorder._validate_config(config)


if __name__ == "__main__":
    unittest.main()
