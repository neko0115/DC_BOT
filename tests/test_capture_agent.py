from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from discord_ai_assistant.capture_agent.profiles import CaptureProfile, ProfileStore, ScreenBounds
from discord_ai_assistant.capture_agent.slash import install_capture_agent_commands
from discord_ai_assistant.slash_groups import GroupedSlashCommands


class CaptureProfileTests(unittest.TestCase):
    def test_normalized_roi_scales_to_current_screen(self) -> None:
        profile = CaptureProfile(
            id="p1",
            name="LifeAfter",
            game="LifeAfter",
            screen=ScreenBounds(0, 0, 1920, 1080),
            roi_x=0.1,
            roi_y=0.5,
            roi_width=0.3,
            roi_height=0.2,
        )
        self.assertEqual(
            profile.pixel_roi(ScreenBounds(0, 0, 2560, 1440)),
            {"x": 256, "y": 720, "width": 768, "height": 288},
        )

    def test_store_syncs_active_profile_into_meeting_config(self) -> None:
        from discord_ai_assistant.capture_agent import profiles as module

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            identity = root / "agent.json"
            profile_file = root / "profiles.json"
            meeting_config = root / "meeting.json"
            default_config = root / "default.json"
            default_config.write_text(json.dumps({"chat": {"enabled": False}, "audio": {}, "output": {}}), encoding="utf-8")
            with (
                mock.patch.object(module, "IDENTITY_PATH", identity),
                mock.patch.object(module, "PROFILES_PATH", profile_file),
                mock.patch.object(module, "MEETING_CONFIG_PATH", meeting_config),
                mock.patch.object(module, "MEETING_DEFAULT_CONFIG_PATH", default_config),
            ):
                store = ProfileStore()
                profile = CaptureProfile(
                    id="p1",
                    name="Dorm-PC",
                    game="LifeAfter",
                    screen=ScreenBounds(0, 0, 1920, 1080),
                    roi_x=0.25,
                    roi_y=0.5,
                    roi_width=0.5,
                    roi_height=0.25,
                )
                store.upsert(profile)
                store.sync_meeting_config(ScreenBounds(0, 0, 2560, 1440))
                payload = json.loads(meeting_config.read_text(encoding="utf-8"))
                self.assertTrue(payload["chat"]["enabled"])
                self.assertEqual(payload["chat"]["x"], 640)
                self.assertEqual(payload["chat"]["y"], 720)
                self.assertEqual(payload["chat"]["width"], 1280)
                self.assertEqual(payload["chat"]["height"], 360)
                self.assertEqual(payload["chat"]["profile_id"], "p1")


class CaptureSlashTests(unittest.TestCase):
    def test_capture_commands_are_added_to_meeting_group(self) -> None:
        class FakeCore:
            bot = object()

        grouped = GroupedSlashCommands(FakeCore())
        install_capture_agent_commands(grouped)
        names = {command.name for command in grouped.meeting.commands}
        self.assertTrue({"agent", "calibrate", "profiles", "profile"}.issubset(names))


if __name__ == "__main__":
    unittest.main()
