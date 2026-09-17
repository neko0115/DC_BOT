from __future__ import annotations

import unittest

from discord_ai_assistant.slash_groups import (
    GroupedSlashCommands,
    LEGACY_GROUPED_TOP_LEVEL_COMMANDS,
)


class SlashGroupTests(unittest.TestCase):
    def test_expected_top_level_groups_exist(self) -> None:
        self.assertEqual(GroupedSlashCommands.meeting.name, "meeting")
        self.assertEqual(GroupedSlashCommands.comms.name, "comms")
        self.assertEqual(GroupedSlashCommands.voice.name, "voice")
        self.assertEqual(GroupedSlashCommands.music.name, "music")
        self.assertEqual(GroupedSlashCommands.library.name, "library")
        self.assertEqual(GroupedSlashCommands.playlist.name, "playlist")
        self.assertEqual(GroupedSlashCommands.persona.name, "persona")
        self.assertEqual(GroupedSlashCommands.memory.name, "memory")

    def test_meeting_group_contains_control_surface(self) -> None:
        names = {command.name for command in GroupedSlashCommands.meeting.commands}
        self.assertEqual(
            names,
            {"status", "start", "stop", "transcript", "devices", "output", "ocr"},
        )

    def test_voice_group_matches_requested_user_facing_actions(self) -> None:
        names = {command.name for command in GroupedSlashCommands.voice.commands}
        self.assertEqual(names, {"ask", "say", "read", "stopread", "join", "leave"})

    def test_common_commands_are_not_hidden_from_top_level(self) -> None:
        for name in ("play", "skip", "queue", "pause", "resume", "ask"):
            self.assertNotIn(name, LEGACY_GROUPED_TOP_LEVEL_COMMANDS)

    def test_migrated_noisy_commands_are_hidden_from_top_level(self) -> None:
        for name in (
            "comfort_auto",
            "voice_recognition_tuning",
            "playlist_create",
            "library_add",
            "memory_passive",
            "repeat_one",
        ):
            self.assertIn(name, LEGACY_GROUPED_TOP_LEVEL_COMMANDS)


if __name__ == "__main__":
    unittest.main()
