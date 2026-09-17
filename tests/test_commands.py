from __future__ import annotations

import unittest

from discord_ai_assistant.commands import (
    is_sensitive_memory,
    parse_track_ids,
    voice_chat_announcement,
    youtube_track_from_info,
)


class CommandHelperTests(unittest.TestCase):
    def test_track_ids_keep_the_entered_order(self) -> None:
        self.assertEqual(parse_track_ids("12, 5 18"), [12, 5, 18])

    def test_track_ids_reject_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            parse_track_ids("12, hello")

    def test_youtube_search_entry_uses_canonical_video_url(self) -> None:
        track = youtube_track_from_info(
            {
                "entries": [
                    {
                        "title": "The Box",
                        "webpage_url": "https://www.youtube.com/watch?v=UNZqm3dxd2w",
                    }
                ]
            },
            requested_by=12,
            fallback_url="The Box Roddy Ricch",
        )

        self.assertIsNotNone(track)
        assert track is not None
        self.assertEqual(track.title, "The Box")
        self.assertEqual(track.original_name, "https://www.youtube.com/watch?v=UNZqm3dxd2w")
        self.assertIsNone(track.stream_url)

    def test_voice_chat_announcement_uses_display_name_and_text(self) -> None:
        self.assertEqual(voice_chat_announcement("布丁", "今天坐公車很暈"), "布丁說今天坐公車很暈")

    def test_voice_chat_announcement_rejects_empty_or_excessive_text(self) -> None:
        self.assertIsNone(voice_chat_announcement("布丁", "   \n"))
        self.assertIsNone(voice_chat_announcement("布丁", "很累" * 100))

    def test_sensitive_memories_are_rejected(self) -> None:
        self.assertTrue(is_sensitive_memory("我的 API Key 是 abc"))
        self.assertFalse(is_sensitive_memory("我偏好搖滾樂"))
