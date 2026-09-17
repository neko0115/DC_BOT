from __future__ import annotations

import unittest

from discord_ai_assistant.lyrics_commands import LiveLyricsSession, LyricsCommands
from discord_ai_assistant.music.lyrics import LyricLine, LyricsResult


class LyricsOffsetTests(unittest.TestCase):
    def test_positive_offset_delays_lyric_position(self) -> None:
        session = LiveLyricsSession(channel_id=1, elapsed_seconds=12.5, offset_seconds=1.5)
        self.assertEqual(LyricsCommands._lyric_position(session), 11.0)

    def test_offset_delays_line_change_until_adjusted_time_reaches_timestamp(self) -> None:
        lyrics = LyricsResult(
            track_name="Example",
            artist_name="Artist",
            album_name="",
            duration=30.0,
            instrumental=False,
            plain_lyrics=None,
            synced_lyrics=None,
            lines=(
                LyricLine(10.0, "first"),
                LyricLine(20.0, "second"),
            ),
            source="test",
        )
        session = LiveLyricsSession(
            channel_id=1,
            lyrics=lyrics,
            elapsed_seconds=10.5,
            offset_seconds=1.0,
        )
        cog = object.__new__(LyricsCommands)
        self.assertIsNone(cog._current_line(session))

        session.elapsed_seconds = 11.0
        self.assertEqual(cog._current_line(session), 0)

    def test_negative_offset_can_advance_lyrics_when_source_is_late(self) -> None:
        session = LiveLyricsSession(channel_id=1, elapsed_seconds=8.0, offset_seconds=-0.75)
        self.assertEqual(LyricsCommands._lyric_position(session), 8.75)


if __name__ == "__main__":
    unittest.main()
