from __future__ import annotations

import unittest

from discord_ai_assistant.lyrics_commands import LiveLyricsSession, LyricsCommands
from discord_ai_assistant.music.lyrics import LyricsResult, LyricLine


class LyricsCommandTimingTests(unittest.TestCase):
    def _session(self, elapsed: float) -> LiveLyricsSession:
        return LiveLyricsSession(
            channel_id=123,
            track_key="track",
            elapsed_seconds=elapsed,
            lyrics=LyricsResult(
                track_name="Pets",
                artist_name="Porno For Pyros",
                album_name="",
                duration=220.0,
                instrumental=False,
                plain_lyrics=None,
                synced_lyrics="[02:10.00]You make great pets",
                lines=(LyricLine(130.0, "You make great pets"),),
                source="LRCLIB search",
            ),
        )

    def test_playing_render_key_changes_each_visible_second(self) -> None:
        commands = LyricsCommands.__new__(LyricsCommands)
        session = self._session(130.1)
        first = commands._render_key(session, 0, "playing")
        session.elapsed_seconds = 131.0
        second = commands._render_key(session, 0, "playing")
        self.assertNotEqual(first, second)

    def test_paused_render_key_does_not_tick_with_time(self) -> None:
        commands = LyricsCommands.__new__(LyricsCommands)
        session = self._session(130.1)
        first = commands._render_key(session, 0, "paused")
        session.elapsed_seconds = 131.0
        second = commands._render_key(session, 0, "paused")
        self.assertEqual(first, second)

    def test_footer_time_format_is_stable(self) -> None:
        self.assertEqual(LyricsCommands._format_time(130.9), "2:10")
        self.assertEqual(LyricsCommands._format_time(131.0), "2:11")


if __name__ == "__main__":
    unittest.main()
