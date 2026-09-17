from __future__ import annotations

import asyncio
import unittest

from discord_ai_assistant.models import Track
from discord_ai_assistant.music.lyrics import (
    LyricsProvider,
    current_line_index,
    infer_title_artist,
    parse_synced_lyrics,
    parse_youtube_json3,
    parse_youtube_vtt,
    search_candidates,
)


class _FakeLyricsProvider(LyricsProvider):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[tuple[str | None, str | None, str | None]] = []

    async def _request_records(
        self,
        *,
        track_name: str | None = None,
        artist_name: str | None = None,
        album_name: str | None = None,
        query: str | None = None,
    ) -> list[dict[str, object]]:
        self.requests.append((track_name, artist_name, query))
        if track_name == "後來" and artist_name == "劉若英":
            return [
                {
                    "trackName": "後來",
                    "artistName": "劉若英",
                    "albumName": "我等你",
                    "duration": 341,
                    "plainLyrics": "plain",
                    "syncedLyrics": "[00:01.00]line",
                    "instrumental": False,
                }
            ]
        return []


class _ExactLyricsProvider(LyricsProvider):
    def __init__(self) -> None:
        super().__init__()
        self.exact_calls = 0
        self.search_calls = 0

    async def _request_exact(self, metadata):  # type: ignore[no-untyped-def]
        self.exact_calls += 1
        self.asserted_metadata = metadata
        return {
            "trackName": "後來",
            "artistName": "劉若英",
            "albumName": "我等你",
            "duration": 341,
            "plainLyrics": "plain",
            "syncedLyrics": "[00:01.00]line",
            "instrumental": False,
        }

    async def _request_records(self, **kwargs):  # type: ignore[no-untyped-def]
        self.search_calls += 1
        return []


class LyricsTests(unittest.TestCase):
    def test_parse_synced_lyrics_sorts_and_supports_multiple_timestamps(self) -> None:
        lines = parse_synced_lyrics(
            "[00:10.50]second\n[00:02.00][00:04.00]first\n[ar:metadata]\n"
        )
        self.assertEqual(
            [(line.start_seconds, line.text) for line in lines],
            [(2.0, "first"), (4.0, "first"), (10.5, "second")],
        )

    def test_youtube_json3_is_converted_to_synced_lines(self) -> None:
        lines = parse_youtube_json3(
            '{"events":['
            '{"tStartMs":1250,"segs":[{"utf8":"first "},{"utf8":"line"}]},'
            '{"tStartMs":3500,"segs":[{"utf8":"second &amp; line"}]}'
            ']}'
        )
        self.assertEqual(
            [(line.start_seconds, line.text) for line in lines],
            [(1.25, "first line"), (3.5, "second & line")],
        )

    def test_youtube_vtt_is_converted_to_synced_lines(self) -> None:
        lines = parse_youtube_vtt(
            "WEBVTT\n\n00:01.000 --> 00:03.000\n<c>first line</c>\n\n"
            "00:03.500 --> 00:05.000\nsecond line\n"
        )
        self.assertEqual(
            [(line.start_seconds, line.text) for line in lines],
            [(1.0, "first line"), (3.5, "second line")],
        )

    def test_current_line_index_handles_intro_and_progress(self) -> None:
        lines = parse_synced_lyrics("[00:05.00]one\n[00:08.50]two\n")
        self.assertIsNone(current_line_index(lines, 0.0))
        self.assertEqual(current_line_index(lines, 5.0), 0)
        self.assertEqual(current_line_index(lines, 8.49), 0)
        self.assertEqual(current_line_index(lines, 8.5), 1)

    def test_infer_title_artist_removes_common_youtube_decorations(self) -> None:
        title, artist = infer_title_artist("Example Artist - Example Song (Official Music Video)")
        self.assertEqual(title, "Example Song")
        self.assertEqual(artist, "Example Artist")

    def test_infer_title_artist_handles_bilingual_cjk_music_video_title(self) -> None:
        title, artist = infer_title_artist("劉若英 René Liu 【後來 Later】 Official Music Video")
        self.assertEqual(title, "後來 Later")
        self.assertEqual(artist, "劉若英 René Liu")

    def test_search_candidates_include_language_specific_title_and_artist(self) -> None:
        candidates = search_candidates("劉若英 René Liu 【後來 Later】 Official Music Video")
        self.assertIn(("後來", "劉若英"), candidates)
        self.assertIn(("Later", "René Liu"), candidates)
        self.assertIn(("後來", None), candidates)

    def test_bilingual_title_retries_with_clean_cjk_pair(self) -> None:
        provider = _FakeLyricsProvider()
        result = asyncio.run(provider.fetch_for_title("劉若英 René Liu 【後來 Later】 Official Music Video"))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.track_name, "後來")
        self.assertEqual(result.artist_name, "劉若英")
        self.assertIn(("後來", "劉若英", None), provider.requests)

    def test_track_metadata_uses_lrclib_exact_before_search(self) -> None:
        provider = _ExactLyricsProvider()
        track = Track(
            id=-1,
            title="劉若英 René Liu 【後來 Later】 Official Music Video",
            original_name="https://www.youtube.com/watch?v=example",
            stored_name="https://www.youtube.com/watch?v=example",
            uploaded_by=1,
            track_name="後來",
            artist_name="劉若英",
            album_name="我等你",
            duration=341.0,
        )
        result = asyncio.run(provider.fetch_for_track(track))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.source, "LRCLIB exact")
        self.assertEqual(provider.exact_calls, 1)
        self.assertEqual(provider.search_calls, 0)

    def test_select_best_prefers_matching_synced_record(self) -> None:
        records = [
            {
                "trackName": "Completely Different",
                "artistName": "Someone",
                "albumName": "A",
                "duration": 200,
                "plainLyrics": "plain",
                "syncedLyrics": None,
                "instrumental": False,
            },
            {
                "trackName": "Example Song",
                "artistName": "Example Artist",
                "albumName": "B",
                "duration": 210,
                "plainLyrics": "plain",
                "syncedLyrics": "[00:01.00]hello",
                "instrumental": False,
            },
        ]
        result = LyricsProvider._select_best(records, "Example Song", "Example Artist")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.track_name, "Example Song")
        self.assertEqual(result.artist_name, "Example Artist")
        self.assertEqual(result.lines[0].start_seconds, 1.0)


if __name__ == "__main__":
    unittest.main()
