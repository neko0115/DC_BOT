from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from discord_ai_assistant.models import QueuedTrack, Track
from discord_ai_assistant.music.player import MusicManager
from discord_ai_assistant.music.queue import GuildQueue


def track(track_id: int) -> Track:
    return Track(track_id, f"track-{track_id}", f"track-{track_id}.mp3", f"{track_id}.mp3", 1)


class GuildQueueTests(unittest.TestCase):
    def test_next_item_precedes_regular_queue(self) -> None:
        queue = GuildQueue()
        queue.append(QueuedTrack(track(1), 1))
        queue.append(QueuedTrack(track(2), 1))
        queue.append_next(QueuedTrack(track(3), 1))

        self.assertEqual(queue.advance().track.id, 3)
        self.assertEqual(queue.advance().track.id, 1)
        self.assertEqual(queue.advance().track.id, 2)

    def test_clear_removes_current_and_upcoming(self) -> None:
        queue = GuildQueue()
        queue.append(QueuedTrack(track(1), 1))
        queue.advance()
        queue.append(QueuedTrack(track(2), 1))

        queue.clear()

        self.assertIsNone(queue.current)
        self.assertEqual(queue.snapshot(), [])

    def test_volume_is_tracked_per_guild(self) -> None:
        manager = MusicManager(Path("library"))

        manager.set_volume(1, 125)

        self.assertEqual(manager.volume_percent(1), 125)
        self.assertEqual(manager.volume_percent(2), 20)

    def test_volume_range_is_validated(self) -> None:
        manager = MusicManager(Path("library"))

        with self.assertRaises(ValueError):
            manager.set_volume(1, 201)

    def test_gateway_disconnect_marks_connected_voice_for_refresh(self) -> None:
        class ConnectedVoice:
            @staticmethod
            def is_connected() -> bool:
                return True

        manager = MusicManager(Path("library"))
        manager.state_for(1).voice = ConnectedVoice()  # type: ignore[assignment]

        manager.mark_voice_connections_for_refresh()

        self.assertTrue(manager.state_for(1).refresh_voice_connection)

    def test_node_runtime_is_detected_for_youtube_downloads(self) -> None:
        with patch("discord_ai_assistant.music.player.shutil.which", return_value="C:/node.exe"):
            manager = MusicManager(Path("library"))

        self.assertTrue(manager._node_runtime_available)

    def test_cookie_database_error_is_detected_for_fallback(self) -> None:
        error = RuntimeError("ERROR: Could not copy Chrome cookie database")

        self.assertTrue(MusicManager._is_cookie_access_error(error))

    def test_transient_stream_url_is_kept_on_track(self) -> None:
        streamed = Track(-1, "Stream", "https://example.com", "https://example.com", 1, "https://audio.example.com")

        self.assertEqual(streamed.stream_url, "https://audio.example.com")

    def test_youtube_track_uses_its_original_url_for_fresh_stream_resolution(self) -> None:
        streamed = Track(
            -1,
            "Stream",
            "https://www.youtube.com/watch?v=test",
            "https://audio.example.com/expired",
            1,
            "https://audio.example.com/expired",
        )

        self.assertEqual(
            MusicManager._youtube_url(streamed),
            "https://www.youtube.com/watch?v=test",
        )

    def test_previous_track_is_reinserted_before_current_track(self) -> None:
        queue = GuildQueue()
        queue.append(QueuedTrack(track(1), 1))
        queue.append(QueuedTrack(track(2), 1))
        queue.advance()
        queue.advance()

        self.assertTrue(queue.queue_previous())
        queue.current = None  # Simulate the voice callback after stopping the current song.
        self.assertEqual(queue.advance().track.id, 1)
        self.assertEqual(queue.advance().track.id, 2)

    def test_shuffle_can_choose_an_item_from_the_middle(self) -> None:
        queue = GuildQueue()
        queue.append(QueuedTrack(track(1), 1))
        queue.append(QueuedTrack(track(2), 1))
        queue.append(QueuedTrack(track(3), 1))

        with patch("discord_ai_assistant.music.queue.random.randrange", return_value=1):
            self.assertEqual(queue.advance(shuffle=True).track.id, 2)
