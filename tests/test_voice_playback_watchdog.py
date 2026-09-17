from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

from discord_ai_assistant.models import QueuedTrack, Track
from discord_ai_assistant.voice.playback_watchdog import VoicePlaybackWatchdog


class _FakeQueue:
    def __init__(self, current: QueuedTrack | None, upcoming: list[QueuedTrack] | None = None) -> None:
        self.current = current
        self._upcoming = list(upcoming or [])

    def snapshot(self) -> list[QueuedTrack]:
        return list(self._upcoming)


class _FakeVoice:
    def __init__(self, *, connected: bool = True, playing: bool = False, paused: bool = False) -> None:
        self.connected = connected
        self.playing = playing
        self.paused = paused

    def is_connected(self) -> bool:
        return self.connected

    def is_playing(self) -> bool:
        return self.playing

    def is_paused(self) -> bool:
        return self.paused


class _FakeProcess:
    def __init__(self, return_code: int | None) -> None:
        self.return_code = return_code

    def poll(self) -> int | None:
        return self.return_code


class _FakeMusic:
    def __init__(self, state: SimpleNamespace) -> None:
        self.state = state
        self.stop_calls = 0
        self.start_calls = 0

    def active_guild_ids(self) -> list[int]:
        return [1]

    def state_for(self, guild_id: int) -> SimpleNamespace:
        return self.state

    @staticmethod
    def _ffmpeg_return_code(source: object) -> int | None:
        original = getattr(source, "original", None)
        process = getattr(original, "_process", None)
        return process.poll() if process is not None else None

    def _stop_playback(self, voice: _FakeVoice) -> None:
        self.stop_calls += 1
        voice.playing = False

    async def _start_if_idle(self, guild_id: int) -> bool:
        self.start_calls += 1
        return True


def _track(path: Path | None = None, *, tts: bool = True) -> QueuedTrack:
    track = Track(
        id=-2 if tts else 1,
        title="墨雪語音回覆" if tts else "song",
        original_name=path.name if path else "audio.wav",
        stored_name=path.name if path else "audio.wav",
        uploaded_by=1,
        audio_path=str(path) if path else None,
        delete_after_play=tts,
    )
    return QueuedTrack(track, requested_by=1)


def _source(return_code: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        original=SimpleNamespace(_process=_FakeProcess(return_code))
    )


class VoicePlaybackWatchdogTests(unittest.IsolatedAsyncioTestCase):
    async def test_restarts_upcoming_queue_when_voice_is_idle(self) -> None:
        current = _track(tts=False)
        state = SimpleNamespace(
            voice=_FakeVoice(playing=False),
            queue=_FakeQueue(current, [_track(tts=False)]),
            source=_source(None),
            refresh_voice_connection=False,
        )
        music = _FakeMusic(state)
        watchdog = VoicePlaybackWatchdog(object(), music)  # type: ignore[arg-type]

        await watchdog.check_once(now=10)

        self.assertEqual(music.start_calls, 1)
        self.assertEqual(music.stop_calls, 0)

    async def test_stops_voice_when_ffmpeg_has_already_exited(self) -> None:
        state = SimpleNamespace(
            voice=_FakeVoice(playing=True),
            queue=_FakeQueue(_track(tts=False)),
            source=_source(0),
            refresh_voice_connection=False,
        )
        music = _FakeMusic(state)
        watchdog = VoicePlaybackWatchdog(object(), music)  # type: ignore[arg-type]

        await watchdog.check_once(now=10)

        self.assertEqual(music.stop_calls, 1)
        self.assertFalse(state.voice.playing)

    async def test_stops_tts_that_exceeds_wav_duration_plus_grace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tts.wav"
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(8000)
                audio.writeframes(b"\x00\x00" * 8000)

            state = SimpleNamespace(
                voice=_FakeVoice(playing=True),
                queue=_FakeQueue(_track(path)),
                source=_source(None),
                refresh_voice_connection=False,
            )
            music = _FakeMusic(state)
            watchdog = VoicePlaybackWatchdog(
                object(),  # type: ignore[arg-type]
                music,  # type: ignore[arg-type]
                tts_stall_grace_seconds=3,
            )

            await watchdog.check_once(now=0)
            self.assertEqual(music.stop_calls, 0)
            await watchdog.check_once(now=5)

        self.assertEqual(music.stop_calls, 1)

    async def test_repeated_stall_arms_voice_reconnect(self) -> None:
        state = SimpleNamespace(
            voice=_FakeVoice(playing=True),
            queue=_FakeQueue(_track(tts=False)),
            source=_source(0),
            refresh_voice_connection=False,
        )
        music = _FakeMusic(state)
        watchdog = VoicePlaybackWatchdog(object(), music)  # type: ignore[arg-type]

        await watchdog.check_once(now=10)
        self.assertFalse(state.refresh_voice_connection)

        state.voice.playing = True
        state.source = _source(1)
        await watchdog.check_once(now=40)

        self.assertTrue(state.refresh_voice_connection)
        self.assertEqual(music.stop_calls, 2)


if __name__ == "__main__":
    unittest.main()
