from __future__ import annotations

import sys
import unittest
from array import array
from collections import deque

import discord

from discord_ai_assistant.music.enhanced_player import PCM_FRAME_BYTES, DuckingAudioSource


class FakePCMSource(discord.AudioSource):
    def __init__(self, frames: list[bytes]) -> None:
        self.frames = deque(frames)
        self.cleaned = False

    def read(self) -> bytes:
        return self.frames.popleft() if self.frames else b""

    def is_opus(self) -> bool:
        return False

    def cleanup(self) -> None:
        self.cleaned = True


class FailingPCMSource(FakePCMSource):
    def __init__(self, message: str = "decoder failed") -> None:
        super().__init__([])
        self.message = message

    def read(self) -> bytes:
        raise RuntimeError(self.message)


def pcm_frame(value: int, samples: int = 8) -> bytes:
    data = array("h", [value] * samples)
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()


def first_sample(data: bytes) -> int:
    samples = array("h")
    samples.frombytes(data)
    if sys.byteorder != "little":
        samples.byteswap()
    return int(samples[0])


class DuckingAudioSourceTests(unittest.TestCase):
    def test_music_keeps_playing_and_ducks_only_while_speech_is_active(self) -> None:
        music = FakePCMSource([pcm_frame(1000), pcm_frame(1000), pcm_frame(1000)])
        speech = FakePCMSource([pcm_frame(2000)])
        mixer = DuckingAudioSource(music, volume=1.0, ducking_ratio=0.25)

        self.assertEqual(first_sample(mixer.read()), 1000)
        mixer.enqueue_speech(speech)
        self.assertEqual(first_sample(mixer.read()), 2250)
        self.assertEqual(first_sample(mixer.read()), 1000)
        self.assertTrue(speech.cleaned)

    def test_master_volume_applies_to_music_and_speech(self) -> None:
        music = FakePCMSource([pcm_frame(1000)])
        speech = FakePCMSource([pcm_frame(2000)])
        mixer = DuckingAudioSource(music, volume=0.5, ducking_ratio=0.5)
        mixer.enqueue_speech(speech)

        # 1000 * 0.5 * 0.5 + 2000 * 0.5 = 1250
        self.assertEqual(first_sample(mixer.read()), 1250)

    def test_mixer_clamps_pcm_instead_of_overflowing(self) -> None:
        music = FakePCMSource([pcm_frame(30000)])
        speech = FakePCMSource([pcm_frame(30000)])
        mixer = DuckingAudioSource(music, volume=1.0, ducking_ratio=1.0)
        mixer.enqueue_speech(speech)

        self.assertEqual(first_sample(mixer.read()), 32767)

    def test_cleanup_releases_music_and_pending_speech(self) -> None:
        music = FakePCMSource([pcm_frame(1000)])
        speech = FakePCMSource([pcm_frame(2000)])
        mixer = DuckingAudioSource(music, volume=1.0)
        mixer.enqueue_speech(speech)

        mixer.cleanup()

        self.assertTrue(music.cleaned)
        self.assertTrue(speech.cleaned)
        self.assertEqual(mixer.read(), b"")

    def test_failed_speech_overlay_is_dropped_without_stopping_music(self) -> None:
        music = FakePCMSource([pcm_frame(1000), pcm_frame(1000)])
        speech = FailingPCMSource("bad speech")
        mixer = DuckingAudioSource(music, volume=1.0)
        mixer.enqueue_speech(speech)

        frame = mixer.read()

        self.assertEqual(first_sample(frame), 1000)
        self.assertTrue(speech.cleaned)
        self.assertIsNone(mixer.last_error)
        self.assertEqual(first_sample(mixer.read()), 1000)

    def test_failed_music_source_is_recorded_for_manager_retry(self) -> None:
        music = FailingPCMSource("bad music")
        mixer = DuckingAudioSource(music, volume=1.0)

        self.assertEqual(mixer.read(), b"")
        self.assertIsInstance(mixer.last_error, RuntimeError)
        self.assertTrue(mixer.music_finished)

    def test_short_pcm_tail_is_padded_to_one_discord_frame(self) -> None:
        music = FakePCMSource([pcm_frame(1000, samples=3)])
        mixer = DuckingAudioSource(music, volume=1.0)

        frame = mixer.read()

        self.assertEqual(len(frame), PCM_FRAME_BYTES)
        self.assertEqual(first_sample(frame), 1000)

    def test_silent_speech_does_not_duck_music(self) -> None:
        music = FakePCMSource([pcm_frame(1000)])
        speech = FakePCMSource([pcm_frame(0)])
        mixer = DuckingAudioSource(music, volume=1.0, ducking_ratio=0.1)
        mixer.enqueue_speech(speech)

        self.assertEqual(first_sample(mixer.read()), 1000)


if __name__ == "__main__":
    unittest.main()
