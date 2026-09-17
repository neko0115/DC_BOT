from __future__ import annotations

import asyncio
import unittest

from discord_ai_assistant.voice.recognition import (
    PerSpeakerSegmentBuffer,
    RecognitionTestPublisher,
    VoiceRecognitionSink,
    WhisperTranscriber,
)


class VoiceSegmentBufferTests(unittest.TestCase):
    def test_speakers_are_buffered_separately(self) -> None:
        buffer = PerSpeakerSegmentBuffer()
        buffer.add(1, b"alice", now=0)
        buffer.add(2, b"bob", now=0.1)

        segments = buffer.flush_idle(now=2)

        self.assertEqual({segment.speaker_id for segment in segments}, {1, 2})

    def test_long_speech_is_split_into_bounded_segments(self) -> None:
        buffer = PerSpeakerSegmentBuffer(max_segment_seconds=10)
        buffer.add(1, b"first", now=0)

        segments = buffer.add(1, b"second", now=10)

        self.assertEqual(len(segments), 1)
        self.assertEqual(bytes(segments[0].pcm_frames), b"first")

    def test_discord_pcm_is_converted_to_whisper_sample_rate(self) -> None:
        # 0.1 seconds of 48 kHz stereo, signed 16-bit silence.
        pcm = b"\x00\x00\x00\x00" * 4_800

        samples = WhisperTranscriber._to_whisper_audio(pcm)

        self.assertEqual(samples.dtype.name, "float32")
        self.assertEqual(samples.size, 1_600)
        self.assertTrue((samples == 0).all())

    def test_sink_can_be_instantiated_and_cleaned_up(self) -> None:
        sink = VoiceRecognitionSink(PerSpeakerSegmentBuffer(), lambda segment: None)

        sink.cleanup()

    def test_sink_reports_received_audio_by_speaker(self) -> None:
        received: list[tuple[int, int]] = []

        class User:
            id = 123

        class VoiceData:
            pcm = b"audio"

        sink = VoiceRecognitionSink(
            PerSpeakerSegmentBuffer(),
            lambda segment: None,
            lambda speaker_id, pcm_bytes: received.append((speaker_id, pcm_bytes)),
        )

        sink.write(User(), VoiceData())

        self.assertEqual(received, [(123, 5)])

    def test_recognition_result_uses_display_name_without_a_mention(self) -> None:
        class Member:
            display_name = "墨雪測試者"

        class Channel:
            name = "測試"

            def __init__(self) -> None:
                self.message: str | None = None
                self.allowed_mentions = None

            async def send(self, message: str, *, allowed_mentions: object) -> None:
                self.message = message
                self.allowed_mentions = allowed_mentions

        class Guild:
            def __init__(self, channel: Channel) -> None:
                self.text_channels = [channel]

            @staticmethod
            def get_member(member_id: int) -> Member | None:
                return Member() if member_id == 123 else None

        channel = Channel()
        published = asyncio.run(RecognitionTestPublisher("測試").publish(Guild(channel), 123, "你好"))

        self.assertTrue(published)
        self.assertEqual(channel.message, "語音辨識測試 | 墨雪測試者：你好")
        self.assertNotIn("<@", channel.message or "")
        self.assertIsNotNone(channel.allowed_mentions)
