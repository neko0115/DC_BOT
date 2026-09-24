from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from discord_ai_assistant.meeting_upload import (
    LocalMeetingTranscriber,
    build_meeting_report_prompt,
    format_meeting_timestamp,
    format_timestamped_transcript,
    is_supported_meeting_audio,
    normalize_report_timecode_particle_order,
    resolve_meeting_schedule,
    strip_report_timecodes,
)


class MeetingUploadTests(unittest.TestCase):
    def test_meeting_audio_validation(self) -> None:
        self.assertTrue(is_supported_meeting_audio("weekly.mp3", 65 * 1024 * 1024))
        self.assertTrue(is_supported_meeting_audio("weekly.M4A", 1_024))
        self.assertFalse(is_supported_meeting_audio("weekly.exe", 1_024))
        self.assertFalse(is_supported_meeting_audio("weekly.mp3", 0))
        self.assertFalse(is_supported_meeting_audio("weekly.mp3", 251 * 1024 * 1024))

    def test_timestamped_transcript_marks_low_confidence(self) -> None:
        segments = [
            {"start": 1.2, "end": 4.8, "text": "第一項已完成。", "confidence": 0.91},
            {"start": 65.0, "end": 70.0, "text": "這句需要確認。", "confidence": 0.42},
        ]
        text = format_timestamped_transcript(segments)
        self.assertIn("[00:00:01-00:00:05] 第一項已完成。", text)
        self.assertIn("[00:01:05-00:01:10] [低信心 0.42] 這句需要確認。", text)
        self.assertEqual(format_meeting_timestamp(3661.0), "01:01:01")

    def test_report_prompt_forbids_inventing_missing_facts(self) -> None:
        transcript = "[00:03:00-00:03:10] 下星期再測試，目前還沒實測。"
        prompt = build_meeting_report_prompt("明日之後週會", transcript)
        self.assertIn("明日之後週會", prompt)
        self.assertIn(transcript, prompt)
        self.assertIn("不得自行補", prompt)
        self.assertIn("未指定／待確認", prompt)
        self.assertIn("尚未驗證", prompt)
        self.assertIn("低信心", prompt)

    def test_meeting_schedule_defaults_to_upload_date_and_2030(self) -> None:
        uploaded = datetime(2026, 9, 24, 8, 15, tzinfo=timezone.utc)
        date_text, time_text = resolve_meeting_schedule(uploaded)
        self.assertEqual(date_text, "2026-09-24")
        self.assertEqual(time_text, "20:30:00")

    def test_meeting_schedule_accepts_user_override_and_normalizes_seconds(self) -> None:
        uploaded = datetime(2026, 9, 24, 8, 15, tzinfo=timezone.utc)
        self.assertEqual(
            resolve_meeting_schedule(uploaded, "2026-09-23", "19:45"),
            ("2026-09-23", "19:45:00"),
        )
        with self.assertRaisesRegex(ValueError, "meeting_time"):
            resolve_meeting_schedule(uploaded, None, "25:99")

    def test_review_timecodes_keep_meow_before_evidence_and_publish_strips_evidence(self) -> None:
        draft = "- 決議已確認（[00:02:26-00:02:37]、[00:04:18-00:04:29]）喵。"
        normalized = normalize_report_timecode_particle_order(draft)
        self.assertIn("決議已確認喵 （[00:02:26-00:02:37]、[00:04:18-00:04:29]）。", normalized)
        published = strip_report_timecodes(normalized)
        self.assertEqual(published, "- 決議已確認喵。")

    def test_local_transcriber_preserves_segment_timestamps(self) -> None:
        class FakeModel:
            def transcribe(self, path: str, **kwargs):
                self_test.assertTrue(path.endswith("meeting.mp3"))
                self_test.assertEqual(kwargs["language"], "zh")
                self_test.assertEqual(kwargs["beam_size"], 5)
                self_test.assertTrue(kwargs["vad_filter"])
                return (
                    iter(
                        [
                            SimpleNamespace(
                                start=0.25,
                                end=2.5,
                                text=" 這是一段測試。 ",
                                avg_logprob=-0.1,
                                no_speech_prob=0.05,
                            ),
                            SimpleNamespace(
                                start=2.5,
                                end=4.0,
                                text=" 第二段。 ",
                                avg_logprob=-0.2,
                                no_speech_prob=0.1,
                            ),
                        ]
                    ),
                    SimpleNamespace(language="zh"),
                )

        self_test = self
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcriber = LocalMeetingTranscriber(root)
            with mock.patch.object(transcriber, "_get_model", return_value=FakeModel()):
                config = {
                    "whisper_model": "small",
                    "whisper_language": "zh",
                    "whisper_beam_size": 5,
                    "whisper_initial_prompt": "明日之後週會",
                }
                result = transcriber.transcribe(root / "meeting.mp3", config)

            self.assertEqual(result.model, "small")
            self.assertEqual(result.language, "zh")
            self.assertEqual(result.duration_seconds, 4.0)
            self.assertEqual(len(result.segments), 2)
            self.assertEqual(result.segments[0]["start"], 0.25)
            self.assertEqual(result.segments[1]["end"], 4.0)
            self.assertIn("這是一段測試。", result.text)
            self.assertGreater(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
