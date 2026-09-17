from __future__ import annotations

import unittest
from datetime import time
from unittest.mock import patch

from discord_ai_assistant.ai.persona import BASE_PERSONA_INSTRUCTION, WorkloadMood
from discord_ai_assistant.ai.social import SocialParticipant


class WorkloadMoodTests(unittest.TestCase):
    def test_persona_has_requested_identity(self) -> None:
        self.assertIn("墨染雪", BASE_PERSONA_INSTRUCTION)
        self.assertIn("墨玲", BASE_PERSONA_INSTRUCTION)
        self.assertIn("喵", BASE_PERSONA_INSTRUCTION)

    def test_mood_becomes_tired_after_repeated_requests(self) -> None:
        mood = WorkloadMood()
        with patch("discord_ai_assistant.ai.persona.time.monotonic", return_value=100.0):
            for _ in range(4):
                instruction = mood.instruction_for(1, record_request=True)

        self.assertIn("疲憊", instruction)

    def test_mood_recovers_after_its_window(self) -> None:
        mood = WorkloadMood(window_seconds=10)
        with patch("discord_ai_assistant.ai.persona.time.monotonic", return_value=100.0):
            for _ in range(4):
                mood.instruction_for(1, record_request=True)
        with patch("discord_ai_assistant.ai.persona.time.monotonic", return_value=111.0):
            instruction = mood.instruction_for(1, record_request=False)

        self.assertIn("精神很好", instruction)

    def test_quiet_hours_support_an_overnight_window(self) -> None:
        self.assertTrue(SocialParticipant._time_is_in_window(time(23, 30), time(22), time(7)))
        self.assertTrue(SocialParticipant._time_is_in_window(time(6, 30), time(22), time(7)))
        self.assertFalse(SocialParticipant._time_is_in_window(time(12), time(22), time(7)))

    def test_comfort_signals_require_clear_distress_language(self) -> None:
        self.assertTrue(SocialParticipant.is_comfort_signal("我今天心情不好，真的好累"))
        self.assertFalse(SocialParticipant.is_comfort_signal("今天下雨，出門記得帶傘"))

    def test_crisis_signals_are_detected_separately(self) -> None:
        self.assertTrue(SocialParticipant.is_crisis_signal("我真的不想活了"))
        self.assertFalse(SocialParticipant.is_crisis_signal("我今天心情不好"))
