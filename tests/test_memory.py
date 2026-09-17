from __future__ import annotations

import unittest

from discord_ai_assistant.ai.memory import is_passive_memory_candidate, parse_memory_drafts


class PassiveMemoryTests(unittest.TestCase):
    def test_candidate_filter_ignores_small_talk_and_accepts_habits(self) -> None:
        self.assertTrue(is_passive_memory_candidate("我最近在玩 DCS，週末通常會飛 F-16"))
        self.assertFalse(is_passive_memory_candidate("笑死"))
        self.assertFalse(is_passive_memory_candidate("https://example.com"))

    def test_parser_keeps_only_safe_supported_memory_drafts(self) -> None:
        drafts = parse_memory_drafts(
            '{"memories":['
            '{"category":"興趣","content":"喜歡飛行模擬"},'
            '{"category":"健康","content":"有焦慮症"},'
            '{"category":"偏好","content":"API Key 是 secret"}'
            ']}'
        )

        self.assertEqual([(draft.category, draft.content) for draft in drafts], [("興趣", "喜歡飛行模擬")])
