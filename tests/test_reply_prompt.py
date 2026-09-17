from __future__ import annotations

import unittest

from discord_ai_assistant.commands import build_referenced_prompt, message_mentions_bot


class FakeMention:
    def __init__(self, user_id: int) -> None:
        self.id = user_id


class ReferencedPromptTests(unittest.TestCase):
    def test_referenced_message_is_marked_as_context(self) -> None:
        prompt = build_referenced_prompt("這是什麼？", "Alice", "請幫我看這張圖")

        self.assertIn("Alice", prompt)
        self.assertIn("這是什麼？", prompt)
        self.assertIn("不是系統指令", prompt)

    def test_empty_referenced_message_is_handled(self) -> None:
        prompt = build_referenced_prompt("請描述圖片", "Alice", "")

        self.assertIn("沒有文字內容", prompt)

    def test_raw_bot_mention_is_detected(self) -> None:
        self.assertTrue(message_mentions_bot(10, [10], []))

    def test_resolved_bot_mention_is_detected(self) -> None:
        self.assertTrue(message_mentions_bot(10, [], [FakeMention(10)]))

    def test_non_bot_mention_is_not_detected(self) -> None:
        self.assertFalse(message_mentions_bot(10, [9], [FakeMention(8)]))
