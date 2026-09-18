from __future__ import annotations

import unittest

from discord_ai_assistant.ai.knowledge_help import (
    QUESTION_COMPLEXITY_SEARCH,
    classify_question_complexity,
    is_freshness_dependent,
    is_knowledge_gap_candidate,
)


class KnowledgeHelpFreshnessTests(unittest.TestCase):
    def test_current_update_question_uses_search(self) -> None:
        content = "CS2 今天更新了什麼？"
        self.assertTrue(is_freshness_dependent(content))
        self.assertEqual(classify_question_complexity(content), QUESTION_COMPLEXITY_SEARCH)

    def test_static_update_definition_does_not_force_search(self) -> None:
        content = "遊戲更新是什麼意思？"
        self.assertFalse(is_freshness_dependent(content))
        self.assertNotEqual(classify_question_complexity(content), QUESTION_COMPLEXITY_SEARCH)

    def test_temporal_discourse_about_update_meaning_does_not_force_search(self) -> None:
        for content in ("今天想了解遊戲更新是什麼意思？", "今天我想知道遊戲更新是什麼意思？"):
            with self.subTest(content=content):
                self.assertFalse(is_freshness_dependent(content))
                self.assertNotEqual(classify_question_complexity(content), QUESTION_COMPLEXITY_SEARCH)

    def test_current_update_and_release_predicates_keep_search(self) -> None:
        for content in ("CS2 今天更新了什麼？", "遊戲目前發布了什麼？",
                        "遊戲現在釋出了什麼？", "遊戲最近發佈了什麼？"):
            with self.subTest(content=content):
                self.assertTrue(is_freshness_dependent(content))
                self.assertEqual(classify_question_complexity(content), QUESTION_COMPLEXITY_SEARCH)

    def test_temporal_marker_requires_a_nearby_update_event_in_the_same_clause(self) -> None:
        for content in ("今天想學 Python 是什麼", "遊戲發布是什麼意思？",
                        "今天" + "a" * 13 + "更新了什麼？", "今天休息。遊戲更新是什麼意思？",
                        "今天休息。遊戲更新了什麼？"):
            with self.subTest(content=content):
                self.assertFalse(is_freshness_dependent(content))

    def test_typhoon_timing_question_without_question_mark_uses_search(self) -> None:
        content = "下一個颱風什麼時候來最靠近台北"
        self.assertTrue(is_knowledge_gap_candidate(content))
        self.assertTrue(is_freshness_dependent(content))
        self.assertEqual(classify_question_complexity(content), QUESTION_COMPLEXITY_SEARCH)

    def test_static_typhoon_explanation_does_not_force_search(self) -> None:
        content = "颱風為什麼會形成"
        self.assertTrue(is_knowledge_gap_candidate(content))
        self.assertFalse(is_freshness_dependent(content))


if __name__ == "__main__":
    unittest.main()
