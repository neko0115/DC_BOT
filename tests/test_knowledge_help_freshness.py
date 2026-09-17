from __future__ import annotations

import unittest

from discord_ai_assistant.ai.knowledge_help import (
    QUESTION_COMPLEXITY_SEARCH,
    classify_question_complexity,
    is_freshness_dependent,
    is_knowledge_gap_candidate,
)


class KnowledgeHelpFreshnessTests(unittest.TestCase):
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
