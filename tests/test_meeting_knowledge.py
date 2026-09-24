from __future__ import annotations

import unittest

from discord_ai_assistant.app_knowledge import (
    AppKnowledgeProfile,
    AppKnowledgeSnapshot,
    AppKnowledgeTerm,
)
from discord_ai_assistant.meeting_upload_glossary import (
    _apply_snapshot_corrections,
    _discord_chunks,
    _snapshot_initial_prompt,
)


class MeetingKnowledgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.snapshot = AppKnowledgeSnapshot(
            profile=AppKnowledgeProfile(
                key="lifeafter",
                display_name="明日之後",
                description="遊戲週會",
                hints=("明日之後", "高校"),
            ),
            terms=(
                AppKnowledgeTerm(
                    term="晶蝶無人機",
                    explanation="無人機名稱。",
                    aliases=("蝴蝶無人機",),
                ),
                AppKnowledgeTerm(
                    term="迷霧的 boss",
                    explanation="Boss 戰相關名稱。",
                    aliases=("迷糊的 Force",),
                ),
            ),
        )

    def test_snapshot_prompt_biases_whisper_with_canonical_and_alias_terms(self) -> None:
        prompt = _snapshot_initial_prompt("繁體中文逐字稿。", self.snapshot)
        self.assertIsNotNone(prompt)
        assert prompt is not None
        self.assertIn("晶蝶無人機", prompt)
        self.assertIn("蝴蝶無人機", prompt)
        self.assertIn("迷霧的 boss", prompt)
        self.assertIn("迷糊的 Force", prompt)

    def test_snapshot_corrections_only_replace_explicit_aliases(self) -> None:
        corrected, audit = _apply_snapshot_corrections(
            "蝴蝶無人機很好用，迷糊的 Force 等一下打，第四網還要研究。",
            self.snapshot,
        )
        self.assertIn("晶蝶無人機", corrected)
        self.assertIn("迷霧的 boss", corrected)
        self.assertIn("第四網", corrected)
        self.assertEqual(len(audit), 2)

    def test_approved_alias_correction_replaces_every_occurrence_at_runtime(self) -> None:
        snapshot = AppKnowledgeSnapshot(
            profile=self.snapshot.profile,
            terms=(
                AppKnowledgeTerm(
                    term="阿鳴",
                    explanation="人工核准的參與者名稱。",
                    aliases=("阿霧",),
                ),
            ),
        )
        corrected, audit = _apply_snapshot_corrections("阿霧先說，阿霧再確認，最後還是阿霧。", snapshot)
        self.assertEqual(corrected, "阿鳴先說，阿鳴再確認，最後還是阿鳴。")
        self.assertEqual(audit, [{"from": "阿霧", "to": "阿鳴", "count": 3}])

    def test_discord_chunks_stay_under_message_limit(self) -> None:
        text = ("段落 A\n" * 500) + "\n\n" + ("段落 B\n" * 500)
        chunks = _discord_chunks(text, limit=500)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(0 < len(chunk) <= 500 for chunk in chunks))
        self.assertEqual("".join(chunk.replace("\n", "") for chunk in chunks).count("段落"), 1000)


if __name__ == "__main__":
    unittest.main()
