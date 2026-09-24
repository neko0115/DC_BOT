from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from discord_ai_assistant.meeting_feedback import (
    FeedbackCandidate,
    MeetingFeedbackStore,
    infer_partial_asr_candidates,
    merge_candidates,
    parse_ai_candidates,
    parse_explicit_corrections,
)
from discord_ai_assistant.storage.database import Database


class MeetingFeedbackParsingTests(unittest.TestCase):
    def test_explicit_arrow_corrections_become_high_confidence_asr_aliases(self) -> None:
        candidates = parse_explicit_corrections(
            "卡西 3 -> 海域 3 | 海域關卡名稱\n蝴蝶無人機 → 晶蝶無人機：遊戲無人機"
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].kind, "asr_alias")
        self.assertEqual(candidates[0].source_text, "卡西 3")
        self.assertEqual(candidates[0].canonical_term, "海域 3")
        self.assertEqual(candidates[0].aliases, ("卡西 3",))
        self.assertEqual(candidates[0].confidence, 1.0)
        self.assertIn("遊戲無人機", candidates[1].explanation)

    def test_ai_candidates_are_bounded_and_invalid_facts_are_not_accepted_as_kinds(self) -> None:
        payload = {
            "candidates": [
                {
                    "kind": "asr_alias",
                    "source_text": "信息",
                    "canonical_term": "新興",
                    "explanation": "玩家名稱",
                    "aliases": [],
                    "confidence": 0.8,
                },
                {
                    "kind": "meeting_fact",
                    "source_text": "",
                    "canonical_term": "",
                    "explanation": "今晚 21:30 打王",
                    "aliases": [],
                    "confidence": 1.0,
                },
                {
                    "kind": "report_preference",
                    "source_text": "",
                    "canonical_term": "",
                    "explanation": "所有週會報都要保留時間碼",
                    "aliases": [],
                    "confidence": 0.7,
                },
            ]
        }
        candidates = parse_ai_candidates(json.dumps(payload, ensure_ascii=False))
        self.assertEqual([item.kind for item in candidates], ["asr_alias", "report_preference"])
        self.assertEqual(candidates[0].aliases, ("信息",))

    def test_partial_full_article_replacement_still_becomes_selectable_asr_candidate(self) -> None:
        original = "# 週會報\n- 阿霧負責聯絡，阿霧晚點確認，阿霧再回報。\n"
        revised = "# 週會報\n- 阿鳴負責聯絡，阿霧晚點確認，阿霧再回報。\n"
        candidates = infer_partial_asr_candidates(original, revised, "阿霧負責聯絡，阿霧晚點確認。")
        self.assertTrue(any(
            item.source_text == "阿霧" and item.canonical_term == "阿鳴"
            for item in candidates
        ))

    def test_explicit_human_mapping_wins_when_merging_same_ai_candidate(self) -> None:
        explicit = parse_explicit_corrections("信息 -> 新興")
        ai = [
            FeedbackCandidate(
                kind="asr_alias",
                source_text="信息",
                canonical_term="新興",
                explanation="AI 猜測",
                aliases=("信息",),
                confidence=0.6,
            )
        ]
        merged = merge_candidates(explicit, ai)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].confidence, 1.0)
        self.assertEqual(merged[0].explanation, "")


class MeetingFeedbackStoreTests(unittest.TestCase):
    def test_selection_is_revision_scoped_immutable_and_uses_existing_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database = Database(root / "assistant.sqlite3")
            self.addCleanup(database.close)
            store = MeetingFeedbackStore(database, root)
            revisions = [store.create_revision(
                review_id=f"review-{guild_id}", guild_id=guild_id, profile_key=None,
                audio_sha256="audio", title="週會", correction_text="修訂",
                original_report="草稿", created_by=99,
            ) for guild_id in (10, 11)]
            candidate = FeedbackCandidate("report_preference", "", "", "保留時間碼")
            selected, unselected = store.add_candidates(revisions[0], [candidate, candidate])
            foreign = store.add_candidates(revisions[1], [candidate])[0]
            schema = database.connection.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()

            store.select_candidates(revisions[0], [selected, selected, foreign, -1])
            store.select_candidates(revisions[0], [unselected])

            self.assertEqual([i for i, _ in store.candidates(revisions[0], "approved")], [selected])
            self.assertEqual([i for i, _ in store.candidates(revisions[0], "skipped")], [unselected])
            self.assertEqual([i for i, _ in store.candidates(revisions[1], "pending")], [foreign])
            self.assertEqual(database.connection.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall(), schema)
            database.close()

    def test_reviewed_asr_mapping_creates_timestamped_training_examples_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database = Database(root / "assistant.sqlite3")
            store = MeetingFeedbackStore(database, root)
            revision_id = store.create_revision(
                review_id="review-1",
                guild_id=10,
                profile_key="lifeafter",
                audio_sha256="abc123",
                title="明日之後週會",
                correction_text="卡西 3 -> 海域 3",
                original_report="原草稿",
                created_by=99,
            )
            candidate = FeedbackCandidate(
                kind="asr_alias",
                source_text="卡西 3",
                canonical_term="海域 3",
                aliases=("卡西 3",),
            )
            store.add_candidates(revision_id, [candidate])
            inserted = store.record_asr_examples(
                revision_id=revision_id,
                guild_id=10,
                profile_key="lifeafter",
                audio_sha256="abc123",
                segments=[
                    {"start": 10.0, "end": 15.0, "text": "確認卡西 3 已被擊敗"},
                    {"start": 15.0, "end": 20.0, "text": "其他內容"},
                ],
                candidates=[candidate],
                created_by=99,
            )
            self.assertEqual(inserted, 1)
            self.assertEqual(store.training_stats(10, "lifeafter"), {"examples": 1, "recordings": 1})

            path, count = store.export_training_manifest(10, "lifeafter")
            self.assertEqual(count, 1)
            line = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(line["start_seconds"], 10.0)
            self.assertEqual(line["original_text"], "確認卡西 3 已被擊敗")
            self.assertEqual(line["corrected_text"], "確認海域 3 已被擊敗")
            self.assertIsNone(line["audio_path"])
            database.close()

    def test_report_preferences_and_approved_examples_are_profile_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database = Database(root / "assistant.sqlite3")
            store = MeetingFeedbackStore(database, root)
            revision_id = store.create_revision(
                review_id="review-2",
                guild_id=10,
                profile_key="lifeafter",
                audio_sha256="def456",
                title="明日之後週會",
                correction_text="以後都要保留時間碼",
                original_report="原草稿",
                created_by=99,
            )
            store.add_report_preference(
                guild_id=10,
                profile_key="lifeafter",
                instruction="所有決議都保留時間碼",
                revision_id=revision_id,
                created_by=99,
            )
            store.add_report_example(
                guild_id=10,
                profile_key="lifeafter",
                review_id="review-2",
                title="明日之後週會",
                report_text="# 核准週會報",
                approved_by=99,
            )

            self.assertEqual(store.report_preferences(10, "lifeafter"), ["所有決議都保留時間碼"])
            self.assertEqual(store.recent_report_examples(10, "lifeafter"), ["# 核准週會報"])
            self.assertEqual(store.report_preferences(11, "lifeafter"), [])
            self.assertEqual(store.recent_report_examples(10, "dcs"), [])
            database.close()


if __name__ == "__main__":
    unittest.main()
