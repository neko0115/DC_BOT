from __future__ import annotations

import unittest
from pathlib import Path

import discord
import discord_ai_assistant.meeting_feedback_cog as feedback_module
from discord_ai_assistant.meeting_full_article_cog import (
    MEETING_REPORT_TIMEOUT_SECONDS,
    FullArticleMeetingReportReviewView,
    MeetingReportCommands,
    PendingFullArticleRevision,
    ResilientMeetingFeedbackDecisionView,
    full_revision_diff,
    normalize_full_revision_text,
)


class MeetingFullArticleReviewTests(unittest.TestCase):
    def test_full_revision_normalizes_newlines_and_preserves_whole_article(self) -> None:
        source = "# 週會報\r\n\r\n" + ("完整人工修正版內容。" * 40)
        result = normalize_full_revision_text(source)
        self.assertNotIn("\r", result)
        self.assertTrue(result.startswith("# 週會報\n\n"))
        self.assertIn("完整人工修正版內容。", result)

    def test_full_revision_rejects_short_chat_so_pending_review_does_not_eat_normal_messages(self) -> None:
        with self.assertRaisesRegex(ValueError, "完整修正版內容太短"):
            normalize_full_revision_text("這個等等再改")

    def test_diff_contains_original_and_human_replacement(self) -> None:
        original = "# 週會報\n- 確認卡西 3 已被擊敗\n- 推薦蝴蝶無人機\n"
        revised = "# 週會報\n- 確認海域 3 已被擊敗\n- 推薦晶蝶無人機\n"
        result = full_revision_diff(original, revised)
        self.assertIn("-- 確認卡西 3 已被擊敗", result)
        self.assertIn("+- 確認海域 3 已被擊敗", result)
        self.assertIn("-- 推薦蝴蝶無人機", result)
        self.assertIn("+- 推薦晶蝶無人機", result)

    def test_pending_full_article_revision_expires_after_ten_minutes(self) -> None:
        pending = PendingFullArticleRevision(
            report_path=Path("report.md"),
            review_path=Path("review.json"),
            title="明日之後週會",
            created_at_monotonic=100.0,
        )
        self.assertFalse(pending.expired(699.9))
        self.assertTrue(pending.expired(700.1))

    def test_feedback_cog_uses_full_article_review_view_for_every_review_round(self) -> None:
        self.assertIs(feedback_module.MeetingReportReviewView, FullArticleMeetingReportReviewView)


class MeetingFullArticleDiscordViewTests(unittest.IsolatedAsyncioTestCase):
    async def test_review_view_replaces_short_modal_button_with_full_article_button(self) -> None:
        view = FullArticleMeetingReportReviewView(
            object(),
            report_path=Path("report.md"),
            review_path=Path("review.json"),
            title="明日之後週會",
            requester_id=1,
        )
        labels = [item.label for item in view.children if isinstance(item, discord.ui.Button)]
        self.assertIn("提交整篇修正版", labels)
        self.assertNotIn("修正並重新整理", labels)
        full_button = next(
            item
            for item in view.children
            if isinstance(item, discord.ui.Button) and item.label == "提交整篇修正版"
        )
        self.assertEqual(full_button.custom_id, "meeting_full_article_revision")

    async def test_feedback_buttons_can_be_restored_after_generation_failure(self) -> None:
        view = ResilientMeetingFeedbackDecisionView(
            object(),
            revision_id=7,
            review_path=Path("review.json"),
            report_path=Path("report.md"),
            title="明日之後週會",
            correction_text="人工完整修正版" * 30,
            candidate_ids=[10, 11],
        )
        view._disable()
        self.assertTrue(view._handled)
        self.assertTrue(all(item.disabled for item in view.children if isinstance(item, discord.ui.Button)))

        view._restore_retry_state()

        self.assertFalse(view._handled)
        self.assertTrue(all(not item.disabled for item in view.children if isinstance(item, discord.ui.Button)))

    async def test_apply_learning_stays_disabled_when_there_are_no_candidates(self) -> None:
        view = ResilientMeetingFeedbackDecisionView(
            object(),
            revision_id=8,
            review_path=Path("review.json"),
            report_path=Path("report.md"),
            title="明日之後週會",
            correction_text="人工完整修正版" * 30,
            candidate_ids=[],
        )
        apply_button = next(
            item
            for item in view.children
            if isinstance(item, discord.ui.Button) and item.custom_id == "meeting_feedback_apply"
        )
        self.assertTrue(apply_button.disabled)

    async def test_meeting_report_generation_uses_long_task_specific_timeout(self) -> None:
        class FakeAI:
            def __init__(self) -> None:
                self.timeout_seconds: int | None = None
                self.request_kind: str | None = None

            async def social_reply_with_timeout(
                self,
                prompt: str,
                *,
                persona_instruction: str,
                timeout_seconds: int,
                request_kind: str,
            ) -> str:
                self.timeout_seconds = timeout_seconds
                self.request_kind = request_kind
                self.assert_payload = (prompt, persona_instruction)
                return "# 測試週會報"

        ai = FakeAI()
        cog = object.__new__(MeetingReportCommands)
        cog.ai = ai
        cog.feedback_store = None

        result = await cog._generate_report_with_feedback(
            title="明日之後週會",
            transcript="[00:00] 測試逐字稿",
            guild_id=1,
            snapshot=None,
        )

        self.assertEqual(result, "# 測試週會報")
        self.assertEqual(ai.timeout_seconds, MEETING_REPORT_TIMEOUT_SECONDS)
        self.assertEqual(ai.request_kind, "meeting-report")
        self.assertEqual(MEETING_REPORT_TIMEOUT_SECONDS, 240)


if __name__ == "__main__":
    unittest.main()
