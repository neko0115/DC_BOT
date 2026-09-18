from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import discord
import discord_ai_assistant.meeting_feedback_cog as feedback_module
from discord_ai_assistant.ai.gemini import GeminiRequestError
from discord_ai_assistant.app_knowledge import AppKnowledgeStore
from discord_ai_assistant.meeting_feedback import FeedbackCandidate, MeetingFeedbackStore
from discord_ai_assistant.storage.database import Database
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
        self.assertTrue(all(
            item.disabled == (item.custom_id == "meeting_feedback_apply")
            for item in view.children if isinstance(item, discord.ui.Button)
        ))

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


class MeetingCandidateSelectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.database = Database(self.root / "assistant.sqlite3")
        self.addCleanup(lambda: self.database.close())
        self.cog = object.__new__(MeetingReportCommands)
        self.cog.settings = SimpleNamespace(persona_timezone="UTC")
        self.cog.import_root = self.root
        self.cog.feedback_store = MeetingFeedbackStore(self.database, self.root)
        self.cog.knowledge_store = AppKnowledgeStore(self.database)
        self.cog.knowledge_store.upsert_profile(10, "game", "遊戲")
        self.cog._is_dj_member = lambda member: True
        self.cog.ai = SimpleNamespace(social_reply_with_timeout=AsyncMock(return_value="# 修訂週會報"))
        self.job = self.root / "audio-1"
        self.job.mkdir()
        self.review_path = self.job / "review_1.json"
        self.review_path.write_text(json.dumps({
            "review_id": "review-1", "knowledge_profile": "game",
            "audio_sha256": "audio-1", "report_file": "report.md",
        }), encoding="utf-8")
        self.report_path = self.job / "report.md"
        self.report_path.write_text("# 原草稿", encoding="utf-8")
        (self.job / "transcript.json").write_text(json.dumps({
            "segments": [{"start": 0, "end": 5, "text": "卡西 3 和蝴蝶無人機"}],
            "duration_seconds": 5,
        }), encoding="utf-8")
        self.revision_id = self.cog.feedback_store.create_revision(
            review_id="review-1", guild_id=10, profile_key="game", audio_sha256="audio-1",
            title="週會", correction_text="人工修訂", original_report="# 原草稿", created_by=99,
        )
        self.candidate_ids = self.cog.feedback_store.add_candidates(self.revision_id, [
            FeedbackCandidate("asr_alias", "卡西 3", "海域 3", aliases=("卡西 3",)),
            FeedbackCandidate("asr_alias", "蝴蝶無人機", "晶蝶無人機", aliases=("蝴蝶無人機",)),
            FeedbackCandidate("domain_term", "", "高校", "遊戲活動"),
            FeedbackCandidate("domain_term", "", "護盾", "裝備效果"),
            FeedbackCandidate("report_preference", "", "", "保留時間碼"),
            FeedbackCandidate("report_preference", "", "", "使用表格"),
        ])
        self.selected_ids = {self.candidate_ids[index] for index in (0, 2, 4)}
        self.interaction = Mock(spec=discord.Interaction)
        self.interaction.guild = SimpleNamespace(id=10)
        self.interaction.user = Mock(spec=discord.Member, id=99)
        self.interaction.response = SimpleNamespace(
            defer=AsyncMock(), send_message=AsyncMock(), edit_message=AsyncMock(),
        )
        self.interaction.followup = SimpleNamespace(send=AsyncMock())
        self.interaction.message = SimpleNamespace(edit=AsyncMock())

        self.sent_reports = []

        async def send(**kwargs):
            self.sent_reports.append(kwargs["content"])
            for file in kwargs.get("files", []):
                file.close()

        self.interaction.channel = SimpleNamespace(send=send)

    def make_view(self) -> ResilientMeetingFeedbackDecisionView:
        return ResilientMeetingFeedbackDecisionView(
            self.cog, revision_id=self.revision_id, review_path=self.review_path,
            report_path=self.report_path, title="週會", correction_text="人工修訂",
            candidate_ids=self.candidate_ids,
        )

    def assert_selected_learning_only(self) -> None:
        store = self.cog.feedback_store
        self.assertEqual(
            {candidate_id for candidate_id, _ in store.candidates(self.revision_id, "applied")},
            self.selected_ids,
        )
        self.assertEqual(
            {candidate_id for candidate_id, _ in store.candidates(self.revision_id, "skipped")},
            set(self.candidate_ids) - self.selected_ids,
        )
        self.assertEqual(
            {term.term for term in self.cog.knowledge_store.list_terms(10, "game")}, {"海域 3", "高校"},
        )
        self.assertEqual(store.report_preferences(10, "game"), ["保留時間碼"])
        examples = store.connection.execute(
            "SELECT source_alias, corrected_text FROM meeting_asr_training_examples"
        ).fetchall()
        self.assertEqual([tuple(row) for row in examples], [("卡西 3", "海域 3 和蝴蝶無人機")])

    async def test_apply_learns_only_selected_candidates_of_each_kind(self) -> None:
        view = self.make_view()
        selector = next(item for item in view.children if isinstance(item, discord.ui.Select))
        selector._values = [str(i) for i in self.selected_ids]
        await selector.callback(self.interaction)

        await view._run_regeneration(self.interaction, learn=True)

        self.assert_selected_learning_only()

    async def test_selector_requires_explicit_selection_and_supports_clearing(self) -> None:
        view = self.make_view()
        apply_button = next(item for item in view.children if item.custom_id == "meeting_feedback_apply")
        self.assertTrue(apply_button.disabled)
        selectors = [item for item in view.children if isinstance(item, discord.ui.Select)]
        self.assertEqual(len(selectors), 1)
        selector = selectors[0]
        self.assertEqual((selector.min_values, selector.max_values), (0, 6))
        self.assertEqual({option.value for option in selector.options}, {str(i) for i in self.candidate_ids})
        self.assertTrue(all(not option.default for option in selector.options))
        self.assertIn("海域 3", selector.options[0].label)
        self.assertIn("保留時間碼", selector.options[4].label)

        selector._values = [str(i) for i in self.selected_ids]
        await selector.callback(self.interaction)
        self.assertEqual(view.selected_candidate_ids, self.selected_ids)
        self.assertFalse(apply_button.disabled)
        self.assertEqual({int(option.value) for option in selector.options if option.default}, self.selected_ids)
        view._disable()
        self.assertTrue(all(item.disabled for item in view.children))
        view._restore_retry_state()
        self.assertFalse(apply_button.disabled)

        selector._values = []
        await selector.callback(self.interaction)
        self.assertEqual(view.selected_candidate_ids, set())
        self.assertTrue(apply_button.disabled)

    async def test_empty_selection_cannot_apply_even_if_callback_is_invoked(self) -> None:
        view = self.make_view()

        await view._run_regeneration(self.interaction, learn=True)

        self.assertEqual(len(self.cog.feedback_store.candidates(self.revision_id, "pending")), 6)
        self.assertEqual(self.cog.knowledge_store.list_terms(10, "game"), [])

    def reopen_stores(self) -> None:
        self.database.close()
        self.database = Database(self.root / "assistant.sqlite3")
        cog = object.__new__(MeetingReportCommands)
        cog.settings = self.cog.settings
        cog.import_root = self.root
        cog._is_dj_member = self.cog._is_dj_member
        cog.ai = SimpleNamespace(social_reply_with_timeout=AsyncMock(return_value="# 重試成功"))
        cog.feedback_store = MeetingFeedbackStore(self.database, self.root)
        cog.knowledge_store = AppKnowledgeStore(self.database)
        self.cog = cog

    async def test_gemini_failure_then_restart_retry_does_not_repeat_learning_writes(self) -> None:
        view = self.make_view()
        view.selected_candidate_ids = self.selected_ids.copy()
        self.cog.ai.social_reply_with_timeout.side_effect = GeminiRequestError("test Gemini timeout")
        with self.assertLogs("discord_ai_assistant.meeting_full_article_cog", level="ERROR"):
            await view._run_regeneration(self.interaction, learn=True)
        self.assert_selected_learning_only()
        self.assertEqual(view.selected_candidate_ids, self.selected_ids)
        self.assertFalse(view._handled)
        selector = next(item for item in view.children if isinstance(item, discord.ui.Select))
        self.assertTrue(selector.disabled)

        self.reopen_stores()
        statements = []
        self.database.connection.set_trace_callback(statements.append)
        await MeetingReportCommands.meeting_revision_retry.callback(self.cog, self.interaction)

        self.assert_selected_learning_only()
        learning_writes = [sql for sql in statements if sql.lstrip().upper().startswith(("INSERT", "UPDATE"))
                           and any(table in sql for table in (
                               "app_knowledge_terms", "meeting_report_preferences", "meeting_asr_training_examples",
                           ))]
        self.assertEqual(learning_writes, [])
        revision = self.database.connection.execute(
            "SELECT status, revised_report FROM meeting_feedback_revisions WHERE id = ?", (self.revision_id,),
        ).fetchone()
        self.assertEqual(revision["status"], "draft_after_learning")
        self.assertIn("重試成功", revision["revised_report"])

    async def test_failure_before_learning_persists_selection_for_restart_retry(self) -> None:
        view = self.make_view()
        view.selected_candidate_ids = self.selected_ids.copy()
        with patch.object(self.cog, "_snapshot_for_profile", side_effect=RuntimeError("snapshot unavailable")):
            with self.assertLogs("discord_ai_assistant.meeting_full_article_cog", level="ERROR"):
                await view._run_regeneration(self.interaction, learn=True)
        store = self.cog.feedback_store
        self.assertEqual({i for i, _ in store.candidates(self.revision_id, "approved")}, self.selected_ids)
        self.assertEqual({i for i, _ in store.candidates(self.revision_id, "skipped")},
                         set(self.candidate_ids) - self.selected_ids)
        self.assertEqual(self.cog.knowledge_store.list_terms(10, "game"), [])

        self.reopen_stores()
        await MeetingReportCommands.meeting_revision_retry.callback(self.cog, self.interaction)

        self.assert_selected_learning_only()

    async def test_regenerate_only_and_restart_retry_never_learn(self) -> None:
        view = self.make_view()
        view.selected_candidate_ids = self.selected_ids.copy()
        self.cog.ai.social_reply_with_timeout.side_effect = GeminiRequestError("test Gemini timeout")
        with self.assertLogs("discord_ai_assistant.meeting_full_article_cog", level="ERROR"):
            await view._run_regeneration(self.interaction, learn=False)
        self.reopen_stores()
        await MeetingReportCommands.meeting_revision_retry.callback(self.cog, self.interaction)

        self.assertEqual(len(self.cog.feedback_store.candidates(self.revision_id, "skipped")), 6)
        self.assertEqual(self.cog.knowledge_store.list_terms(10, "game"), [])
        self.assertEqual(self.cog.feedback_store.report_preferences(10, "game"), [])
        self.assertEqual(self.cog.feedback_store.training_stats(10, "game")["examples"], 0)

    async def test_same_view_retry_keeps_original_selection_and_learning_mode(self) -> None:
        view = self.make_view()
        view.selected_candidate_ids = self.selected_ids.copy()
        self.cog.ai.social_reply_with_timeout.side_effect = GeminiRequestError("test Gemini timeout")
        with self.assertLogs("discord_ai_assistant.meeting_full_article_cog", level="ERROR"):
            await view._run_regeneration(self.interaction, learn=True)

        selector = next(item for item in view.children if isinstance(item, discord.ui.Select))
        selector._values = [str(i) for i in self.candidate_ids]
        await selector.callback(self.interaction)
        await view._run_regeneration(self.interaction, learn=False)
        self.assertEqual(view.selected_candidate_ids, self.selected_ids)
        self.assert_selected_learning_only()
        self.cog.ai.social_reply_with_timeout.side_effect = None
        await view._run_regeneration(self.interaction, learn=True)

        self.assert_selected_learning_only()
        self.assertTrue(all(item.disabled for item in view.children))

    async def test_cancel_after_learning_failure_preserves_audit_and_stops_retry(self) -> None:
        view = self.make_view()
        view.selected_candidate_ids = self.selected_ids.copy()
        self.cog.ai.social_reply_with_timeout.side_effect = GeminiRequestError("test Gemini timeout")
        with self.assertLogs("discord_ai_assistant.meeting_full_article_cog", level="ERROR"):
            await view._run_regeneration(self.interaction, learn=True)

        await view.cancel.callback(self.interaction)

        self.assert_selected_learning_only()
        self.assertIsNone(self.cog._latest_recoverable_revision(10, 99))
        self.assertTrue(all(item.disabled for item in view.children))

    async def test_apply_clicks_queued_during_selection_publish_only_one_draft(self) -> None:
        view = self.make_view()
        selector = next(item for item in view.children if isinstance(item, discord.ui.Select))
        selector._values = [str(i) for i in self.selected_ids]
        editing = asyncio.Event()
        release = asyncio.Event()

        async def edit(**kwargs):
            editing.set()
            await release.wait()

        self.interaction.response.edit_message.side_effect = edit
        selection = asyncio.create_task(selector.callback(self.interaction))
        await asyncio.wait_for(editing.wait(), timeout=2)
        first = asyncio.create_task(view._run_regeneration(self.interaction, learn=True))
        second = asyncio.create_task(view._run_regeneration(self.interaction, learn=True))
        await asyncio.sleep(0)  # Both clicks reach the lock while Discord's edit is pending.
        release.set()
        await asyncio.wait_for(asyncio.gather(selection, first, second), timeout=2)

        self.assertEqual(len(self.sent_reports), 1)
        self.assert_selected_learning_only()


if __name__ == "__main__":
    unittest.main()
