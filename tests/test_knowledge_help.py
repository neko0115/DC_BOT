from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import discord

from discord_ai_assistant.ai.knowledge_help import (
    KNOWLEDGE_HELP_WAIT_SECONDS,
    QUESTION_COMPLEXITY_COMPLEX,
    QUESTION_COMPLEXITY_NORMAL,
    QUESTION_COMPLEXITY_SEARCH,
    QUESTION_COMPLEXITY_SIMPLE,
    KnowledgeHelpState,
    classify_question_complexity,
    is_cancel_signal,
    is_freshness_dependent,
    is_knowledge_gap_candidate,
)
from discord_ai_assistant.ai.social import SocialParticipant


class KnowledgeGapDetectorTests(unittest.TestCase):
    def test_human_first_delay_is_eight_seconds(self) -> None:
        self.assertEqual(KNOWLEDGE_HELP_WAIT_SECONDS, 8.0)

    def test_clear_how_to_gaps_are_detected(self) -> None:
        positives = (
            "我忘了怎麼查記憶體型號了",
            "我忘記這個要怎麼設定",
            "我不記得 Python 版本要去哪裡看",
            "我記不清楚這個指令怎麼用",
            "我想不起來這個型號叫什麼",
            "這個要怎麼設定才對？",
            "有人知道如何檢查目前的 Python 版本嗎",
            "I can't remember how to check my RAM model",
            "How do I enable this setting?",
        )
        for content in positives:
            with self.subTest(content=content):
                self.assertTrue(is_knowledge_gap_candidate(content))

    def test_common_spoken_questions_do_not_require_question_mark(self) -> None:
        positives = (
            "F16V用什麼雷達",
            "3090多少顯存",
            "為什麼剛剛會 timeout",
            "有人知道這是什麼",
            "這兩個差在哪",
            "這樣正常嗎",
            "這個能不能直接用",
            "這個可不可以接在一起",
            "有沒有更簡單的方法",
            "哪一個比較好",
        )
        for content in positives:
            with self.subTest(content=content):
                self.assertTrue(is_knowledge_gap_candidate(content))

    def test_contextual_follow_up_can_use_prior_channel_context(self) -> None:
        context = "最近 Discord 對話：\nA: Gemini 3.7 剛剛一直 timeout"
        self.assertTrue(is_knowledge_gap_candidate("所以原因呢", context=context))
        self.assertFalse(is_knowledge_gap_candidate("所以原因呢", context=""))

    def test_rhetorical_low_information_questions_are_not_candidates(self) -> None:
        for content in ("蛤？", "你認真？", "不是吧？", "真的假的", "really?"):
            with self.subTest(content=content):
                self.assertFalse(is_knowledge_gap_candidate(content))

    def test_casual_social_questions_from_screenshot_are_not_candidates(self) -> None:
        negatives = (
            "誰調誰",
            "誰跟誰在一起",
            "你們在聊什麼",
            "你今天吃什麼",
            "他是不是喜歡她",
            "謝謝解說？",
        )
        context = "布丁：只有我調情調多了很累嗎\n布偶貓：被調情的也很累"
        for content in negatives:
            with self.subTest(content=content):
                self.assertFalse(is_knowledge_gap_candidate(content, context=context))

    def test_clear_public_who_questions_remain_candidates(self) -> None:
        self.assertTrue(is_knowledge_gap_candidate("F-16 是誰設計的？"))

    def test_clear_public_current_fact_gaps_are_candidates(self) -> None:
        positives = (
            "我不知道這班車幾點",
            "我忘了這班車的時刻表",
            "我不知道今天的天氣",
            "我忘了這間店的營業時間",
            "今天台北天氣怎樣",
        )
        for content in positives:
            with self.subTest(content=content):
                self.assertTrue(is_knowledge_gap_candidate(content))
                self.assertTrue(is_freshness_dependent(content))

    def test_ambiguous_or_sensitive_topics_are_not_candidates(self) -> None:
        negatives = (
            "我不知道他怎麼想",
            "我不知道她是不是喜歡我",
            "這個政治議題要怎麼看",
            "這個藥物要怎麼吃",
            "股票要怎麼買",
            "我忘記 API key 在哪裡",
            "今天好累",
        )
        for content in negatives:
            with self.subTest(content=content):
                self.assertFalse(is_knowledge_gap_candidate(content))

    def test_freshness_questions_are_identified_for_search_routing(self) -> None:
        self.assertTrue(is_freshness_dependent("我忘了這班車的時刻表"))
        self.assertTrue(is_freshness_dependent("我不知道這班車幾點"))
        self.assertTrue(is_freshness_dependent("今天幾點開門？"))
        self.assertTrue(is_freshness_dependent("I don't know how to check the latest schedule"))
        self.assertFalse(is_freshness_dependent("我忘了怎麼查記憶體型號"))

    def test_complexity_classification_is_deterministic(self) -> None:
        self.assertEqual(classify_question_complexity("3090多少顯存"), QUESTION_COMPLEXITY_SIMPLE)
        self.assertEqual(
            classify_question_complexity(
                "為什麼剛剛會 timeout",
                context="A: Gemini 3.7 request 卡住了",
            ),
            QUESTION_COMPLEXITY_NORMAL,
        )
        self.assertEqual(
            classify_question_complexity("幫我比較這兩種架構的優缺點和取捨"),
            QUESTION_COMPLEXITY_COMPLEX,
        )
        self.assertEqual(classify_question_complexity("今天台北天氣怎樣"), QUESTION_COMPLEXITY_SEARCH)

    def test_cancel_signals_are_detected(self) -> None:
        self.assertTrue(is_cancel_signal("喔找到了，沒事了"))
        self.assertFalse(is_cancel_signal("Windows 11 上面"))


class KnowledgeHelpStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = (1, 10)
        self.state = KnowledgeHelpState(
            cooldown_seconds=600,
            acknowledgement_seconds=120,
            quiet_seconds=1800,
        )

    def test_initial_candidate_can_schedule_and_pending_blocks_duplicate(self) -> None:
        self.assertTrue(self.state.can_schedule(self.key, 1000))
        self.state.start_pending(self.key, author_id=11, source_message_id=101)
        self.assertFalse(self.state.can_schedule(self.key, 1001))
        self.assertTrue(self.state.pending_matches(self.key, 101))

    def test_different_human_cancels_pending_help(self) -> None:
        self.state.start_pending(self.key, author_id=11, source_message_id=101)
        cancelled = self.state.observe_human(
            self.key,
            author_id=22,
            content="你可以按 Win+R",
            now=1005,
        )
        self.assertTrue(cancelled)
        self.assertFalse(self.state.pending_matches(self.key, 101))

    def test_same_author_clarification_does_not_cancel_but_cancel_signal_does(self) -> None:
        self.state.start_pending(self.key, author_id=11, source_message_id=101)
        self.assertFalse(
            self.state.observe_human(
                self.key,
                author_id=11,
                content="Windows 11 上面",
                now=1005,
            )
        )
        self.assertTrue(self.state.pending_matches(self.key, 101))
        self.assertTrue(
            self.state.observe_human(
                self.key,
                author_id=11,
                content="喔找到了，算了",
                now=1006,
            )
        )
        self.assertFalse(self.state.pending_matches(self.key, 101))

    def test_sent_help_enforces_ten_minute_channel_cooldown(self) -> None:
        self.state.start_pending(self.key, author_id=11, source_message_id=101)
        self.state.record_sent(self.key, response_message_id=9001, now=1000)
        self.assertFalse(self.state.can_schedule(self.key, 1599))
        self.assertTrue(self.state.can_schedule(self.key, 1600))

    def test_explicit_engagement_resets_unacknowledged_streak(self) -> None:
        self.state.record_sent(self.key, response_message_id=9001, now=1000)
        self.assertEqual(self.state.unacknowledged_count(self.key, 1121), 1)
        self.state.record_sent(self.key, response_message_id=9002, now=1700)
        self.state.observe_human(
            self.key,
            author_id=11,
            content="謝啦",
            now=1710,
            reply_to_message_id=9002,
        )
        self.assertEqual(self.state.unacknowledged_count(self.key, 1900), 0)

    def test_explicit_bot_mention_also_resets_unacknowledged_streak(self) -> None:
        self.state.record_sent(self.key, response_message_id=9001, now=1000)
        self.assertEqual(self.state.unacknowledged_count(self.key, 1121), 1)
        self.state.record_sent(self.key, response_message_id=9002, now=1700)
        self.state.observe_human(
            self.key,
            author_id=11,
            content="@墨雪 再問一個",
            now=1710,
            mentions_bot=True,
        )
        self.assertEqual(self.state.unacknowledged_count(self.key, 1900), 0)

    def test_two_unacknowledged_responses_trigger_longer_quiet_period(self) -> None:
        self.state.record_sent(self.key, response_message_id=9001, now=1000)
        self.assertEqual(self.state.unacknowledged_count(self.key, 1121), 1)
        self.state.record_sent(self.key, response_message_id=9002, now=1700)
        self.assertEqual(self.state.unacknowledged_count(self.key, 1821), 0)
        self.assertGreater(self.state.quiet_until(self.key, 1821), 1821)
        self.assertFalse(self.state.can_schedule(self.key, 2400))
        self.assertTrue(self.state.can_schedule(self.key, 3621))


class SocialParticipantKnowledgeHelpRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_generic_social_participant_does_not_immediately_answer_gap_candidate(self) -> None:
        social = SocialParticipant.__new__(SocialParticipant)
        social.ai = SimpleNamespace(enabled=True)
        social.enabled_for = lambda message: True
        social.automatic_enabled = lambda guild_id: True
        social.is_do_not_disturb = lambda guild_id: False
        social._small_question_context = lambda message: ""
        message = SimpleNamespace(
            guild=SimpleNamespace(id=1),
            clean_content="我忘了怎麼查記憶體型號了",
        )

        self.assertIsNone(await social.consider(message))

    def test_knowledge_help_is_not_limited_to_persona_channel(self) -> None:
        social = SocialParticipant.__new__(SocialParticipant)
        social.ai = SimpleNamespace(enabled=True)
        social.automatic_enabled = lambda guild_id: True
        social.is_do_not_disturb = lambda guild_id: False
        social.enabled_for = lambda message: False
        social._small_question_context = lambda message: "A: 前文"

        channel = MagicMock(spec=discord.TextChannel)
        channel.permissions_for.return_value = SimpleNamespace(view_channel=True, send_messages=True)
        message = SimpleNamespace(
            guild=SimpleNamespace(id=1, me=object()),
            channel=channel,
            clean_content="3090多少顯存",
        )

        self.assertTrue(social.can_offer_knowledge_help(message))


if __name__ == "__main__":
    unittest.main()
