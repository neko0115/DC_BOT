from __future__ import annotations

import asyncio
import inspect
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import discord
from discord_ai_assistant import chat_style_runtime

from discord_ai_assistant.chat_style_runtime import ChatStyleRuntime
from discord_ai_assistant.chat_style_store import ChatStyleStore
from discord_ai_assistant.main import AssistantBot
from discord_ai_assistant.storage.agent_database import AgentDatabase


UTC = timezone.utc
BASE = datetime(2026, 9, 10, 0, 0, tzinfo=UTC)


def _valid_profile(*, length: str = "balanced") -> str:
    return json.dumps(
        {
            "style": {
                "formality": "casual",
                "length": length,
                "teasing": "medium",
                "reply_chain": "medium",
                "emoji": "low",
                "punctuation": "normal",
                "code_switching": "medium",
                "directness": "high",
            },
            "understanding": [],
            "confidence": 0.8,
        },
        ensure_ascii=False,
    )


class FakeAI:
    def __init__(self, responses: list[object]) -> None:
        self.enabled = True
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def social_reply_with_timeout(self, prompt: str, **kwargs: object) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return str(response)


def _message(
    index: int,
    *,
    at: datetime,
    guild_id: int | None = 1,
    user_id: int = 10,
    bot: bool = False,
    content: str | None = None,
    reply_to: int | None = None,
):
    reference = SimpleNamespace(message_id=reply_to) if reply_to is not None else None
    return SimpleNamespace(
        id=10_000 + index,
        guild=SimpleNamespace(id=guild_id) if guild_id is not None else None,
        author=SimpleNamespace(id=user_id, bot=bot),
        channel=SimpleNamespace(id=100),
        content=content or f"第 {index} 則正常聊天 alpha",
        clean_content=content or f"第 {index} 則正常聊天 alpha",
        created_at=at,
        reference=reference,
        stickers=[],
        is_system=lambda: False,
    )


class ChatStyleRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = AgentDatabase(Path(self.tempdir.name) / "assistant.sqlite3")
        self.store = ChatStyleStore(self.database)

    async def asyncTearDown(self) -> None:
        self.database.close()
        self.tempdir.cleanup()

    def _raw_message(self, raw: str, index: int, user_id: int):
        message = _message(index, at=BASE + timedelta(minutes=index % 100), user_id=user_id)
        message.content = raw
        message.guild.get_member = lambda _: SimpleNamespace(display_name="SyntheticMember")
        message.guild.get_role = lambda _: SimpleNamespace(name="SyntheticRole")
        message.guild._resolve_channel = lambda _: SimpleNamespace(name="synthetic-channel")
        message.mentions, message.role_mentions = [], []
        message.clean_content = discord.Message.clean_content.function(message)
        return message

    async def test_raw_mentions_cannot_become_thirty_effective_clean_samples(self) -> None:
        cases = (
            "<@123456789012345678>", "<@&123456789012345678>", "<#123456789012345678>",
            "<@123456789012345678> <:cat:222222222222222222>",
            "<@123456789012345678> https://example.com",
            "<@123456789012345678> <@&123456789012345678> <#123456789012345678>",
        )
        for case_index, raw in enumerate(cases):
            with self.subTest(raw=raw):
                owner = 10 + case_index
                ai = FakeAI([_valid_profile()])
                runtime = ChatStyleRuntime(SimpleNamespace(), self.store, ai, None)
                for index in range(30):
                    await runtime.observe_message(self._raw_message(raw, case_index * 100 + index, owner))
                rows = self.database.connection.execute(
                    "SELECT COUNT(*) FROM chat_style_samples WHERE guild_id=1 AND user_id=?", (owner,),
                ).fetchone()[0]
                self.assertEqual(
                    {"rows": rows, "model_calls": len(ai.calls), "profile": self.store.get_profile(1, owner) is not None},
                    {"rows": 0, "model_calls": 0, "profile": False},
                )

    async def test_thirty_meaningful_raw_mentions_persist_clean_text_without_numeric_ids(self) -> None:
        for case_index, raw in enumerate((
            "<@123456789012345678> 今天真的笑死",
            "<#123456789012345678> 去這邊看一下紅石比較器",
        )):
            with self.subTest(raw=raw):
                owner = 10 + case_index
                ai = FakeAI([_valid_profile()])
                runtime = ChatStyleRuntime(SimpleNamespace(), self.store, ai, None)
                for index in range(30):
                    message = self._raw_message(raw, case_index * 100 + index, owner)
                    await runtime.observe_message(message)
                rows = self.database.connection.execute(
                    "SELECT content FROM chat_style_samples WHERE guild_id=1 AND user_id=?", (owner,),
                ).fetchall()
                self.assertEqual(len(rows), 30)
                self.assertEqual([row[0] for row in rows], [message.clean_content] * 30)
                self.assertTrue(all("123456789012345678" not in row[0] for row in rows))

    async def test_thirty_keycap_or_mass_mentions_never_trigger_profile(self) -> None:
        for case_index, raw in enumerate(("1️⃣", "@everyone", "@here")):
            with self.subTest(raw=raw):
                owner = 10 + case_index
                ai = FakeAI([_valid_profile()])
                runtime = ChatStyleRuntime(SimpleNamespace(), self.store, ai, None)
                for index in range(30):
                    await runtime.observe_message(self._raw_message(raw, case_index * 100 + index, owner))
                rows = self.database.connection.execute(
                    "SELECT COUNT(*) FROM chat_style_samples WHERE guild_id=1 AND user_id=?", (owner,),
                ).fetchone()[0]
                self.assertEqual(
                    {"rows": rows, "model_calls": len(ai.calls), "profile": self.store.get_profile(1, owner) is not None},
                    {"rows": 0, "model_calls": 0, "profile": False},
                )

    def _seed_sample(
        self,
        index: int,
        *,
        at: datetime,
        content: str | None = None,
        session_key: str | None = None,
    ) -> int | None:
        return self.store.record_sample(
            guild_id=1,
            user_id=10,
            channel_id=100,
            message_id=50_000 + index,
            content=content or f"seed {index} alpha",
            observed_at=at,
            session_key=session_key,
            is_reply=False,
            is_bot=False,
            is_dm=False,
            has_stickers=False,
            is_system=False,
        )

    async def test_no_model_call_before_threshold_then_thirtieth_message_updates_profile(self) -> None:
        ai = FakeAI([_valid_profile(length="short")])
        runtime = ChatStyleRuntime(SimpleNamespace(), self.store, ai, session_state=None)

        for index in range(29):
            await runtime.observe_message(_message(index, at=BASE + timedelta(minutes=index)))
        self.assertEqual(len(ai.calls), 0)
        self.assertIsNone(self.store.get_profile(1, 10))

        await runtime.observe_message(_message(29, at=BASE + timedelta(minutes=29)))
        self.assertEqual(len(ai.calls), 1)
        profile = self.store.get_profile(1, 10)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.profile["style"]["length"], "short")
        self.assertEqual(profile.sample_watermark_id, self.store.latest_sample_id(1, 10, now=BASE + timedelta(minutes=29)))
        self.assertEqual(ai.calls[0]["request_kind"], "memory-chat-style")

    async def _complete_delayed_update(self, delay: timedelta = timedelta(seconds=45)) -> datetime:
        for index in range(29):
            self.assertIsNotNone(self._seed_sample(index, at=BASE - timedelta(minutes=29 - index)))
        clock = BASE

        async def summarize(ai, previous, samples):
            nonlocal clock
            self.assertEqual(len(samples), 30)
            await asyncio.sleep(0)
            clock += delay
            return json.loads(_valid_profile())

        runtime = ChatStyleRuntime(SimpleNamespace(), self.store, FakeAI([]), None)
        with patch.object(chat_style_runtime, "_utc", side_effect=lambda value: value or clock), \
                patch.object(chat_style_runtime, "summarize_chat_style_profile", side_effect=summarize) as summary:
            await runtime.observe_message(_message(29, at=BASE))
        summary.assert_awaited_once()
        self.assertIsNotNone(self.store.get_profile(1, 10))
        return clock

    async def test_delayed_update_records_success_at_completion(self) -> None:
        completed = await self._complete_delayed_update()
        profile = self.store.get_profile(1, 10)
        audit = self.database.connection.execute(
            "SELECT completed_at FROM chat_style_profile_updates WHERE guild_id=1 AND user_id=10"
        ).fetchall()
        self.assertEqual(len(audit), 1)
        self.assertEqual(
            (profile.last_successful_update_at, profile.updated_at, datetime.fromisoformat(audit[0][0])),
            (completed, completed, completed),
        )

    async def test_completion_never_precedes_observation_if_clock_moves_backwards(self) -> None:
        await self._complete_delayed_update(timedelta(seconds=-45))
        profile = self.store.get_profile(1, 10)
        self.assertEqual((profile.updated_at, profile.last_successful_update_at), (BASE, BASE))
        self.assertEqual(profile.effective_sample_count, 30)

    async def test_completion_includes_second_public_summary_await(self) -> None:
        for index in range(29):
            self.assertIsNotNone(self._seed_sample(index, at=BASE - timedelta(minutes=29 - index),
                                                   content=f"public alpha-hop use {index}"))
        with self.database.connection:
            self.database.connection.execute("UPDATE chat_style_samples SET is_public_evidence=1")
        clock = BASE

        async def summarize(ai, previous, samples):
            nonlocal clock
            await asyncio.sleep(0)
            clock += timedelta(seconds=45)
            profile = json.loads(_valid_profile())
            profile["understanding"] = [{"term": "alpha-hop", "meaning": "public travel", "confidence": .9}]
            return profile

        message = _message(29, at=BASE, content="public alpha-hop use")
        message.guild.default_role = object()
        message.channel.type = discord.ChannelType.text
        message.channel.permissions_for = Mock(return_value=SimpleNamespace(view_channel=True))
        runtime = ChatStyleRuntime(SimpleNamespace(), self.store, FakeAI([]), None)
        with patch.object(chat_style_runtime, "_utc", side_effect=lambda value: value or clock), \
                patch.object(chat_style_runtime, "summarize_chat_style_profile", side_effect=summarize) as summary:
            await runtime.observe_message(message)
        self.assertEqual(summary.await_count, 2)
        self.assertEqual(self.store.get_profile(1, 10).last_successful_update_at, BASE + timedelta(seconds=90))
        self.assertEqual(self.store.get_personal_term(1, 10, "alpha-hop").meaning, "public travel")

    async def test_delayed_completion_starts_exact_eight_hour_cooldown(self) -> None:
        completed = await self._complete_delayed_update()
        for index in range(30, 54):
            self.assertIsNotNone(self._seed_sample(index, at=completed + timedelta(minutes=index)))
        blocked = self.store.profile_update_eligibility(1, 10, now=completed + timedelta(hours=8, seconds=-1))
        allowed = self.store.profile_update_eligibility(1, 10, now=completed + timedelta(hours=8))
        self.assertEqual((blocked.eligible, blocked.reason, blocked.new_sample_count), (False, "cooldown", 24))
        self.assertEqual((allowed.eligible, allowed.reason, allowed.new_sample_count), (True, "incremental_threshold", 24))

    async def test_delayed_completion_keeps_full_rolling_twenty_four_hour_budget(self) -> None:
        completed = await self._complete_delayed_update()
        second = completed + timedelta(hours=8)
        for index in range(30, 54):
            self.assertIsNotNone(self._seed_sample(index, at=second - timedelta(minutes=54 - index)))
        self.assertTrue(self.store.profile_update_eligibility(1, 10, now=second).eligible)
        self.store.commit_profile(1, 10, json.loads(_valid_profile()),
                                  sample_watermark_id=self.store.latest_sample_id(1, 10, now=second), updated_at=second)
        for index in range(54, 78):
            self.assertIsNotNone(self._seed_sample(index, at=second + timedelta(minutes=index)))
        blocked = self.store.profile_update_eligibility(1, 10, now=completed + timedelta(hours=24, seconds=-1))
        allowed = self.store.profile_update_eligibility(1, 10, now=completed + timedelta(hours=24))
        self.assertEqual((blocked.eligible, blocked.reason, blocked.new_sample_count), (False, "daily_budget", 24))
        self.assertEqual((allowed.eligible, allowed.reason, allowed.new_sample_count), (True, "incremental_threshold", 24))

    async def test_sample_arriving_during_summary_remains_past_committed_watermark(self) -> None:
        for index in range(29):
            self.assertIsNotNone(self._seed_sample(index, at=BASE - timedelta(minutes=29 - index)))
        entered, release = asyncio.Event(), asyncio.Event()
        summarized_ids = []
        clock = BASE

        async def summarize(ai, previous, samples):
            summarized_ids.extend(sample.id for sample in samples)
            entered.set()
            await release.wait()
            return json.loads(_valid_profile())

        runtime = ChatStyleRuntime(SimpleNamespace(), self.store, FakeAI([]), None)
        with patch.object(chat_style_runtime, "_utc", side_effect=lambda value: value or clock), \
                patch.object(chat_style_runtime, "summarize_chat_style_profile", side_effect=summarize):
            task = asyncio.create_task(runtime.observe_message(_message(29, at=BASE)))
            try:
                await asyncio.wait_for(entered.wait(), timeout=2)
                # Same observation time, inserted after snapshot: completion-time
                # filtering alone must not consume this unsummarized message.
                late_id = self._seed_sample(100, at=BASE, content="UNSUMMARIZED_LATE_SAMPLE")
                self.assertIsNotNone(late_id)
                self.assertNotIn(late_id, summarized_ids)
                clock += timedelta(seconds=45)
            finally:
                release.set()
                await task
        profile = self.store.get_profile(1, 10)
        eligibility = self.store.profile_update_eligibility(1, 10, now=clock)
        self.assertEqual(len(summarized_ids), 30)
        self.assertEqual((profile.sample_watermark_id, profile.effective_sample_count, eligibility.new_sample_count),
                         (max(summarized_ids), 30, 1))

    async def test_thirty_non_text_artifacts_never_collect_or_trigger_summary(self) -> None:
        cases = (
            ("custom_emoji", ("<:cat:123456789012345678>", "<a:dance:123456789012345678>")),
            ("multiple_urls", ("https://a.example/x https://b.example/y",)),
        )
        for case_index, (label, contents) in enumerate(cases):
            with self.subTest(case=label):
                user_id = 10 + case_index
                ai = FakeAI([_valid_profile()])
                runtime = ChatStyleRuntime(SimpleNamespace(), self.store, ai, session_state=None)
                for index in range(30):
                    await runtime.observe_message(_message(
                        case_index * 100 + index, user_id=user_id,
                        at=BASE + timedelta(minutes=index), content=contents[index % len(contents)],
                    ))
                rows = self.database.connection.execute(
                    "SELECT COUNT(*) FROM chat_style_samples WHERE guild_id=? AND user_id=?",
                    (1, user_id),
                ).fetchone()[0]
                self.assertEqual(
                    {"raw_rows": rows, "model_calls": len(ai.calls),
                     "profile_exists": self.store.get_profile(1, user_id) is not None},
                    {"raw_rows": 0, "model_calls": 0, "profile_exists": False},
                )

    async def test_invalid_output_preserves_previous_profile_and_uses_failure_backoff(self) -> None:
        for index in range(30):
            self._seed_sample(index, at=BASE + timedelta(minutes=index), session_key="session-a")
        watermark = self.store.latest_sample_id(1, 10, now=BASE + timedelta(hours=1))
        assert watermark is not None
        old_profile = json.loads(_valid_profile(length="detailed"))
        self.store.commit_profile(1, 10, old_profile, sample_watermark_id=watermark, updated_at=BASE + timedelta(hours=1))

        for offset in range(23):
            self._seed_sample(
                100 + offset,
                at=BASE + timedelta(hours=2, minutes=offset),
                content=f"new sample {offset} alpha",
                session_key="session-b",
            )

        ai = FakeAI(["not json"])
        runtime = ChatStyleRuntime(SimpleNamespace(), self.store, ai, session_state=None)
        await runtime.observe_message(_message(100, at=BASE + timedelta(hours=9), content="第 24 則新樣本 alpha"))
        self.assertEqual(len(ai.calls), 1)
        profile = self.store.get_profile(1, 10)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.profile, old_profile)
        successes = self.database.connection.execute(
            "SELECT COUNT(*) FROM chat_style_profile_updates WHERE guild_id = 1 AND user_id = 10"
        ).fetchone()[0]
        self.assertEqual(successes, 1)

        await runtime.observe_message(_message(101, at=BASE + timedelta(hours=9, minutes=1), content="再一則 alpha"))
        self.assertEqual(len(ai.calls), 1, "invalid output must not cause a tight per-message retry loop")

    async def test_model_failure_is_fail_soft_and_preserves_previous_profile(self) -> None:
        for index in range(30):
            self._seed_sample(index, at=BASE + timedelta(minutes=index), session_key="session-a")
        watermark = self.store.latest_sample_id(1, 10, now=BASE + timedelta(hours=1))
        assert watermark is not None
        old_profile = json.loads(_valid_profile(length="balanced"))
        self.store.commit_profile(1, 10, old_profile, sample_watermark_id=watermark, updated_at=BASE + timedelta(hours=1))
        for offset in range(23):
            self._seed_sample(
                200 + offset,
                at=BASE + timedelta(hours=2, minutes=offset),
                content=f"增量 {offset} alpha",
                session_key="session-b",
            )

        ai = FakeAI([RuntimeError("model unavailable")])
        runtime = ChatStyleRuntime(SimpleNamespace(), self.store, ai, session_state=None)
        await runtime.observe_message(_message(300, at=BASE + timedelta(hours=9), content="第 24 則增量 alpha"))
        profile = self.store.get_profile(1, 10)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.profile, old_profile)
        self.assertEqual(len(ai.calls), 1)

    async def test_dm_bot_and_learning_off_do_not_collect_or_call_model(self) -> None:
        ai = FakeAI([_valid_profile()])
        runtime = ChatStyleRuntime(SimpleNamespace(), self.store, ai, session_state=None)
        await runtime.observe_message(_message(1, at=BASE, guild_id=None))
        await runtime.observe_message(_message(2, at=BASE, bot=True))
        self.store.set_learning_enabled(1, 10, False, updated_at=BASE)
        await runtime.observe_message(_message(3, at=BASE + timedelta(minutes=1)))
        self.assertEqual(self.store.list_samples(1, 10, now=BASE + timedelta(hours=1)), [])
        self.assertEqual(ai.calls, [])

    async def test_sample_storage_failure_never_breaks_message_processing(self) -> None:
        class BrokenStore:
            def learning_enabled(self, guild_id: int, user_id: int) -> bool:
                return True

            def record_sample(self, **kwargs: object) -> int | None:
                raise RuntimeError("sqlite unavailable")

        runtime = ChatStyleRuntime(SimpleNamespace(), BrokenStore(), FakeAI([]), session_state=None)
        await runtime.observe_message(_message(1, at=BASE))

    def test_production_setup_registers_chat_style_runtime_with_shared_store(self) -> None:
        source = inspect.getsource(AssistantBot.setup_hook)
        self.assertIn("ChatStyleRuntime(", source)
        self.assertIn("self.chat_style_store", source)

    async def test_public_evidence_is_separate_from_private_b_layer_sampling(self) -> None:
        for public in (False, True):
            with self.subTest(public=public):
                owner = 20 if public else 30
                profile_data = json.loads(_valid_profile())
                profile_data["understanding"] = [
                    {"term": "alpha-hop", "meaning": "快速傳送", "confidence": 0.8}
                ]
                ai = FakeAI([json.dumps(profile_data), json.dumps(profile_data)])
                runtime = ChatStyleRuntime(SimpleNamespace(), self.store, ai, session_state=None)
                for index in range(30):
                    message = _message(owner * 100 + index, at=BASE + timedelta(minutes=index),
                                       user_id=owner, content=f"使用 alpha-hop 第 {index} 次")
                    message.guild.default_role = object()
                    message.channel.type = discord.ChannelType.text
                    message.channel.permissions_for = Mock(return_value=SimpleNamespace(view_channel=public))
                    with patch.object(chat_style_runtime, "_utc", side_effect=lambda value: value or BASE + timedelta(hours=1)):
                        await runtime.observe_message(message)
                self.assertEqual(len(ai.calls), 2 if public else 1)
                profile = self.store.get_profile(1, owner)
                self.assertIsNotNone(profile, "private guild channels must still support B-layer learning")
                samples = self.store.list_samples(1, owner, now=BASE + timedelta(hours=1))
                self.assertEqual(len(samples), 30)
                self.assertTrue(all(sample.is_public_evidence == public for sample in samples))
                term = self.store.get_personal_term(1, owner, "alpha-hop")
                self.assertEqual(term is not None, public)
                self.assertEqual(bool(profile.profile["understanding"]), public)

    async def test_c_meaning_cannot_inherit_private_samples_or_previous_profile(self) -> None:
        b_result = json.loads(_valid_profile())
        b_result["understanding"] = [{"term": "alpha-hop", "meaning": "private-only meaning", "confidence": 0.9}]
        c_result = json.loads(_valid_profile())
        c_result["understanding"] = [{"term": "alpha-hop", "meaning": "public meaning", "confidence": 0.9}]
        ai = FakeAI([json.dumps(b_result), json.dumps(c_result)])
        runtime = ChatStyleRuntime(SimpleNamespace(), self.store, ai, session_state=None)
        for index in range(30):
            self._seed_sample(index, at=BASE + timedelta(minutes=index))
        self.store.commit_profile(1, 10, b_result, sample_watermark_id=30, updated_at=BASE + timedelta(hours=1))
        for index in range(24):
            message = _message(100 + index, at=BASE + timedelta(hours=10, minutes=index),
                               content="public alpha-hop usage" if index < 3 else "private-only definition alpha-hop")
            message.guild.default_role = object()
            message.channel.type = discord.ChannelType.text
            message.channel.permissions_for = Mock(return_value=SimpleNamespace(view_channel=index < 3))
            with patch.object(chat_style_runtime, "_utc", side_effect=lambda value: value or BASE + timedelta(hours=11)):
                await runtime.observe_message(message)
        self.assertEqual(len(ai.calls), 2)
        self.assertIn("private-only", ai.calls[0]["prompt"])
        self.assertNotIn("private-only", ai.calls[1]["prompt"])
        self.assertNotIn("seed", ai.calls[1]["prompt"])
        self.assertIn("public alpha-hop usage", ai.calls[1]["prompt"])
        self.assertEqual(self.store.get_personal_term(1, 10, "alpha-hop").meaning, "public meaning")

    async def test_public_summary_failure_or_opt_out_does_not_write_partial_c_data(self) -> None:
        for fail_mode in ("invalid", "exception", "opt_out", "expired"):
            with self.subTest(fail_mode=fail_mode):
                owner = 40 + ("invalid", "exception", "opt_out", "expired").index(fail_mode)
                result = json.loads(_valid_profile())
                result["understanding"] = [{"term": "alpha-hop", "meaning": "公開傳送", "confidence": 0.9}]
                clock = BASE + timedelta(hours=1)
                store = self.store

                class BoundaryAI(FakeAI):
                    async def social_reply_with_timeout(self, prompt, **kwargs):
                        nonlocal clock
                        if len(self.calls) == 1:
                            if fail_mode == "opt_out":
                                store.set_learning_enabled(1, owner, False, updated_at=clock)
                            elif fail_mode == "expired":
                                clock = BASE + timedelta(hours=80)
                        return await super().social_reply_with_timeout(prompt, **kwargs)

                second = "not json" if fail_mode == "invalid" else (
                    RuntimeError("unavailable") if fail_mode == "exception" else json.dumps(result))
                ai = BoundaryAI([json.dumps(result), second])
                runtime = ChatStyleRuntime(SimpleNamespace(), store, ai, session_state=None)
                for index in range(30):
                    message = _message(owner * 100 + index, at=BASE + timedelta(minutes=index),
                                       user_id=owner, content="public alpha-hop usage")
                    message.guild.default_role = object()
                    message.channel.type = discord.ChannelType.text
                    message.channel.permissions_for = Mock(return_value=SimpleNamespace(view_channel=True))
                    with patch.object(chat_style_runtime, "_utc", side_effect=lambda value: value or clock):
                        await runtime.observe_message(message)
                self.assertEqual(len(ai.calls), 2)
                self.assertIsNone(store.get_profile(1, owner))
                self.assertIsNone(store.get_personal_term(1, owner, "alpha-hop"))


class PublicChannelEvidenceTests(unittest.TestCase):
    def _message(self, *, public=True, kind=discord.ChannelType.text):
        return SimpleNamespace(
            guild=SimpleNamespace(id=1, default_role=object()),
            channel=SimpleNamespace(type=kind, permissions_for=Mock(
                return_value=SimpleNamespace(view_channel=public, send_messages=False))),
        )

    def _check(self, message):
        helper = getattr(chat_style_runtime, "is_public_channel_evidence", None)
        self.assertTrue(callable(helper), "Task 4 needs a pure public-evidence helper")
        return helper(message)

    def test_everyone_read_permission_is_sufficient_without_send(self):
        message = self._message()
        self.assertTrue(self._check(message))
        message.channel.permissions_for.assert_called_once_with(message.guild.default_role)

    def test_bot_access_does_not_make_role_gated_channel_public(self):
        message = self._message(public=False)
        message.guild.me = object()
        message.channel.permissions_for.side_effect = lambda role: SimpleNamespace(
            view_channel=role is message.guild.me, send_messages=role is message.guild.me)
        self.assertFalse(self._check(message))
        message.channel.permissions_for.assert_called_once_with(message.guild.default_role)

    def test_private_thread_is_never_public(self):
        message = self._message(kind=discord.ChannelType.private_thread)
        message.channel.parent = self._message().channel
        self.assertFalse(self._check(message))
        message.channel.permissions_for.assert_not_called()

    def test_public_thread_uses_parent_everyone_permission(self):
        for public in (True, False):
            with self.subTest(public=public):
                message = self._message(kind=discord.ChannelType.public_thread)
                message.channel.parent = self._message(public=public).channel
                self.assertEqual(self._check(message), public)
                message.channel.parent.permissions_for.assert_called_once_with(message.guild.default_role)
                message.channel.permissions_for.assert_not_called()

    def test_missing_or_failing_context_fails_closed(self):
        cases = [SimpleNamespace(guild=None), SimpleNamespace(), self._message(),
                 self._message(kind=discord.ChannelType.public_thread), self._message(),
                 self._message(), self._message(), self._message()]
        cases[2].guild.default_role = None
        cases[4].channel.permissions_for.side_effect = RuntimeError("unavailable")
        cases[5].channel.permissions_for.return_value = None
        cases[6].channel.permissions_for.return_value = SimpleNamespace(view_channel=None)
        del cases[7].channel
        for message in cases:
            with self.subTest(message=message):
                self.assertFalse(self._check(message))


if __name__ == "__main__":
    unittest.main()
