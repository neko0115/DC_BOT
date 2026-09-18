from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import discord

from discord_ai_assistant import chat_style_commands as controls
from discord_ai_assistant import main
from discord_ai_assistant.ai.memory_shared import record_shared_candidate
from discord_ai_assistant.app_knowledge import AppKnowledgeStore
from discord_ai_assistant.chat_style_runtime import ChatStyleRuntime
from discord_ai_assistant.chat_style_store import ChatStyleStore
from discord_ai_assistant.commands import AssistantCommands
from discord_ai_assistant.knowledge_enrichment import (
    GUILD_LEXICON_PROFILE_KEY, KnowledgeEnrichmentStore, PublicKnowledgeKey,
    PublicKnowledgeStore, PublicSource, PublicWebResult,
)
from discord_ai_assistant.memory_add import install_memory_add_command
from discord_ai_assistant.memory_inspection_commands import install_memory_inspection_commands
from discord_ai_assistant.slash_groups import GroupedSlashCommands
from discord_ai_assistant.storage.agent_database import AgentDatabase


BASE = datetime(2026, 9, 10, 8, tzinfo=timezone.utc)
COMMANDS = {"chat_profile", "chat_profile_reset", "chat_learning", "lexicon_search", "lexicon_forget"}
OWNER_TABLES = ("chat_style_profiles", "chat_style_samples", "chat_style_terms", "chat_style_profile_updates")
PROFILE = {
    "style": {"formality": "casual", "length": "short", "teasing": "medium",
              "reply_chain": "medium", "emoji": "low", "punctuation": "normal",
              "code_switching": "medium", "directness": "high"},
    "understanding": [], "confidence": 0.9,
}


class Response:
    def __init__(self):
        self.sent = []
        self.deferred = asyncio.Event()
        self.defer_ephemeral = None

    def is_done(self):
        return bool(self.sent) or self.deferred.is_set()

    async def send_message(self, text, *, ephemeral=False):
        self.sent.append((text, ephemeral))

    async def defer(self, *, ephemeral=False, thinking=False):
        self.defer_ephemeral = ephemeral
        self.deferred.set()


class Core:
    # Exercise the existing DJ/admin authority, including Discord Member checks.
    _is_dj = AssistantCommands._is_dj
    _is_dj_member = AssistantCommands._is_dj_member

    def __init__(self, database):
        self.database = database
        self.bot = None
        self.settings = SimpleNamespace(dj_role_id=77, dj_role_name="DJ")
        self.social = SimpleNamespace()
        self.memory_session = SimpleNamespace()


class ChatStyleCommandsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.database = AgentDatabase(self.folder / "test.sqlite3")
        self.addCleanup(self.database.close)
        self.store = ChatStyleStore(self.database)
        self.app = AppKnowledgeStore(self.database)
        self.lexicon = KnowledgeEnrichmentStore(self.database, self.app)
        self.cache = PublicKnowledgeStore(self.database)
        self.ai = SimpleNamespace(enabled=True, social_reply_with_timeout=AsyncMock(return_value=json.dumps(PROFILE)))
        self.runtime = ChatStyleRuntime(None, self.store, self.ai, SimpleNamespace())
        self.core = Core(self.database)
        self.grouped = GroupedSlashCommands(self.core)
        install_memory_add_command(self.grouped)
        install_memory_inspection_commands(self.grouped)
        controls.install_chat_style_commands(self.grouped, self.runtime)
        self.message_id = 1000
        self.clock = patch.object(controls, "_now", return_value=BASE)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def interaction(self, *, guild=1, user=10, permission="member"):
        member = Mock(spec=discord.Member)
        member.id = user
        member.guild_permissions = SimpleNamespace(manage_guild=permission == "admin")
        member.roles = [SimpleNamespace(id=77, name="DJ")] if permission == "dj" else []
        response = Response()
        return SimpleNamespace(
            guild=SimpleNamespace(id=guild) if guild is not None else None,
            guild_id=guild, user=member, response=response,
            followup=SimpleNamespace(send=response.send_message),
        )

    async def invoke(self, name, interaction=None, **kwargs):
        interaction = interaction or self.interaction()
        await self.grouped.memory.get_command(name).callback(interaction, **kwargs)
        self.assertTrue(interaction.response.sent)
        self.assertTrue(all(ephemeral for _, ephemeral in interaction.response.sent))
        self.assertTrue(all(len(text) <= 2000 for text, _ in interaction.response.sent))
        return interaction.response.sent[-1][0]

    def samples(self, *, guild=1, user=10, term="alpha-hop", count=3, session="a", at=BASE):
        for index in range(count):
            self.message_id += 1
            self.store.record_sample(
                guild_id=guild, user_id=user, channel_id=100, message_id=self.message_id,
                content=f"RAW_SAMPLE_DO_NOT_SHOW {term} 使用 {index}", observed_at=at,
                session_key=session, is_reply=False, is_bot=False, is_dm=False,
                has_stickers=False, is_system=False, is_public_evidence=True,
            )
        return self.store.list_samples(guild, user, now=at)

    def personal(self, *, guild=1, user=10, term="alpha-hop", meaning="快速傳送", session="a"):
        samples = self.samples(guild=guild, user=user, term=term, session=session)
        self.assertIsNotNone(self.lexicon.register_personal_understanding(
            guild, user, term=term, meaning=meaning, confidence=0.9, samples=samples, now=BASE))

    def profile(self, *, guild=1, user=10, length="short"):
        samples = self.samples(guild=guild, user=user)
        profile = {**PROFILE, "style": {**PROFILE["style"], "length": length},
                   "internal_raw": "JSON_DUMP_DO_NOT_SHOW"}
        self.store.commit_profile(guild, user, profile, sample_watermark_id=samples[-1].id, updated_at=BASE)

    def active(self, *, guild=1, term="alpha-hop"):
        self.personal(guild=guild, user=10, term=term, session="a")
        self.personal(guild=guild, user=11, term=term, session="b")
        entry = self.lexicon.get_guild_lexicon_entry(guild, term)
        self.assertIsNotNone(entry)
        return entry

    def public_entry(self):
        key = PublicKnowledgeKey("public-meaning", domain="game")
        result = PublicWebResult("public-meaning", "Public definition", 0.9,
                                (PublicSource("https://example.org/term", "Reference", "Public definition"),))
        self.assertIsNotNone(self.cache.store_web_result(key, result, now=BASE))
        return key

    def preserved_snapshot(self, *, excluding_owner=False):
        snapshot = {}
        for (table,) in self.database.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'"):
            if excluding_owner and table in OWNER_TABLES:
                rows = self.database.connection.execute(
                    f'SELECT * FROM "{table}" WHERE NOT (guild_id = 1 AND user_id = 10)').fetchall()
            else:
                rows = self.database.connection.execute(f'SELECT * FROM "{table}"').fetchall()
            snapshot[table] = sorted(map(tuple, rows), key=repr)
        return snapshot

    async def test_profile_shows_only_caller_current_guild_and_human_summaries(self):
        self.profile()
        self.personal()
        self.profile(user=11, length="detailed")
        self.personal(user=11, term="OTHER_USER_TERM")
        self.profile(guild=2, length="formal")
        self.personal(guild=2, term="OTHER_GUILD_TERM")
        text = await self.invoke("chat_profile")
        for expected in ("精簡", "隨性", "alpha-hop", "快速傳送", "啟用"):
            self.assertIn(expected, text)
        for private in ("other_user_term", "other_guild_term", "RAW_SAMPLE_DO_NOT_SHOW", "JSON_DUMP_DO_NOT_SHOW", '"style"'):
            self.assertNotIn(private, text)

    async def test_profile_reports_current_counts_watermark_and_last_success(self):
        self.profile()
        self.samples(count=2)
        text = await self.invoke("chat_profile")
        self.assertIn("有效樣本：5", text)
        self.assertIn("上次成功更新後新增：2", text)
        self.assertIn("2026-09-10 08:00 UTC", text)

    async def test_profile_counts_remain_accurate_when_learning_disabled(self):
        self.profile()
        self.samples(count=2)
        self.store.set_learning_enabled(1, 10, False, updated_at=BASE)
        text = await self.invoke("chat_profile")
        self.assertIn("停用", text)
        self.assertIn("有效樣本：5", text)
        self.assertIn("上次成功更新後新增：2", text)

    async def test_missing_profile_is_not_fabricated_from_another_owner(self):
        self.profile(user=11)
        text = await self.invoke("chat_profile")
        self.assertIn("尚未建立", text)
        self.assertIn("有效樣本：0", text)
        self.assertIn("啟用", text)
        self.assertNotIn("精簡", text)

    async def test_profile_counts_respect_raw_sample_ttl(self):
        self.samples(at=BASE - timedelta(hours=73))
        text = await self.invoke("chat_profile")
        self.assertIn("有效樣本：0", text)

    async def test_profile_aids_are_bounded_and_escaped(self):
        self.profile()
        for i in range(8):
            self.personal(term=f"aid-{i}", meaning="這是 **詞彙** 說明 " * 12)
        text = await self.invoke("chat_profile")
        self.assertEqual(text.count("aid-"), 5)
        self.assertIn(r"\*\*詞彙\*\*", text)
        self.assertNotIn("RAW_SAMPLE_DO_NOT_SHOW", text)

    async def test_all_commands_reject_dm_ephemerally_without_mutation(self):
        before = self.preserved_snapshot()
        for name, kwargs in (("chat_profile", {}), ("chat_profile_reset", {}),
                             ("chat_learning", {"mode": "啟用"}),
                             ("lexicon_search", {"term": "alpha"}), ("lexicon_forget", {"entry_id": 1})):
            with self.subTest(command=name):
                text = await self.invoke(name, self.interaction(guild=None, permission="admin"), **kwargs)
                self.assertIn("伺服器", text)
        self.assertEqual(self.preserved_snapshot(), before)

    async def test_reset_removes_only_owner_chat_style_data_and_counters(self):
        self.active()
        self.profile()
        self.profile(user=11)
        self.profile(guild=2)
        self.personal(guild=2)
        self.public_entry()
        self.database.add_user_memory(1, 10, "偏好", "喜歡簡潔的技術解釋")
        self.database.add_user_memory(1, 10, "專案", "測試專案使用 Python")
        shared_id = record_shared_candidate(
            self.database, guild_id=1, author_id=10,
            content="上次 Minecraft 紅石農場被拆壞變成群內梗", domain="game", subdomain="minecraft",
            memory_kind="inside_joke", confidence=.95, importance=2, entity_type="event",
            entity="redstone_joke", session_key="60:1", channel_id=60, message_id=600,
            observed_at=BASE.isoformat(), shared_group_event=True, participant_ids=(10, 11),
        )
        self.assertIsNotNone(shared_id)
        self.store.set_learning_enabled(1, 10, False, updated_at=BASE)
        self.runtime._last_attempt.update({(1, 10): BASE, (1, 11): BASE, (2, 10): BASE})
        preserved = self.preserved_snapshot(excluding_owner=True)
        lock = self.runtime._locks.setdefault((1, 10), asyncio.Lock())
        text = await self.invoke("chat_profile_reset")
        self.assertIn("重設", text)
        for table in OWNER_TABLES:
            with self.subTest(table=table):
                count = self.database.connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE guild_id = 1 AND user_id = 10").fetchone()[0]
                self.assertEqual(count, 0)
        self.assertEqual(self.preserved_snapshot(excluding_owner=True), preserved)
        self.assertFalse(self.store.learning_enabled(1, 10))
        self.assertEqual(self.runtime._last_attempt, {(1, 11): BASE, (2, 10): BASE})
        self.assertIs(self.runtime._locks[(1, 10)], lock)

    async def test_reset_preserves_default_enabled_and_is_repeatable(self):
        for _ in range(2):
            await self.invoke("chat_profile_reset")
            self.assertTrue(self.store.learning_enabled(1, 10))

    async def test_reset_database_failure_rolls_back_and_reports_ephemeral_error(self):
        self.profile()
        self.personal()
        self.runtime._last_attempt[(1, 10)] = BASE
        before = self.preserved_snapshot()
        self.database.connection.execute(
            "CREATE TEMP TRIGGER fail_reset BEFORE DELETE ON chat_style_profile_updates "
            "BEGIN SELECT RAISE(ABORT, 'PRIVATE_DIAGNOSTIC'); END")
        command = self.grouped.memory.get_command("chat_profile_reset")
        interaction = self.interaction()
        with self.assertRaises(sqlite3.IntegrityError) as raised:
            await command.callback(interaction)
        with self.assertLogs(controls.LOGGER, level="WARNING"):
            await command.on_error(interaction, discord.app_commands.CommandInvokeError(command, raised.exception))
        self.assertTrue(interaction.response.defer_ephemeral)
        text, ephemeral = interaction.response.sent[-1]
        self.assertTrue(ephemeral)
        self.assertIn("未完成", text)
        self.assertNotIn("PRIVATE_DIAGNOSTIC", text)
        self.assertEqual(self.preserved_snapshot(), before)
        self.assertEqual(self.runtime._last_attempt[(1, 10)], BASE)

    def message(self):
        self.message_id += 1
        return SimpleNamespace(
            id=self.message_id, guild=SimpleNamespace(id=1), author=SimpleNamespace(id=10, bot=False),
            channel=SimpleNamespace(id=100), clean_content="正常聊天 alpha-hop", created_at=BASE,
            reference=None, stickers=[], is_system=lambda: False,
        )

    async def test_reset_waits_for_inflight_update_then_clears_data_and_backoff(self):
        self.samples(count=29)
        entered, release = asyncio.Event(), asyncio.Event()

        async def summary(*args, **kwargs):
            entered.set()
            await release.wait()
            return json.dumps(PROFILE)

        self.ai.social_reply_with_timeout.side_effect = summary
        update = asyncio.create_task(self.runtime.observe_message(self.message()))
        await asyncio.wait_for(entered.wait(), 2)
        interaction = self.interaction()
        reset = asyncio.create_task(self.invoke("chat_profile_reset", interaction))
        try:
            await asyncio.wait_for(interaction.response.deferred.wait(), 2)
            self.assertTrue(interaction.response.defer_ephemeral)
            self.assertFalse(reset.done())
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(update, reset), 2)
        self.assertIsNone(self.store.get_profile(1, 10))
        self.assertEqual(self.store.list_samples(1, 10, now=BASE), [])
        self.assertNotIn((1, 10), self.runtime._last_attempt)
        self.samples(count=29)
        await self.runtime.observe_message(self.message())
        self.assertIsNotNone(self.store.get_profile(1, 10))
        self.assertEqual(self.ai.social_reply_with_timeout.await_count, 2)

    async def test_learning_choices_are_fixed_and_private_commands_have_no_target_option(self):
        command = self.grouped.memory.get_command("chat_learning")
        self.assertEqual([(c.name, c.value) for c in command.parameters[0].choices], [("啟用", "啟用"), ("停用", "停用")])
        self.assertEqual([p.name for p in command.parameters], ["mode"])
        for name in ("chat_profile", "chat_profile_reset"):
            self.assertEqual(self.grouped.memory.get_command(name).parameters, [])

    async def test_caller_toggle_is_authoritative_and_preserves_existing_data(self):
        self.profile()
        self.personal()
        before = {table: rows for table, rows in self.preserved_snapshot().items() if table != "chat_style_learning"}
        self.assertTrue(self.store.learning_enabled(1, 10))
        await self.invoke("chat_learning", mode="停用")
        self.assertFalse(self.store.learning_enabled(1, 10))
        await self.runtime.observe_message(self.message())
        self.ai.social_reply_with_timeout.assert_not_awaited()
        self.assertEqual({table: rows for table, rows in self.preserved_snapshot().items() if table != "chat_style_learning"}, before)
        await self.invoke("chat_learning", mode="啟用")
        self.assertTrue(self.store.learning_enabled(1, 10))
        self.assertTrue(self.store.learning_enabled(2, 10))
        self.assertTrue(self.store.learning_enabled(1, 11))

    async def test_admin_toggle_cannot_enable_another_member(self):
        self.store.set_learning_enabled(1, 10, False, updated_at=BASE)
        await self.invoke("chat_learning", self.interaction(user=11, permission="admin"), mode="啟用")
        self.assertFalse(self.store.learning_enabled(1, 10))
        self.assertTrue(self.store.learning_enabled(1, 11))

    async def test_learning_toggle_does_not_change_other_guild_opt_out(self):
        self.store.set_learning_enabled(2, 10, False, updated_at=BASE)
        for mode in ("停用", "啟用"):
            await self.invoke("chat_learning", mode=mode)
            self.assertFalse(self.store.learning_enabled(2, 10))
            self.assertEqual(self.store.learning_enabled(1, 10), mode == "啟用")

    async def test_invalid_learning_choice_fails_closed(self):
        self.store.set_learning_enabled(1, 10, False, updated_at=BASE)
        await self.invoke("chat_learning", mode="true")
        self.assertFalse(self.store.learning_enabled(1, 10))

    async def test_disable_during_summary_prevents_profile_write_without_deleting_samples(self):
        self.samples(count=29)

        async def summary(*args, **kwargs):
            await self.invoke("chat_learning", mode="停用")
            return json.dumps(PROFILE)

        self.ai.social_reply_with_timeout.side_effect = summary
        await self.runtime.observe_message(self.message())
        self.assertIsNone(self.store.get_profile(1, 10))
        self.assertEqual(len(self.store.list_samples(1, 10, now=BASE)), 30)
        await self.runtime.observe_message(self.message())
        self.assertEqual(self.ai.social_reply_with_timeout.await_count, 1)

    async def test_lexicon_search_and_forget_deny_ordinary_member(self):
        entry = self.active()
        for name, kwargs in (("lexicon_search", {"term": "alpha"}), ("lexicon_forget", {"entry_id": entry.id})):
            text = await self.invoke(name, **kwargs)
            self.assertIn("DJ", text)
            self.assertNotIn("快速傳送", text)
        self.assertIsNotNone(self.lexicon.get_guild_lexicon_entry(1, "alpha-hop"))

    async def test_lexicon_search_allows_existing_dj_and_admin_authority(self):
        entry = self.active()
        for permission in ("dj", "admin"):
            text = await self.invoke("lexicon_search", self.interaction(permission=permission), term="alpha")
            self.assertIn(str(entry.id), text)
            self.assertIn("alpha-hop", text)
            self.assertIn("快速傳送", text)

    async def test_lexicon_search_only_current_guild_active_not_other_layers_or_evidence(self):
        self.active()
        self.active(guild=2, term="alpha-other-guild")
        self.personal(term="alpha-personal")
        self.personal(user=12, term="alpha-candidate")
        self.personal(user=13, term="alpha-candidate")
        self.public_entry()
        text = await self.invoke("lexicon_search", self.interaction(permission="dj"), term="")
        self.assertIn("alpha-hop", text)
        for forbidden in ("alpha-other-guild", "alpha-personal", "alpha-candidate", "public-meaning", "RAW_SAMPLE_DO_NOT_SHOW"):
            self.assertNotIn(forbidden, text)

    async def test_lexicon_search_is_bounded(self):
        for i in range(12):
            self.active(term=f"term-{i:02d}")
        text = await self.invoke("lexicon_search", self.interaction(permission="admin"), term="term-")
        self.assertEqual(text.count("term-"), 5)
        self.assertIn("最多 5", text)

    async def test_lexicon_search_empty_is_explicit(self):
        text = await self.invoke("lexicon_search", self.interaction(permission="dj"), term="missing")
        self.assertIn("沒有找到", text)

    async def test_lexicon_search_rejects_oversized_query(self):
        self.active()
        with patch.object(KnowledgeEnrichmentStore, "search_guild_lexicon", side_effect=AssertionError("unbounded query")):
            text = await self.invoke("lexicon_search", self.interaction(permission="dj"), term="a" * 81)
        self.assertIn("80", text)

    async def test_admin_can_forget_current_guild_entry(self):
        entry = self.active()
        text = await self.invoke("lexicon_forget", self.interaction(permission="admin"), entry_id=entry.id)
        self.assertIn("已刪除", text)
        self.assertIsNone(self.lexicon.get_guild_lexicon_entry(1, "alpha-hop"))

    async def test_forget_reuses_store_and_preserves_every_other_layer(self):
        entry = self.active()
        other = self.active(guild=2)
        self.profile()
        self.database.add_user_memory(1, 10, "偏好", "喜歡簡潔的技術解釋")
        key = self.public_entry()
        before = self.preserved_snapshot()
        original = KnowledgeEnrichmentStore.forget_guild_lexicon_entry
        with patch.object(KnowledgeEnrichmentStore, "forget_guild_lexicon_entry", autospec=True, side_effect=original) as delete:
            await self.invoke("lexicon_forget", self.interaction(permission="dj"), entry_id=entry.id)
            self.assertEqual(delete.call_count, 1)
            self.assertEqual(delete.call_args.args[1:], (1, entry.id))
        self.assertIsNone(self.lexicon.get_guild_lexicon_entry(1, "alpha-hop"))
        self.assertIsNone(self.app.get_term(1, GUILD_LEXICON_PROFILE_KEY, "alpha-hop"))
        self.assertEqual(self.database.connection.execute(
            "SELECT COUNT(*) FROM guild_lexicon_evidence WHERE guild_id=1 AND entry_id=?", (entry.id,)).fetchone()[0], 0)
        self.assertEqual(self.lexicon.get_guild_lexicon_entry(2, "alpha-hop"), other)
        self.assertIsNotNone(self.cache.get(key, now=BASE))
        after = self.preserved_snapshot()
        for table in (*OWNER_TABLES, "chat_style_learning", "user_memories", "public_knowledge_cache"):
            self.assertEqual(after[table], before[table], table)

    async def test_forget_foreign_and_nonexistent_id_are_indistinguishable(self):
        foreign = self.active(guild=2)
        before = self.preserved_snapshot()
        responses = []
        for entry_id in (foreign.id, 999999):
            responses.append(await self.invoke("lexicon_forget", self.interaction(permission="admin"), entry_id=entry_id))
        self.assertEqual(responses[0], responses[1])
        self.assertIn("找不到", responses[0])
        self.assertEqual(self.preserved_snapshot(), before)

    async def test_installer_reuses_memory_group_and_is_idempotent(self):
        group = self.grouped.memory
        before = {c.name: c for c in group.commands}
        controls.install_chat_style_commands(self.grouped, self.runtime)
        self.assertIs(self.grouped.memory, group)
        self.assertEqual({c.name: c for c in group.commands}, before)
        self.assertTrue(COMMANDS.issubset(before))
        self.assertIn("add", before)
        self.assertEqual([g.name for g in self.grouped.get_app_commands()].count("memory"), 1)

    async def test_production_setup_installs_actual_commands_before_group_registration_and_sync(self):
        settings = SimpleNamespace(project_root=self.folder, gemini_api_key=None, discord_guild_id=None,
                                   tts_remote_workers="", tts_worker_mode="auto")
        for name in ("provider", "gemini_model", "gemini_voice", "gemini_style", "gemini_timeout_seconds",
                     "kokoro_enabled", "kokoro_voice", "kokoro_speed", "worker_token", "worker_health_timeout_seconds",
                     "worker_request_timeout_seconds", "worker_failure_cooldown_seconds"):
            setattr(settings, "tts_" + name, None)
        installed = []

        async def add_cog(cog):
            if isinstance(cog, GroupedSlashCommands):
                self.assertTrue(COMMANDS.issubset({c.name for c in cog.memory.commands}))
                installed.append(cog)

        async def sync():
            self.assertEqual(len(installed), 1)
            # Invoke a real newly installed callback before sync returns.
            interaction = self.interaction()
            await installed[0].memory.get_command("chat_learning").callback(interaction, "停用")
            self.assertFalse(self.store.learning_enabled(1, 10))
            self.assertTrue(interaction.response.sent[-1][1])
            return []

        bot = SimpleNamespace(settings=settings, database=self.database, chat_style_store=self.store,
                              agent=SimpleNamespace(start=AsyncMock()), agent_bus=object(),
                              library=object(), music=object(), ai=self.ai, tool_gateway_server=None,
                              capture_hub_server=None, add_cog=add_cog,
                              tree=SimpleNamespace(sync=sync, remove_command=Mock()))
        with ExitStack() as stack:
            stack.enter_context(patch.object(main, "ArtifactToolEffectAssistantCommands", return_value=self.core))
            for name in ("configure_event_loop", "ResilientWindowsSpeechSynthesizer", "WorkerPoolSpeechSynthesizer",
                         "MemoryV2PassiveRuntime", "KnowledgeHelpRuntime", "AgentEventBridge", "TTSWorkerCommands",
                         "VoicePlaybackWatchdog", "LyricsCommands", "MeetingReportCommands", "install_capture_agent_commands"):
                stack.enter_context(patch.object(main, name))
            await main.AssistantBot.setup_hook(bot)


if __name__ == "__main__":
    unittest.main()
