from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from discord_ai_assistant.app_knowledge import AppKnowledgeStore
from discord_ai_assistant.chat_style_runtime import ChatStyleRuntime
from discord_ai_assistant.chat_style_store import ChatStyleStore
from discord_ai_assistant.knowledge_enrichment import GUILD_LEXICON_PROFILE_KEY, KnowledgeEnrichmentStore
from discord_ai_assistant.storage.agent_database import AgentDatabase


UTC = timezone.utc
BASE = datetime(2026, 9, 10, 0, 0, tzinfo=UTC)


class KnowledgeEnrichmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = AgentDatabase(Path(self.tempdir.name) / "assistant.sqlite3")
        self.chat_store = ChatStyleStore(self.database)
        self.app_knowledge = AppKnowledgeStore(self.database)
        self.enrichment = KnowledgeEnrichmentStore(self.database, self.app_knowledge)
        self._message_id = 50_000

    def tearDown(self) -> None:
        self.database.close()
        self.tempdir.cleanup()

    def _samples(
        self,
        *,
        guild_id: int,
        user_id: int,
        term: str,
        count: int = 3,
        session_key: str | None = "session-a",
        start: datetime = BASE,
        is_public: bool = True,
    ):
        for index in range(count):
            self._message_id += 1
            self.chat_store.record_sample(
                guild_id=guild_id,
                user_id=user_id,
                channel_id=100,
                message_id=self._message_id,
                content=f"{term} 是我們常用的快速傳送說法 {index}",
                observed_at=start + timedelta(minutes=index),
                session_key=session_key,
                is_reply=False,
                is_bot=False,
                is_dm=False,
                has_stickers=False,
                is_system=False,
                is_public_evidence=is_public,
            )
        return self.chat_store.list_samples(guild_id, user_id, now=start + timedelta(hours=1))

    def _register(
        self,
        *,
        guild_id: int,
        user_id: int,
        term: str = "alpha-hop",
        meaning: str = "在這個社群表示快速傳送",
        session_key: str | None = "session-a",
        count: int = 3,
        start: datetime = BASE,
        is_public: bool = True,
    ):
        samples = self._samples(
            guild_id=guild_id,
            user_id=user_id,
            term=term,
            count=count,
            session_key=session_key,
            start=start,
            is_public=is_public,
        )
        return self.enrichment.register_personal_understanding(
            guild_id,
            user_id,
            term=term,
            meaning=meaning,
            confidence=0.9,
            samples=samples,
            now=start + timedelta(hours=1),
        )

    def test_personal_term_needs_three_occurrences_and_one_user_stays_private(self) -> None:
        samples = self._samples(guild_id=1, user_id=10, term="alpha-hop", count=2)
        self.assertIsNone(
            self.enrichment.register_personal_understanding(
                1,
                10,
                term="alpha-hop",
                meaning="快速傳送",
                confidence=0.9,
                samples=samples,
                now=BASE + timedelta(hours=1),
            )
        )
        self.assertIsNone(self.enrichment.get_personal_term(1, 10, "alpha-hop"))

        personal = self._register(guild_id=1, user_id=10, count=1, start=BASE + timedelta(minutes=2))
        self.assertIsNotNone(personal)
        assert personal is not None
        self.assertEqual(personal.occurrence_count, 3)
        self.assertEqual(personal.user_id, 10)
        self.assertIsNone(self.enrichment.get_guild_lexicon_entry(1, "alpha-hop", include_candidate=True))
        self.assertIsNone(self.app_knowledge.get_profile(1, GUILD_LEXICON_PROFILE_KEY))

    def test_two_users_same_session_create_candidate_but_not_active_knowledge(self) -> None:
        self._register(guild_id=1, user_id=10, session_key="same-session")
        self._register(guild_id=1, user_id=11, session_key="same-session", start=BASE + timedelta(hours=2))

        entry = self.enrichment.get_guild_lexicon_entry(1, "alpha-hop", include_candidate=True)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.status, "candidate")
        self.assertEqual(entry.distinct_user_count, 2)
        self.assertEqual(entry.distinct_session_count, 1)
        self.assertEqual(self.enrichment.search_guild_lexicon(1, "alpha-hop"), [])
        self.assertIsNone(self.app_knowledge.get_profile(1, GUILD_LEXICON_PROFILE_KEY))

    def test_two_users_in_distinct_sessions_activate_and_reuse_app_knowledge_store(self) -> None:
        self._register(guild_id=1, user_id=10, session_key="session-a")
        self._register(guild_id=1, user_id=11, session_key="session-b", start=BASE + timedelta(hours=2))

        entry = self.enrichment.get_guild_lexicon_entry(1, "alpha-hop", include_candidate=True)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.status, "active")
        self.assertEqual(entry.distinct_user_count, 2)
        self.assertGreaterEqual(entry.distinct_session_count, 2)
        active = self.enrichment.search_guild_lexicon(1, "alpha-hop")
        self.assertEqual([item.id for item in active], [entry.id])

        profile = self.app_knowledge.get_profile(1, GUILD_LEXICON_PROFILE_KEY)
        self.assertIsNotNone(profile)
        term = self.app_knowledge.get_term(1, GUILD_LEXICON_PROFILE_KEY, "alpha-hop")
        self.assertIsNotNone(term)
        assert term is not None
        self.assertEqual(term.explanation, "在這個社群表示快速傳送")

    def test_three_users_activate_even_when_session_is_the_same(self) -> None:
        for offset, user_id in enumerate((10, 11, 12)):
            self._register(
                guild_id=1,
                user_id=user_id,
                session_key="same-session",
                start=BASE + timedelta(hours=offset * 2),
            )
        entry = self.enrichment.get_guild_lexicon_entry(1, "alpha-hop", include_candidate=True)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.status, "active")
        self.assertEqual(entry.distinct_user_count, 3)

    def test_guild_lexicon_is_guild_scoped(self) -> None:
        self._register(guild_id=1, user_id=10, session_key="a")
        self._register(guild_id=1, user_id=11, session_key="b", start=BASE + timedelta(hours=2))
        self._register(guild_id=2, user_id=10, session_key="c")

        self.assertEqual(len(self.enrichment.search_guild_lexicon(1, "alpha-hop")), 1)
        self.assertEqual(self.enrichment.search_guild_lexicon(2, "alpha-hop"), [])
        self.assertIsNone(self.enrichment.get_guild_lexicon_entry(2, "alpha-hop", include_candidate=True))
        self.assertIsNotNone(self.enrichment.get_personal_term(2, 10, "alpha-hop"))

    def test_forgetting_lexicon_entry_does_not_delete_personal_or_memory_v2_data(self) -> None:
        memory = self.database.add_user_memory(1, 10, "偏好", "我喜歡咖啡")
        self._register(guild_id=1, user_id=10, session_key="a")
        self._register(guild_id=1, user_id=11, session_key="b", start=BASE + timedelta(hours=2))
        entry = self.enrichment.get_guild_lexicon_entry(1, "alpha-hop", include_candidate=True)
        assert entry is not None

        self.assertTrue(self.enrichment.forget_guild_lexicon_entry(1, entry.id))
        self.assertIsNone(self.enrichment.get_guild_lexicon_entry(1, "alpha-hop", include_candidate=True))
        self.assertIsNotNone(self.enrichment.get_personal_term(1, 10, "alpha-hop"))
        self.assertIsNotNone(self.enrichment.get_personal_term(1, 11, "alpha-hop"))
        row = self.database.connection.execute(
            "SELECT content FROM user_memories WHERE id = ?", (memory.id,)
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "我喜歡咖啡")
        self.assertIsNone(self.app_knowledge.get_term(1, GUILD_LEXICON_PROFILE_KEY, "alpha-hop"))

    def test_lexicon_evidence_schema_contains_no_raw_conversation_content(self) -> None:
        columns = {
            str(row[1])
            for row in self.database.connection.execute(
                "PRAGMA table_info(guild_lexicon_evidence)"
            ).fetchall()
        }
        self.assertTrue({"entry_id", "user_id", "session_key", "source_sample_id", "observed_at"}.issubset(columns))
        self.assertNotIn("content", columns)
        self.assertNotIn("raw_text", columns)

    def test_term_metadata_retains_confidence_state_and_observation_times(self) -> None:
        self._register(guild_id=1, user_id=10, session_key="a")
        self._register(guild_id=1, user_id=11, session_key="b", start=BASE + timedelta(hours=2))
        row = self.database.connection.execute(
            "SELECT confidence, status, first_observed_at, last_observed_at FROM guild_lexicon_entries"
        ).fetchone()
        self.assertEqual((row[0], row[1]), (0.9, "active"))
        self.assertEqual(datetime.fromisoformat(row[2]), BASE)
        self.assertEqual(datetime.fromisoformat(row[3]), BASE + timedelta(hours=2, minutes=2))
        personal = self.database.connection.execute(
            "SELECT status FROM chat_style_terms WHERE guild_id = 1 AND user_id = 10"
        ).fetchone()
        self.assertEqual(personal[0], "active")

    def test_repeated_registration_does_not_duplicate_evidence(self) -> None:
        samples = self._samples(guild_id=1, user_id=10, term="alpha-hop")
        for _ in range(3):
            self.enrichment.register_personal_understanding(
                1, 10, term="alpha-hop", meaning="快速傳送", confidence=0.9,
                samples=samples, now=BASE + timedelta(hours=1))
        self.assertEqual(self.database.connection.execute(
            "SELECT COUNT(*) FROM guild_lexicon_evidence").fetchone()[0], 3)
        self.assertIsNone(self.enrichment.get_guild_lexicon_entry(1, "alpha-hop", include_candidate=True))

    def test_stale_second_user_cannot_promote_existing_evidence(self) -> None:
        self._register(guild_id=1, user_id=10, session_key="a")
        stale = self._samples(guild_id=1, user_id=11, term="alpha-hop", session_key="b")
        self.enrichment.register_personal_understanding(
            1, 11, term="alpha-hop", meaning="在這個社群表示快速傳送", confidence=0.9,
            samples=stale, now=BASE + timedelta(hours=74))
        self.assertIsNone(self.enrichment.get_guild_lexicon_entry(1, "alpha-hop", include_candidate=True))

    def test_private_flags_and_session_keys_cannot_be_forged_on_sample_copies(self) -> None:
        self._register(guild_id=1, user_id=10, session_key="a")
        private = self._samples(guild_id=1, user_id=11, term="alpha-hop", is_public=False)
        self.enrichment.register_personal_understanding(
            1, 11, term="alpha-hop", meaning="在這個社群表示快速傳送", confidence=0.9,
            samples=[replace(s, is_public_evidence=True, session_key="forged") for s in private],
            now=BASE + timedelta(hours=1))
        self.assertIsNone(self.enrichment.get_guild_lexicon_entry(1, "alpha-hop", include_candidate=True))

    def test_learning_disabled_blocks_registration_from_existing_samples(self) -> None:
        samples = self._samples(guild_id=1, user_id=10, term="alpha-hop")
        self.chat_store.set_learning_enabled(1, 10, False, updated_at=BASE + timedelta(minutes=3))
        self.assertIsNone(self.enrichment.register_personal_understanding(
            1, 10, term="alpha-hop", meaning="快速傳送", confidence=0.9,
            samples=samples, now=BASE + timedelta(hours=1)))
        self.assertEqual(self.database.connection.execute("SELECT COUNT(*) FROM guild_lexicon_evidence").fetchone()[0], 0)

    def test_chat_style_runtime_reads_existing_session_context_without_mutating_it(self) -> None:
        session_state = SimpleNamespace(
            context_for_message=lambda channel_id, message_id: SimpleNamespace(session_key="resolved-session")
        )
        runtime = ChatStyleRuntime(SimpleNamespace(), self.chat_store, SimpleNamespace(), session_state)
        message = SimpleNamespace(channel=SimpleNamespace(id=100), id=999)
        self.assertEqual(runtime._session_key(message), "resolved-session")

    def test_personal_terms_are_private_to_both_guild_and_user(self) -> None:
        self._register(guild_id=1, user_id=10)
        self.assertIsNone(self.enrichment.get_personal_term(1, 11, "alpha-hop"))
        self.assertIsNone(self.enrichment.get_personal_term(2, 10, "alpha-hop"))
        self.assertEqual(self.database.connection.execute("SELECT COUNT(*) FROM user_memories").fetchone()[0], 0)
        row = self.database.connection.execute("SELECT guild_id, user_id FROM chat_style_terms").fetchone()
        self.assertEqual(tuple(row), (1, 10))

    def test_one_user_many_sessions_never_creates_a_guild_candidate(self) -> None:
        for offset in range(10):
            self._register(guild_id=1, user_id=10, session_key=f"session-{offset}",
                           start=BASE + timedelta(hours=offset))
        self.assertIsNotNone(self.enrichment.get_personal_term(1, 10, "alpha-hop"))
        self.assertIsNone(self.enrichment.get_guild_lexicon_entry(1, "alpha-hop", include_candidate=True))
        self.assertIsNone(self.app_knowledge.get_profile(1, GUILD_LEXICON_PROFILE_KEY))

    def test_private_samples_keep_b_layer_data_but_cannot_create_personal_or_guild_terms(self) -> None:
        self._register(guild_id=1, user_id=10, session_key="a", is_public=False)
        self._register(guild_id=1, user_id=11, session_key="b")
        self.assertEqual(len(self.chat_store.list_samples(1, 10, now=BASE + timedelta(hours=1))), 3)
        self.assertIsNone(self.enrichment.get_personal_term(1, 10, "alpha-hop"))
        self.assertIsNone(self.enrichment.get_guild_lexicon_entry(1, "alpha-hop", include_candidate=True))

    def test_stale_or_forged_samples_cannot_supply_new_evidence(self) -> None:
        samples = self._samples(guild_id=1, user_id=10, term="alpha-hop")
        for now, supplied in (
            (BASE + timedelta(hours=74), samples),
            (BASE + timedelta(hours=74), [replace(s, observed_at=BASE + timedelta(hours=74)) for s in samples]),
        ):
            self.assertIsNone(self.enrichment.register_personal_understanding(
                1, 10, term="alpha-hop", meaning="快速傳送", confidence=0.9, samples=supplied, now=now,
            ))
        self.assertIsNone(self.enrichment.get_personal_term(1, 10, "alpha-hop"))
        self.assertEqual(self.database.connection.execute("SELECT COUNT(*) FROM guild_lexicon_evidence").fetchone()[0], 0)

    def test_other_owners_and_duplicate_sample_ids_do_not_satisfy_personal_threshold(self) -> None:
        samples = self._samples(guild_id=1, user_id=10, term="alpha-hop", count=1)
        for guild_id, user_id, supplied in ((1, 10, samples * 3), (1, 11, samples), (2, 10, samples)):
            self.assertIsNone(self.enrichment.register_personal_understanding(
                guild_id, user_id, term="alpha-hop", meaning="快速傳送", confidence=0.9,
                samples=supplied, now=BASE + timedelta(hours=1),
            ))

    def test_unknown_sessions_do_not_activate_two_user_candidate(self) -> None:
        self._register(guild_id=1, user_id=10, session_key=None)
        self._register(guild_id=1, user_id=11, session_key=None)
        entry = self.enrichment.get_guild_lexicon_entry(1, "alpha-hop", include_candidate=True)
        self.assertEqual((entry.status, entry.distinct_session_count), ("candidate", 0))
        self.assertIsNone(self.enrichment.get_guild_lexicon_entry(1, "alpha-hop"))

    def test_guild_threshold_does_not_require_three_samples_from_each_user(self) -> None:
        self._register(guild_id=1, user_id=10, count=1, session_key="a")
        self._register(guild_id=1, user_id=11, count=1, session_key="b")
        self.assertIsNone(self.enrichment.get_personal_term(1, 10, "alpha-hop"))
        self.assertEqual(self.enrichment.get_guild_lexicon_entry(1, "alpha-hop").status, "active")

    def test_existing_lexicon_evidence_does_not_expire_with_raw_sample_ttl(self) -> None:
        self._register(guild_id=1, user_id=10, session_key="a")
        self.assertEqual(self.chat_store.list_samples(1, 10, now=BASE + timedelta(hours=80)), [])
        self._register(guild_id=1, user_id=11, session_key="b", start=BASE + timedelta(hours=80))
        self.assertEqual(self.enrichment.get_guild_lexicon_entry(1, "alpha-hop").distinct_user_count, 2)

    def test_normalized_term_and_meaning_merge_consistent_evidence(self) -> None:
        self._register(guild_id=1, user_id=10, term=" Alpha-Hop ", meaning=" Quick   Teleport ", session_key="a")
        self._register(guild_id=1, user_id=11, term="alpha-hop", meaning="quick teleport", session_key="b")
        entry = self.enrichment.get_guild_lexicon_entry(1, "  ALPHA-HOP  ")
        self.assertEqual((entry.status, entry.distinct_user_count), ("active", 2))

    def test_conflicting_meanings_cannot_combine_users_or_sessions(self) -> None:
        self._register(guild_id=1, user_id=10, meaning="快速傳送", session_key="a")
        self._register(guild_id=1, user_id=11, meaning="快速傳送", session_key="a")
        self._register(guild_id=1, user_id=12, meaning="某種武器", session_key="b")
        entry = self.enrichment.get_guild_lexicon_entry(1, "alpha-hop", include_candidate=True)
        self.assertEqual((entry.status, entry.distinct_user_count, entry.distinct_session_count), ("candidate", 2, 1))
        self.assertIsNone(self.enrichment.get_guild_lexicon_entry(1, "alpha-hop"))
        self.assertIsNone(self.app_knowledge.get_profile(1, GUILD_LEXICON_PROFILE_KEY))

    def test_conflicting_later_consensus_cannot_overwrite_active_meaning(self) -> None:
        self._register(guild_id=1, user_id=10, meaning="快速傳送", session_key="a")
        self._register(guild_id=1, user_id=11, meaning="快速傳送", session_key="b")
        for offset, user_id in enumerate((12, 13, 14)):
            self._register(guild_id=1, user_id=user_id, meaning="某種武器", session_key=f"other-{offset}")
        entry = self.enrichment.get_guild_lexicon_entry(1, "alpha-hop")
        self.assertEqual(entry.distinct_user_count, 2)
        self.assertEqual(self.app_knowledge.get_term(1, GUILD_LEXICON_PROFILE_KEY, "alpha-hop").explanation, "快速傳送")

        conflict_id = self.database.connection.execute(
            "SELECT id FROM guild_lexicon_entries WHERE guild_id = 1 AND meaning_key = ?", ("某種武器",)
        ).fetchone()[0]
        self.enrichment.forget_guild_lexicon_entry(1, conflict_id)
        self.assertEqual(self.app_knowledge.get_term(1, GUILD_LEXICON_PROFILE_KEY, "alpha-hop").explanation, "快速傳送")

    def test_forget_is_scoped_to_one_guild_entry_and_its_evidence(self) -> None:
        for guild_id in (1, 2):
            self._register(guild_id=guild_id, user_id=10, session_key="a")
            self._register(guild_id=guild_id, user_id=11, session_key="b")
        entry = self.enrichment.get_guild_lexicon_entry(1, "alpha-hop")
        self.assertFalse(self.enrichment.forget_guild_lexicon_entry(2, entry.id))
        self.assertTrue(self.enrichment.forget_guild_lexicon_entry(1, entry.id))
        self.assertEqual(self.database.connection.execute(
            "SELECT COUNT(*) FROM guild_lexicon_evidence WHERE entry_id = ?", (entry.id,),
        ).fetchone()[0], 0)
        self.assertIsNotNone(self.enrichment.get_guild_lexicon_entry(2, "alpha-hop"))
        self.assertIsNotNone(self.enrichment.get_personal_term(1, 10, "alpha-hop"))

    def test_runtime_session_lookup_is_read_only_and_fail_soft(self) -> None:
        message = SimpleNamespace(channel=SimpleNamespace(id=100), id=999)
        for result in (None, RuntimeError("unavailable")):
            state = SimpleNamespace(context_for_message=Mock(), observe=Mock(side_effect=AssertionError("must not observe")))
            if isinstance(result, Exception):
                state.context_for_message.side_effect = result
            else:
                state.context_for_message.return_value = result
            runtime = ChatStyleRuntime(SimpleNamespace(), self.chat_store, SimpleNamespace(), state)
            self.assertIsNone(runtime._session_key(message))
            state.context_for_message.assert_called_once_with(100, 999)
            state.observe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
