from __future__ import annotations

import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

from discord_ai_assistant.storage.agent_database import AgentDatabase


MODULE_NAME = "discord_ai_assistant.ai.memory_shared"
MODULE_SPEC = importlib.util.find_spec(MODULE_NAME)


class MemorySharedModuleTests(unittest.TestCase):
    def test_shared_memory_module_exists(self) -> None:
        self.assertIsNotNone(MODULE_SPEC)


@unittest.skipIf(MODULE_SPEC is None, "memory_shared module not implemented yet")
class MemorySharedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = AgentDatabase(Path(self.tempdir.name) / "assistant.sqlite3")
        module = importlib.import_module(MODULE_NAME)
        self.ensure_schema = getattr(module, "ensure_shared_memory_schema", None)
        self.record_candidate = getattr(module, "record_shared_candidate", None)
        self.search_shared = getattr(module, "search_guild_memories", None)
        self.assertTrue(callable(self.ensure_schema))
        self.assertTrue(callable(self.record_candidate))
        self.assertTrue(callable(self.search_shared))
        self.ensure_schema(self.database)

    def tearDown(self) -> None:
        self.database.close()
        self.tempdir.cleanup()

    def _record(self, **overrides):
        payload = {
            "guild_id": 1,
            "author_id": 10,
            "content": "上次 Minecraft 有人拆紅石導致整個農場停掉",
            "domain": "game",
            "subdomain": "minecraft",
            "memory_kind": "shared_episode",
            "confidence": 0.92,
            "importance": 2,
            "entity_type": "event",
            "entity": "redstone_farm_accident",
            "session_key": "99:1",
            "channel_id": 5,
            "message_id": 100,
            "observed_at": "2026-09-06T10:00:00+00:00",
            "shared_group_event": False,
            "participant_ids": (10,),
        }
        payload.update(overrides)
        return self.record_candidate(self.database, **payload)

    def _row(self):
        return self.database.connection.execute(
            "SELECT * FROM guild_memories WHERE guild_id = 1 ORDER BY id LIMIT 1"
        ).fetchone()

    def test_schema_keeps_raw_content_out_of_evidence_table(self) -> None:
        tables = {
            str(row["name"])
            for row in self.database.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        self.assertIn("guild_memories", tables)
        self.assertIn("guild_memory_evidence", tables)
        evidence_columns = {
            str(row["name"])
            for row in self.database.connection.execute("PRAGMA table_info(guild_memory_evidence)").fetchall()
        }
        self.assertNotIn("content", evidence_columns)
        self.assertTrue({"guild_memory_id", "user_id", "channel_id", "message_id", "observed_at", "session_key"}.issubset(evidence_columns))

    def test_single_observation_stays_candidate_and_is_not_searchable(self) -> None:
        self.assertIsNotNone(self._record())
        row = self._row()
        self.assertEqual(row["status"], "candidate")
        self.assertEqual(self.search_shared(self.database, 1, "Minecraft 紅石農場"), [])

    def test_stale_candidate_expires_before_late_evidence_can_promote_it(self) -> None:
        self._record()
        first = self._row()
        self.assertEqual(first["status"], "candidate")
        self.assertEqual(first["expires_at"], "2026-10-06T10:00:00+00:00")

        self._record(
            author_id=11,
            message_id=101,
            session_key="99:2",
            participant_ids=(11,),
            observed_at="2026-10-07T10:00:00+00:00",
        )
        rows = self.database.connection.execute(
            "SELECT id, status, expires_at FROM guild_memories WHERE guild_id = 1 ORDER BY id"
        ).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["status"], "expired")
        self.assertEqual(rows[1]["status"], "candidate")
        self.assertEqual(rows[1]["expires_at"], "2026-11-06T10:00:00+00:00")

    def test_same_entity_does_not_merge_unrelated_group_episodes(self) -> None:
        self._record(
            content="Minecraft 昨晚基地倉庫被苦力怕炸掉一半",
            entity="main_base",
        )
        self._record(
            content="Minecraft 今天大家在基地旁邊蓋了一座櫻花塔",
            entity="main_base",
            message_id=120,
            session_key="99:2",
            observed_at="2026-09-07T10:00:00+00:00",
        )
        rows = self.database.connection.execute(
            "SELECT id, status, content FROM guild_memories WHERE guild_id = 1 ORDER BY id"
        ).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["status"] for row in rows], ["candidate", "candidate"])

    def test_same_entity_can_still_reconcile_a_clear_paraphrase(self) -> None:
        self._record(
            content="Minecraft 昨晚基地倉庫被苦力怕炸掉一半",
            entity="main_base",
        )
        self._record(
            content="Minecraft 昨晚苦力怕把基地倉庫炸掉一半",
            entity="main_base",
            author_id=11,
            message_id=121,
            session_key="99:2",
            observed_at="2026-09-07T10:00:00+00:00",
        )
        rows = self.database.connection.execute(
            "SELECT id, status FROM guild_memories WHERE guild_id = 1 ORDER BY id"
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "active")

    def test_second_distinct_member_promotes_candidate(self) -> None:
        self._record()
        self._record(
            author_id=11,
            message_id=101,
            session_key="99:1",
            participant_ids=(10, 11),
            observed_at="2026-09-06T10:01:00+00:00",
        )
        row = self._row()
        self.assertEqual(row["status"], "active")
        self.assertEqual(row["retention"], "shared")
        self.assertEqual(int(row["reinforcement_count"]), 0)
        self.assertIsNotNone(row["expires_at"])

    def test_second_distinct_session_promotes_candidate(self) -> None:
        self._record()
        self._record(
            message_id=102,
            session_key="99:2",
            observed_at="2026-09-07T10:00:00+00:00",
        )
        self.assertEqual(self._row()["status"], "active")

    def test_group_event_can_promote_immediately_with_multiple_participants(self) -> None:
        self._record(
            shared_group_event=True,
            participant_ids=(10, 11),
        )
        self.assertEqual(self._row()["status"], "active")

    def test_reinforcement_ladder_counts_at_most_once_per_session(self) -> None:
        self._record(shared_group_event=True, participant_ids=(10, 11))
        self._record(message_id=110, session_key="99:2", observed_at="2026-09-10T10:00:00+00:00")
        row = self._row()
        self.assertEqual(int(row["reinforcement_count"]), 1)
        self.assertEqual(row["retention"], "shared")

        self._record(message_id=111, session_key="99:2", observed_at="2026-09-10T10:05:00+00:00")
        self.assertEqual(int(self._row()["reinforcement_count"]), 1)

        self._record(message_id=112, session_key="99:3", observed_at="2026-09-20T10:00:00+00:00")
        row = self._row()
        self.assertEqual(int(row["reinforcement_count"]), 2)
        self.assertEqual(row["retention"], "medium")
        self.assertIsNotNone(row["expires_at"])

        self._record(message_id=113, session_key="99:4", observed_at="2026-10-01T10:00:00+00:00")
        row = self._row()
        self.assertEqual(int(row["reinforcement_count"]), 3)
        self.assertEqual(row["retention"], "long")
        self.assertIsNone(row["expires_at"])

    def test_private_gossip_and_sensitive_profiles_never_form_shared_memory(self) -> None:
        rejected = (
            "聽說 A 喜歡 B",
            "我今晚住台北市某路 123 號 5 樓，等等直接來",
            "A 的政治立場很偏某黨",
            "B 最近被診斷出糖尿病",
            "C 的銀行帳號是 123456789",
        )
        for index, content in enumerate(rejected, start=200):
            with self.subTest(content=content):
                result = self._record(
                    content=content,
                    message_id=index,
                    session_key=f"private:{index}",
                    shared_group_event=True,
                    participant_ids=(10, 11),
                )
                self.assertIsNone(result)
        count = int(self.database.connection.execute("SELECT COUNT(*) FROM guild_memories").fetchone()[0])
        self.assertEqual(count, 0)

    def test_search_returns_only_active_matching_subdomain(self) -> None:
        self._record(shared_group_event=True, participant_ids=(10, 11))
        self._record(
            content="上次原神有人手滑把保底抽掉了",
            subdomain="genshin",
            entity="gacha_accident",
            message_id=300,
            session_key="genshin:1",
            shared_group_event=True,
            participant_ids=(10, 12),
        )
        matches = self.search_shared(
            self.database,
            1,
            "Minecraft 紅石農場",
            domain="game",
            subdomain="minecraft",
            limit=3,
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].subdomain, "minecraft")
        self.assertIn("紅石", matches[0].content)


if __name__ == "__main__":
    unittest.main()
