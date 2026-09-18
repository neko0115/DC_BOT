from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from discord_ai_assistant.ai.memory_phase2 import (
    PassiveMemoryV2Draft,
    is_unified_passive_memory_candidate,
)
from discord_ai_assistant.ai.memory_phase4 import MemoryObservation
from discord_ai_assistant.memory_v2_runtime import store_passive_memory_v2_drafts
from discord_ai_assistant.storage.agent_database import AgentDatabase


class MemorySharedRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = AgentDatabase(Path(self.tempdir.name) / "assistant.sqlite3")

    def tearDown(self) -> None:
        self.database.close()
        self.tempdir.cleanup()

    @staticmethod
    def _draft() -> PassiveMemoryV2Draft:
        return PassiveMemoryV2Draft(
            category="事件",
            content="上次 Minecraft 有人拆紅石導致整個農場停掉",
            confidence=0.92,
            importance=2,
            domain="game",
            subdomain="minecraft",
            memory_kind="shared_episode",
            entity_type="event",
            entity="redstone_farm_accident",
            retention="shared",
            shared_candidate=True,
            shared_group_event=True,
        )

    @staticmethod
    def _observation(*, shared_allowed: bool) -> MemoryObservation:
        return MemoryObservation(
            monotonic_at=1.0,
            content="上次 Minecraft 有人拆紅石導致整個農場停掉",
            channel_id=5,
            message_id=100,
            created_at="2026-09-06T10:00:00+00:00",
            session_key="5:1",
            domain="game",
            subdomain="minecraft",
            topic="redstone",
            shared_allowed=shared_allowed,
            # Session participants are ambient conversation context, not proof that
            # both users corroborated this exact extracted event.
            participant_ids=(10, 11),
        )

    def test_public_game_episode_enters_unified_passive_gate(self) -> None:
        self.assertTrue(
            is_unified_passive_memory_candidate(
                "上次 Minecraft 有人拆紅石導致整個農場停掉"
            )
        )

    def test_shared_only_draft_goes_to_guild_store_not_personal_store(self) -> None:
        stored = store_passive_memory_v2_drafts(
            self.database,
            1,
            10,
            [self._draft()],
            observations=[self._observation(shared_allowed=True)],
        )
        self.assertEqual(stored, 0)
        guild_row = self.database.connection.execute(
            "SELECT status FROM guild_memories WHERE guild_id = 1"
        ).fetchone()
        personal_count = int(
            self.database.connection.execute(
                "SELECT COUNT(*) FROM user_memories WHERE guild_id = 1 AND user_id = 10"
            ).fetchone()[0]
        )
        self.assertIsNotNone(guild_row)
        self.assertEqual(guild_row["status"], "candidate")
        self.assertEqual(personal_count, 0)

    def test_ambient_session_participants_do_not_immediately_promote_one_users_event(self) -> None:
        store_passive_memory_v2_drafts(
            self.database,
            1,
            10,
            [self._draft()],
            observations=[self._observation(shared_allowed=True)],
        )
        row = self.database.connection.execute(
            """SELECT gm.status,
                      COUNT(DISTINCT ge.user_id) AS users,
                      COUNT(DISTINCT ge.session_key) AS sessions
               FROM guild_memories gm
               LEFT JOIN guild_memory_evidence ge ON ge.guild_memory_id = gm.id
               WHERE gm.guild_id = 1
               GROUP BY gm.id, gm.status"""
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(int(row["users"]), 1)
        self.assertEqual(int(row["sessions"]), 1)
        self.assertEqual(row["status"], "candidate")

    def test_per_user_passive_shared_episode_evidence_is_not_group_event(self) -> None:
        store_passive_memory_v2_drafts(
            self.database,
            1,
            10,
            [self._draft()],
            observations=[self._observation(shared_allowed=True)],
        )
        row = self.database.connection.execute(
            "SELECT evidence_kind FROM guild_memory_evidence ORDER BY id LIMIT 1"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["evidence_kind"], "passive")

    def test_shared_evidence_keeps_hashed_event_signature_not_raw_observation(self) -> None:
        store_passive_memory_v2_drafts(
            self.database,
            1,
            10,
            [self._draft()],
            observations=[self._observation(shared_allowed=True)],
        )
        columns = {
            str(row["name"])
            for row in self.database.connection.execute(
                "PRAGMA table_info(guild_memory_evidence)"
            ).fetchall()
        }
        self.assertIn("event_signature", columns)
        row = self.database.connection.execute(
            "SELECT event_signature FROM guild_memory_evidence ORDER BY id LIMIT 1"
        ).fetchone()
        self.assertIsNotNone(row)
        signature = str(row["event_signature"] or "")
        self.assertTrue(signature.startswith("v1:"))
        self.assertLessEqual(len(signature), 4096)
        self.assertNotIn("Minecraft", signature)
        self.assertNotIn("紅石", signature)

    def test_live_blue_ice_event_reconciles_from_original_observation_signatures_without_entity_metadata(self) -> None:
        first_draft = PassiveMemoryV2Draft(
            category="事件",
            content="Minecraft 地獄交通站的藍冰船道曾因苦力怕爆炸而斷裂，導致多人無法返回基地。",
            confidence=0.95,
            importance=2,
            domain="game",
            subdomain="minecraft",
            memory_kind="shared_episode",
            entity_type=None,
            entity=None,
            retention="shared",
            shared_candidate=True,
            shared_group_event=True,
        )
        first_observation = MemoryObservation(
            monotonic_at=1.0,
            content="Minecraft 上次地獄交通站的藍冰船道被苦力怕炸斷，結果大家都沒辦法回基地",
            channel_id=5,
            message_id=501,
            created_at="2026-09-08T04:58:06+00:00",
            session_key="5:1",
            domain="game",
            subdomain="minecraft",
            topic="minecraft",
            shared_allowed=True,
            participant_ids=(10, 11),
        )
        second_draft = PassiveMemoryV2Draft(
            category="專案",
            content="專案 Minecraft：地獄交通站的藍冰船道曾被苦力怕炸毀，導致基地交通中斷。",
            confidence=0.95,
            importance=2,
            project="Minecraft",
            role="event",
            domain="project",
            subdomain="Minecraft",
            memory_kind="project",
            entity_type=None,
            entity=None,
            retention="shared",
            shared_candidate=True,
            shared_group_event=True,
        )
        second_observation = MemoryObservation(
            monotonic_at=2.0,
            content="Minecraft 上次地獄交通站的藍冰船道被苦力怕炸毀，害大家都無法回到基地",
            channel_id=5,
            message_id=502,
            created_at="2026-09-08T05:01:06+00:00",
            session_key="5:1",
            domain="game",
            subdomain="minecraft",
            topic="minecraft",
            shared_allowed=True,
            participant_ids=(10, 11),
        )

        store_passive_memory_v2_drafts(
            self.database,
            1,
            10,
            [first_draft],
            observations=[first_observation],
        )
        store_passive_memory_v2_drafts(
            self.database,
            1,
            11,
            [second_draft],
            observations=[second_observation],
        )

        rows = self.database.connection.execute(
            "SELECT id, status, reinforcement_count FROM guild_memories WHERE guild_id = 1 ORDER BY id"
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "active")
        self.assertEqual(int(rows[0]["reinforcement_count"]), 0)
        evidence = self.database.connection.execute(
            """SELECT COUNT(DISTINCT user_id) AS users,
                      COUNT(DISTINCT session_key) AS sessions
               FROM guild_memory_evidence WHERE guild_memory_id = ?""",
            (int(rows[0]["id"]),),
        ).fetchone()
        self.assertEqual(int(evidence["users"]), 2)
        self.assertEqual(int(evidence["sessions"]), 1)

    def test_observation_signature_does_not_merge_different_incidents_without_entity_metadata(self) -> None:
        first_draft = PassiveMemoryV2Draft(
            category="事件",
            content="Minecraft 地獄交通站的藍冰船道曾被炸斷，導致成員無法返回基地。",
            confidence=0.95,
            importance=2,
            domain="game",
            subdomain="minecraft",
            memory_kind="shared_episode",
            entity_type=None,
            entity=None,
            retention="shared",
            shared_candidate=True,
            shared_group_event=True,
        )
        second_draft = PassiveMemoryV2Draft(
            category="事件",
            content="Minecraft 地獄交通站後來新增終界箱，方便成員整理物資。",
            confidence=0.95,
            importance=2,
            domain="game",
            subdomain="minecraft",
            memory_kind="shared_episode",
            entity_type=None,
            entity=None,
            retention="shared",
            shared_candidate=True,
            shared_group_event=True,
        )
        first_observation = MemoryObservation(
            monotonic_at=1.0,
            content="Minecraft 上次地獄交通站的藍冰船道被苦力怕炸斷，結果大家都沒辦法回基地",
            channel_id=5,
            message_id=601,
            created_at="2026-09-08T05:10:00+00:00",
            session_key="5:1",
            domain="game",
            subdomain="minecraft",
            topic="minecraft",
            shared_allowed=True,
            participant_ids=(10,),
        )
        second_observation = MemoryObservation(
            monotonic_at=2.0,
            content="Minecraft 上次地獄交通站新裝了一個終界箱，大家回基地前可以先整理物資",
            channel_id=5,
            message_id=602,
            created_at="2026-09-08T05:20:00+00:00",
            session_key="5:1",
            domain="game",
            subdomain="minecraft",
            topic="minecraft",
            shared_allowed=True,
            participant_ids=(11,),
        )
        store_passive_memory_v2_drafts(
            self.database,
            1,
            10,
            [first_draft],
            observations=[first_observation],
        )
        store_passive_memory_v2_drafts(
            self.database,
            1,
            11,
            [second_draft],
            observations=[second_observation],
        )
        rows = self.database.connection.execute(
            "SELECT id, status FROM guild_memories WHERE guild_id = 1 ORDER BY id"
        ).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["status"] for row in rows], ["candidate", "candidate"])

    def test_shared_draft_is_dropped_when_channel_policy_disallows_sharing(self) -> None:
        stored = store_passive_memory_v2_drafts(
            self.database,
            1,
            10,
            [self._draft()],
            observations=[self._observation(shared_allowed=False)],
        )
        self.assertEqual(stored, 0)
        count = int(
            self.database.connection.execute("SELECT COUNT(*) FROM guild_memories").fetchone()[0]
        ) if self.database.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='guild_memories'"
        ).fetchone() else 0
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
