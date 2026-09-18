from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from discord_ai_assistant.ai.memory_phase2 import PassiveMemoryV2Draft
from discord_ai_assistant.ai.memory_phase4 import MemoryObservation
from discord_ai_assistant.memory_v2_runtime import store_passive_memory_v2_drafts
from discord_ai_assistant.storage.agent_database import AgentDatabase


class SharedMemoryReferenceReconciliationLiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = AgentDatabase(Path(self.tempdir.name) / "assistant.sqlite3")

    def tearDown(self) -> None:
        self.database.close()
        self.tempdir.cleanup()

    @staticmethod
    def _observation(
        content: str,
        *,
        user_id: int,
        message_id: int,
        session_key: str = "5:1",
        monotonic_at: float = 1.0,
    ) -> MemoryObservation:
        return MemoryObservation(
            monotonic_at=monotonic_at,
            content=content,
            channel_id=5,
            message_id=message_id,
            created_at="2026-09-08T06:00:00+00:00",
            session_key=session_key,
            domain="game",
            subdomain="minecraft",
            topic="minecraft",
            shared_allowed=True,
            participant_ids=(user_id,),
        )

    @staticmethod
    def _draft(
        content: str,
        *,
        entity_type: str | None = None,
        entity: str | None = None,
    ) -> PassiveMemoryV2Draft:
        return PassiveMemoryV2Draft(
            category="事件",
            content=content,
            confidence=0.95,
            importance=2,
            domain="game",
            subdomain="minecraft",
            memory_kind="shared_episode",
            entity_type=entity_type,
            entity=entity,
            retention="shared",
            shared_candidate=True,
            shared_group_event=True,
        )

    def _activate_live_end_farm_episode(self) -> int:
        first_message = (
            "Minecraft 上次終界農場的收集箱被人挖掉，"
            "結果大家刷到的終界珍珠全掉進虛空"
        )
        second_message = (
            "Minecraft 上次終界農場的收集箱被挖走，"
            "害大家刷出來的終界珍珠全部掉到虛空裡"
        )
        first_draft = self._draft(
            "專案 Minecraft：終界農場發生過收集箱被挖掉、"
            "終界珍珠全數掉落虛空的意外事故。"
        )
        second_draft = self._draft(
            "Minecraft 終界農場的收集箱曾被挖走，導致終界珍珠全部掉進虛空。"
        )

        store_passive_memory_v2_drafts(
            self.database,
            1,
            10,
            [first_draft],
            observations=[self._observation(first_message, user_id=10, message_id=701)],
        )
        store_passive_memory_v2_drafts(
            self.database,
            1,
            11,
            [second_draft],
            observations=[
                self._observation(
                    second_message,
                    user_id=11,
                    message_id=702,
                    monotonic_at=2.0,
                )
            ],
        )
        row = self.database.connection.execute(
            "SELECT id, status FROM guild_memories WHERE guild_id = 1 ORDER BY id LIMIT 1"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "active")
        return int(row["id"])

    def test_same_session_short_reference_reuses_active_live_episode_without_duplicate(self) -> None:
        memory_id = self._activate_live_end_farm_episode()
        short_reference = "Minecraft 那次終界農場的箱子被挖掉真的笑死"
        live_reference_draft = self._draft(
            "專案 Minecraft：曾發生終界農場箱子被挖掉的趣事。",
            entity_type="event",
            entity="終界農場箱子被挖事件",
        )

        store_passive_memory_v2_drafts(
            self.database,
            1,
            11,
            [live_reference_draft],
            observations=[
                self._observation(
                    short_reference,
                    user_id=11,
                    message_id=703,
                    monotonic_at=3.0,
                )
            ],
        )

        rows = self.database.connection.execute(
            "SELECT id, status, reinforcement_count FROM guild_memories WHERE guild_id = 1 ORDER BY id"
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0]["id"]), memory_id)
        self.assertEqual(rows[0]["status"], "active")
        self.assertEqual(int(rows[0]["reinforcement_count"]), 0)

        evidence = self.database.connection.execute(
            """SELECT COUNT(*) AS rows,
                      COUNT(DISTINCT session_key) AS sessions
               FROM guild_memory_evidence WHERE guild_memory_id = ?""",
            (memory_id,),
        ).fetchone()
        self.assertEqual(int(evidence["rows"]), 3)
        self.assertEqual(int(evidence["sessions"]), 1)

    def test_same_session_different_incident_at_same_place_stays_separate(self) -> None:
        memory_id = self._activate_live_end_farm_episode()
        different_incident = "Minecraft 那次終界農場的傳送門壞掉真的超慘"
        different_draft = self._draft(
            "專案 Minecraft：終界農場的傳送門曾損壞，導致成員無法返回主世界。",
            entity_type="設施",
            entity="終界農場傳送門",
        )

        store_passive_memory_v2_drafts(
            self.database,
            1,
            11,
            [different_draft],
            observations=[
                self._observation(
                    different_incident,
                    user_id=11,
                    message_id=705,
                    monotonic_at=4.0,
                )
            ],
        )

        rows = self.database.connection.execute(
            "SELECT id, status, reinforcement_count FROM guild_memories WHERE guild_id = 1 ORDER BY id"
        ).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(int(rows[0]["id"]), memory_id)
        self.assertEqual(rows[0]["status"], "active")
        self.assertEqual(int(rows[0]["reinforcement_count"]), 0)
        self.assertEqual(rows[1]["status"], "candidate")

    def test_new_session_explicit_reference_reinforces_active_live_episode_without_duplicate(self) -> None:
        memory_id = self._activate_live_end_farm_episode()
        later_reference = (
            "Minecraft 上次終界農場收集箱那個事故還記得嗎，"
            "這次先別讓人亂挖"
        )
        later_reference_draft = self._draft(
            "專案 Minecraft：曾發生終界農場收集箱被挖掉的事故。",
            entity_type="event",
            entity="終界農場收集箱事故",
        )

        store_passive_memory_v2_drafts(
            self.database,
            1,
            11,
            [later_reference_draft],
            observations=[
                self._observation(
                    later_reference,
                    user_id=11,
                    message_id=704,
                    session_key="5:2",
                    monotonic_at=1900.0,
                )
            ],
        )

        rows = self.database.connection.execute(
            "SELECT id, status, retention, reinforcement_count FROM guild_memories WHERE guild_id = 1 ORDER BY id"
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0]["id"]), memory_id)
        self.assertEqual(rows[0]["status"], "active")
        self.assertEqual(rows[0]["retention"], "shared")
        self.assertEqual(int(rows[0]["reinforcement_count"]), 1)

        evidence = self.database.connection.execute(
            """SELECT COUNT(*) AS rows,
                      COUNT(DISTINCT session_key) AS sessions
               FROM guild_memory_evidence WHERE guild_memory_id = ?""",
            (memory_id,),
        ).fetchone()
        self.assertEqual(int(evidence["rows"]), 3)
        self.assertEqual(int(evidence["sessions"]), 2)


if __name__ == "__main__":
    unittest.main()
