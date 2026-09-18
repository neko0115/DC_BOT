from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from discord_ai_assistant.ai.memory_shared import (
    build_event_signature,
    ensure_shared_memory_schema,
    record_shared_candidate,
)
from discord_ai_assistant.storage.agent_database import AgentDatabase


class SharedMemoryLiveReconciliationRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = AgentDatabase(Path(self.tempdir.name) / "assistant.sqlite3")
        ensure_shared_memory_schema(self.database)

    def tearDown(self) -> None:
        self.database.close()
        self.tempdir.cleanup()

    def test_live_minecraft_paraphrases_reconcile_without_entity_metadata(self) -> None:
        first = record_shared_candidate(
            self.database,
            guild_id=1,
            author_id=10,
            content="專案 Minecraft：紅石活塞橋曾被拆除，導致成員無法過河。",
            domain="game",
            subdomain="minecraft",
            memory_kind="shared_episode",
            confidence=1.0,
            importance=2,
            entity_type=None,
            entity=None,
            session_key="99:1",
            channel_id=5,
            message_id=100,
            observed_at="2026-09-06T17:11:41+00:00",
            shared_group_event=True,
            participant_ids=(10,),
        )
        second = record_shared_candidate(
            self.database,
            guild_id=1,
            author_id=11,
            content="專案 Minecraft：紅石活塞橋曾遭拆除，導致無法過河，影響全體成員通行。",
            domain="game",
            subdomain="minecraft",
            memory_kind="shared_episode",
            confidence=0.95,
            importance=2,
            entity_type=None,
            entity=None,
            session_key="99:1",
            channel_id=5,
            message_id=101,
            observed_at="2026-09-06T17:25:40+00:00",
            shared_group_event=True,
            participant_ids=(10, 11),
        )

        self.assertEqual(first, second)
        rows = self.database.connection.execute(
            "SELECT id, status, reinforcement_count FROM guild_memories WHERE guild_id = 1 ORDER BY id"
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "active")
        self.assertEqual(int(rows[0]["reinforcement_count"]), 0)
        evidence = self.database.connection.execute(
            "SELECT COUNT(DISTINCT user_id) AS users FROM guild_memory_evidence WHERE guild_memory_id = ?",
            (int(rows[0]["id"]),),
        ).fetchone()
        self.assertEqual(int(evidence["users"]), 2)

    def test_live_villager_trading_hall_paraphrases_reconcile_without_entity_metadata(self) -> None:
        first = record_shared_candidate(
            self.database,
            guild_id=1,
            author_id=10,
            content="專案 Minecraft：村民交易所曾因苦力怕爆炸而損毀，導致附魔書交易服務中斷。",
            domain="game",
            subdomain="minecraft",
            memory_kind="shared_episode",
            confidence=0.95,
            importance=2,
            entity_type=None,
            entity=None,
            session_key="100:1",
            channel_id=5,
            message_id=300,
            observed_at="2026-09-07T19:14:37+00:00",
            shared_group_event=True,
            participant_ids=(10,),
        )
        second = record_shared_candidate(
            self.database,
            guild_id=1,
            author_id=11,
            content="專案 Minecraft：村民交易所遭苦力怕炸毀，導致無法交換附魔書，屬於群體共有的負面事件。",
            domain="game",
            subdomain="minecraft",
            memory_kind="shared_episode",
            confidence=0.95,
            importance=2,
            entity_type=None,
            entity=None,
            session_key="100:1",
            channel_id=5,
            message_id=301,
            observed_at="2026-09-07T19:25:00+00:00",
            shared_group_event=True,
            participant_ids=(10, 11),
        )

        self.assertEqual(first, second)
        rows = self.database.connection.execute(
            "SELECT id, status, reinforcement_count FROM guild_memories WHERE guild_id = 1 ORDER BY id"
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "active")
        self.assertEqual(int(rows[0]["reinforcement_count"]), 0)
        evidence = self.database.connection.execute(
            "SELECT COUNT(DISTINCT user_id) AS users FROM guild_memory_evidence WHERE guild_memory_id = ?",
            (int(rows[0]["id"]),),
        ).fetchone()
        self.assertEqual(int(evidence["users"]), 2)

    def test_high_text_similarity_does_not_merge_opposite_bridge_state(self) -> None:
        first = record_shared_candidate(
            self.database,
            guild_id=1,
            author_id=10,
            content="專案 Minecraft：紅石活塞橋曾被拆除，導致成員無法過河。",
            domain="game",
            subdomain="minecraft",
            memory_kind="shared_episode",
            confidence=1.0,
            importance=2,
            entity_type=None,
            entity=None,
            session_key="99:1",
            channel_id=5,
            message_id=200,
            observed_at="2026-09-06T17:11:41+00:00",
            shared_group_event=True,
            participant_ids=(10,),
        )
        second = record_shared_candidate(
            self.database,
            guild_id=1,
            author_id=11,
            content="專案 Minecraft：紅石活塞橋已經修復完成，現在大家可以正常過河。",
            domain="game",
            subdomain="minecraft",
            memory_kind="shared_episode",
            confidence=0.95,
            importance=2,
            entity_type=None,
            entity=None,
            session_key="99:1",
            channel_id=5,
            message_id=201,
            observed_at="2026-09-06T17:30:00+00:00",
            shared_group_event=True,
            participant_ids=(10, 11),
        )

        self.assertNotEqual(first, second)
        rows = self.database.connection.execute(
            "SELECT id, status FROM guild_memories WHERE guild_id = 1 ORDER BY id"
        ).fetchall()
        self.assertEqual(len(rows), 2)

    def test_live_villager_toolbox_corroboration_reconciles_from_original_observations(self) -> None:
        first_message = (
            "昨天 Minecraft 村民交易所的工具箱被人挖掉了，"
            "結果我們存的綠寶石全部掉進岩漿"
        )
        second_message = (
            "昨天 Minecraft 村民交易所那個工具箱真的被拆走了，"
            "害我們存的綠寶石全部燒在岩漿裡"
        )

        first = record_shared_candidate(
            self.database,
            guild_id=1,
            author_id=10,
            content=(
                "專案 Minecraft：村民交易所的工具箱遭人挖掘，"
                "導致庫存綠寶石全數掉入岩漿損毀。"
            ),
            domain="game",
            subdomain="minecraft",
            memory_kind="shared_episode",
            confidence=0.95,
            importance=2,
            entity_type="item",
            entity="綠寶石",
            session_key="101:1",
            channel_id=5,
            message_id=400,
            observed_at="2026-09-09T11:53:14+00:00",
            event_signature=build_event_signature(first_message),
            shared_group_event=False,
            participant_ids=(10,),
        )

        second = record_shared_candidate(
            self.database,
            guild_id=1,
            author_id=11,
            content=(
                "專案 Minecraft：村民交易所的工具箱被拆除，"
                "導致儲存的綠寶石損失在岩漿中。"
            ),
            domain="game",
            subdomain="minecraft",
            memory_kind="shared_episode",
            confidence=0.95,
            importance=2,
            entity_type=None,
            entity=None,
            session_key="101:1",
            channel_id=5,
            message_id=401,
            observed_at="2026-09-09T11:53:51+00:00",
            event_signature=build_event_signature(second_message),
            shared_group_event=False,
            participant_ids=(11,),
        )

        self.assertEqual(first, second)

        rows = self.database.connection.execute(
            "SELECT id, status FROM guild_memories WHERE guild_id = 1 ORDER BY id"
        ).fetchall()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "active")

        evidence = self.database.connection.execute(
            """SELECT COUNT(DISTINCT user_id) AS users
               FROM guild_memory_evidence
               WHERE guild_memory_id = ?""",
            (int(rows[0]["id"]),),
        ).fetchone()

        self.assertEqual(int(evidence["users"]), 2)

    def test_candidate_same_session_different_incident_from_other_user_stays_separate(self) -> None:
        first_message = (
            "昨天 Minecraft 村民交易所的工具箱被人挖掉了，"
            "結果我們存的綠寶石全部掉進岩漿"
        )
        second_message = (
            "昨天 Minecraft 村民交易所的傳送門被人拆掉了，"
            "結果大家全部卡住回不來"
        )

        first = record_shared_candidate(
            self.database,
            guild_id=1,
            author_id=10,
            content=(
                "專案 Minecraft：村民交易所的工具箱遭人挖掘，"
                "導致庫存綠寶石全數掉入岩漿損毀。"
            ),
            domain="game",
            subdomain="minecraft",
            memory_kind="shared_episode",
            confidence=0.95,
            importance=2,
            entity_type=None,
            entity=None,
            session_key="102:1",
            channel_id=5,
            message_id=500,
            observed_at="2026-09-09T12:00:00+00:00",
            event_signature=build_event_signature(first_message),
            shared_group_event=False,
            participant_ids=(10,),
        )

        second = record_shared_candidate(
            self.database,
            guild_id=1,
            author_id=11,
            content=(
                "專案 Minecraft：村民交易所的傳送門遭人拆除，"
                "導致成員無法返回。"
            ),
            domain="game",
            subdomain="minecraft",
            memory_kind="shared_episode",
            confidence=0.95,
            importance=2,
            entity_type=None,
            entity=None,
            session_key="102:1",
            channel_id=5,
            message_id=501,
            observed_at="2026-09-09T12:00:30+00:00",
            event_signature=build_event_signature(second_message),
            shared_group_event=False,
            participant_ids=(11,),
        )

        self.assertNotEqual(first, second)

        rows = self.database.connection.execute(
            "SELECT id, status FROM guild_memories WHERE guild_id = 1 ORDER BY id"
        ).fetchall()

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["status"], "candidate")
        self.assertEqual(rows[1]["status"], "candidate")


    def test_candidate_same_session_similar_template_different_event_stays_separate(self) -> None:
        first_message = (
            "昨天 Minecraft 村民交易所的工具箱被人挖掉了，"
            "結果我們存的綠寶石全部掉進岩漿"
        )
        second_message = (
            "昨天 Minecraft 地獄交通站的補給箱被人挖掉了，"
            "結果我們存的金錠全部掉進岩漿"
        )

        first = record_shared_candidate(
            self.database,
            guild_id=1,
            author_id=10,
            content=(
                "專案 Minecraft：村民交易所的工具箱遭人挖掘，"
                "導致庫存綠寶石全數掉入岩漿損毀。"
            ),
            domain="game",
            subdomain="minecraft",
            memory_kind="shared_episode",
            confidence=0.95,
            importance=2,
            entity_type="item",
            entity="綠寶石",
            session_key="103:1",
            channel_id=5,
            message_id=600,
            observed_at="2026-09-09T11:53:14+00:00",
            event_signature=build_event_signature(first_message),
            shared_group_event=False,
            participant_ids=(10,),
        )

        second = record_shared_candidate(
            self.database,
            guild_id=1,
            author_id=11,
            content=(
                "專案 Minecraft：地獄交通站的補給箱遭人挖掘，"
                "導致庫存金錠全數掉入岩漿損毀。"
            ),
            domain="game",
            subdomain="minecraft",
            memory_kind="shared_episode",
            confidence=0.95,
            importance=2,
            entity_type="item",
            entity="金錠",
            session_key="103:1",
            channel_id=5,
            message_id=601,
            observed_at="2026-09-09T11:53:51+00:00",
            event_signature=build_event_signature(second_message),
            shared_group_event=False,
            participant_ids=(11,),
        )

        self.assertNotEqual(first, second)

        rows = self.database.connection.execute(
            "SELECT id, status FROM guild_memories WHERE guild_id = 1 ORDER BY id"
        ).fetchall()

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["status"], "candidate")
        self.assertEqual(rows[1]["status"], "candidate")


if __name__ == "__main__":
    unittest.main()
