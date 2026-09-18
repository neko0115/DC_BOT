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


class SharedMemoryEntityHierarchyLiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = AgentDatabase(Path(self.tempdir.name) / "assistant.sqlite3")
        ensure_shared_memory_schema(self.database)

    def tearDown(self) -> None:
        self.database.close()
        self.tempdir.cleanup()

    def _record(
        self,
        *,
        author_id: int,
        content: str,
        entity: str,
        message_id: int,
        event_signature: str | None = None,
    ) -> int | None:
        return record_shared_candidate(
            self.database,
            guild_id=1,
            author_id=author_id,
            content=content,
            domain="game",
            subdomain="minecraft",
            memory_kind="shared_episode",
            confidence=0.95,
            importance=2,
            entity_type="設施",
            entity=entity,
            session_key="synthetic-session:1",
            channel_id=5,
            message_id=message_id,
            observed_at="2026-09-08T04:58:06+00:00",
            event_signature=event_signature,
            shared_group_event=True,
            participant_ids=(author_id,),
        )

    def test_live_parent_child_facility_entities_reconcile_same_netherit_transit_event(self) -> None:
        first = self._record(
            author_id=10,
            content="Minecraft 地獄交通站的藍冰船道曾因苦力怕爆炸而斷裂，導致多人無法返回基地。",
            entity="地獄交通站",
            message_id=500,
        )
        second = self._record(
            author_id=11,
            content="專案 Minecraft：地獄交通站的藍冰船道曾被苦力怕炸毀，導致基地交通中斷。",
            entity="地獄交通站藍冰船道",
            message_id=501,
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

    def test_parent_child_facility_entities_do_not_merge_different_incidents(self) -> None:
        first = self._record(
            author_id=10,
            content="Minecraft 地獄交通站的藍冰船道曾因苦力怕爆炸而斷裂，導致多人無法返回基地。",
            entity="地獄交通站",
            message_id=600,
        )
        second = self._record(
            author_id=11,
            content="Minecraft 地獄交通站藍冰船道旁的補給箱被偷走，導致煙火火箭補給中斷。",
            entity="地獄交通站藍冰船道",
            message_id=601,
        )

        self.assertNotEqual(first, second)
        rows = self.database.connection.execute(
            "SELECT id, status FROM guild_memories WHERE guild_id = 1 ORDER BY id"
        ).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["status"] for row in rows], ["candidate", "candidate"])

    def test_parent_child_different_incidents_ignore_broad_signature_fallback(self) -> None:
        shared_signature = build_event_signature(
            "昨天 Minecraft 地獄交通站藍冰船道那次事故大家都在討論"
        )
        self.assertIsNotNone(shared_signature)

        first = self._record(
            author_id=10,
            content="Minecraft 地獄交通站的藍冰船道曾因苦力怕爆炸而斷裂，導致多人無法返回基地。",
            entity="地獄交通站",
            message_id=700,
            event_signature=shared_signature,
        )
        second = self._record(
            author_id=11,
            content="Minecraft 地獄交通站藍冰船道旁的補給箱被偷走，導致煙火火箭補給中斷。",
            entity="地獄交通站藍冰船道",
            message_id=701,
            event_signature=shared_signature,
        )

        self.assertNotEqual(first, second)
        rows = self.database.connection.execute(
            "SELECT id, status FROM guild_memories WHERE guild_id = 1 ORDER BY id"
        ).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["status"] for row in rows], ["candidate", "candidate"])


if __name__ == "__main__":
    unittest.main()
