from __future__ import annotations

import importlib
import importlib.util
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from discord_ai_assistant.ai.memory_phase2 import PassiveMemoryV2Draft
from discord_ai_assistant.ai.memory_phase4 import (
    MemoryObservation,
    ensure_memory_phase4_schema,
    record_memory_provenance,
)
from discord_ai_assistant.memory_v2_runtime import store_passive_memory_v2_drafts
from discord_ai_assistant.storage.agent_database import AgentDatabase


class MemoryRetentionPolicyTests(unittest.TestCase):
    def test_retention_module_exposes_exact_windows(self) -> None:
        spec = importlib.util.find_spec("discord_ai_assistant.ai.memory_retention")
        self.assertIsNotNone(spec, "memory_retention module must exist")
        if spec is None:
            return
        module = importlib.import_module("discord_ai_assistant.ai.memory_retention")

        now = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)
        self.assertEqual(module.RETENTION_SHORT, "short")
        self.assertEqual(module.RETENTION_MEDIUM, "medium")
        self.assertEqual(module.RETENTION_LONG, "long")
        self.assertEqual(module.RETENTION_SHARED, "shared")
        self.assertIsNone(module.expires_at_for_retention(module.RETENTION_LONG, now=now))
        self.assertEqual(
            module.expires_at_for_retention(module.RETENTION_SHORT, now=now),
            "2026-09-20T10:00:00+00:00",
        )
        self.assertEqual(
            module.expires_at_for_retention(module.RETENTION_MEDIUM, now=now),
            "2026-12-05T10:00:00+00:00",
        )
        self.assertEqual(
            module.expires_at_for_retention(module.RETENTION_SHARED, now=now),
            "2026-10-06T10:00:00+00:00",
        )


class MemoryRetentionDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = AgentDatabase(Path(self.tempdir.name) / "assistant.sqlite3")
        ensure_memory_phase4_schema(self.database)

    def tearDown(self) -> None:
        self.database.close()
        self.tempdir.cleanup()

    def test_due_memory_is_marked_expired_without_deletion(self) -> None:
        self.assertTrue(
            hasattr(self.database, "expire_due_user_memories"),
            "AgentDatabase must expose expire_due_user_memories",
        )
        if not hasattr(self.database, "expire_due_user_memories"):
            return

        memory, created = self.database._upsert_user_memory(
            1,
            2,
            "自動興趣",
            "最近主要在玩 Minecraft 生存模式",
            source="passive",
            importance=2,
            memory_kind="current_interest",
            domain="game",
            subdomain="minecraft",
            retention="short",
            expires_at="2026-09-01T00:00:00+00:00",
        )
        self.assertTrue(created)

        expired = self.database.expire_due_user_memories(
            1,
            2,
            now_iso="2026-09-06T10:00:00+00:00",
        )
        self.assertEqual(expired, 1)

        row = self.database.connection.execute(
            "SELECT status FROM user_memories WHERE id = ?",
            (memory.id,),
        ).fetchone()
        self.assertEqual(row["status"], "expired")
        self.assertEqual(
            self.database.connection.execute(
                "SELECT COUNT(*) FROM user_memories WHERE id = ?",
                (memory.id,),
            ).fetchone()[0],
            1,
        )
        self.assertEqual(
            self.database.search_user_memories(1, 2, "Minecraft 生存", limit=5),
            [],
        )

    def test_reinforce_personal_memory_is_session_aware_and_refreshes_expiry(self) -> None:
        module = importlib.import_module("discord_ai_assistant.ai.memory_retention")
        reinforce = getattr(module, "reinforce_personal_memory", None)
        self.assertTrue(callable(reinforce), "memory_retention must expose reinforce_personal_memory")
        if not callable(reinforce):
            return

        cases = (
            ("short", "2026-09-24T10:00:00+00:00", 20, "最近都在玩 Minecraft 生存模式", "minecraft"),
            ("medium", "2026-12-09T10:00:00+00:00", 21, "最近都在玩原神 深境螺旋", "genshin"),
        )
        for retention, expected_expiry, user_id, content, subdomain in cases:
            with self.subTest(retention=retention):
                memory, created = self.database._upsert_user_memory(
                    1,
                    user_id,
                    "自動興趣",
                    content,
                    source="passive",
                    importance=2,
                    confidence=0.9,
                    memory_kind="current_main",
                    domain="game",
                    subdomain=subdomain,
                    retention=retention,
                    expires_at="2026-09-07T00:00:00+00:00",
                )
                self.assertTrue(created)
                first = MemoryObservation(
                    1.0,
                    content,
                    5,
                    user_id * 10,
                    "2026-09-06T10:00:00+00:00",
                    session_key="1:1",
                    domain="game",
                    subdomain=subdomain,
                )
                record_memory_provenance(self.database, 1, user_id, memory.id, [first])

                baseline_now = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)
                self.assertFalse(reinforce(self.database, memory.id, session_key="1:1", now=baseline_now))
                self.assertFalse(reinforce(self.database, memory.id, session_key=None, now=baseline_now))

                second = MemoryObservation(
                    2.0,
                    content,
                    5,
                    user_id * 10 + 1,
                    "2026-09-10T10:00:00+00:00",
                    session_key="1:2",
                    domain="game",
                    subdomain=subdomain,
                )
                record_memory_provenance(self.database, 1, user_id, memory.id, [second])
                reinforced_now = datetime(2026, 9, 10, 10, 0, tzinfo=timezone.utc)
                self.assertTrue(reinforce(self.database, memory.id, session_key="1:2", now=reinforced_now))
                self.assertFalse(reinforce(self.database, memory.id, session_key="1:2", now=reinforced_now))

                row = self.database.connection.execute(
                    """SELECT memory_kind, reinforcement_count, last_reinforced, expires_at
                       FROM user_memories WHERE id = ?""",
                    (memory.id,),
                ).fetchone()
                self.assertEqual(row["memory_kind"], "current_main")
                self.assertEqual(int(row["reinforcement_count"]), 1)
                self.assertEqual(row["last_reinforced"], "2026-09-10T10:00:00+00:00")
                self.assertEqual(row["expires_at"], expected_expiry)

    def test_passive_storage_reinforces_only_on_a_new_nonnull_session(self) -> None:
        module = importlib.import_module("discord_ai_assistant.ai.memory_retention")
        reinforce = getattr(module, "reinforce_personal_memory", None)
        self.assertTrue(callable(reinforce), "personal reinforcement helper must exist before runtime wiring")
        if not callable(reinforce):
            return

        draft = PassiveMemoryV2Draft(
            category="興趣",
            content="最近都在玩 Minecraft 生存模式",
            confidence=0.91,
            importance=2,
            domain="game",
            subdomain="minecraft",
            memory_kind="current_main",
            retention="medium",
        )

        def observation(message_id: int, session_key: str) -> MemoryObservation:
            return MemoryObservation(
                float(message_id),
                draft.content,
                5,
                message_id,
                "2026-09-10T10:00:00+00:00",
                session_key=session_key,
                domain="game",
                subdomain="minecraft",
            )

        store_passive_memory_v2_drafts(
            self.database,
            1,
            30,
            [draft],
            observations=[observation(300, "5:1")],
        )
        store_passive_memory_v2_drafts(
            self.database,
            1,
            30,
            [draft],
            observations=[observation(301, "5:1")],
        )
        row = self.database.connection.execute(
            "SELECT id, reinforcement_count FROM user_memories WHERE guild_id = 1 AND user_id = 30 AND status = 'active'"
        ).fetchone()
        self.assertEqual(int(row["reinforcement_count"]), 0)

        store_passive_memory_v2_drafts(
            self.database,
            1,
            30,
            [draft],
            observations=[observation(302, "5:2")],
        )
        row = self.database.connection.execute(
            "SELECT memory_kind, reinforcement_count FROM user_memories WHERE id = ?",
            (int(row["id"]),),
        ).fetchone()
        self.assertEqual(row["memory_kind"], "current_main")
        self.assertEqual(int(row["reinforcement_count"]), 1)


if __name__ == "__main__":
    unittest.main()
