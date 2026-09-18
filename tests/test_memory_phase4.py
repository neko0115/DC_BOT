from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from discord_ai_assistant.ai.memory_phase2 import PassiveMemoryV2Draft
from discord_ai_assistant.ai.memory_phase4 import (
    HIGH_IDLE_SECONDS,
    HIGH_MIN_INTERVAL_SECONDS,
    MEMORY_URGENCY_HIGH,
    MEMORY_URGENCY_NORMAL,
    NORMAL_IDLE_SECONDS,
    NORMAL_MIN_INTERVAL_SECONDS,
    MemoryObservation,
    batch_flush_policy,
    list_open_memory_conflicts,
    memory_provenance_rows,
    memory_urgency,
)
from discord_ai_assistant.memory_v2_runtime import store_passive_memory_v2_drafts
from discord_ai_assistant.storage.agent_database import AgentDatabase


class MemoryPhase4PolicyTests(unittest.TestCase):
    def test_high_impact_updates_use_shorter_flush_policy(self) -> None:
        self.assertEqual(memory_urgency("YuuPo 下一步要處理 Apollo reference"), MEMORY_URGENCY_HIGH)
        self.assertEqual(memory_urgency("YuuPo 目前 blocker 是等待教授 review"), MEMORY_URGENCY_HIGH)
        self.assertEqual(memory_urgency("最近在整理 YuuPo 文件內容"), MEMORY_URGENCY_NORMAL)

        urgent = MemoryObservation(1.0, "YuuPo 下一步要處理 Apollo", 10, 100, None, MEMORY_URGENCY_HIGH)
        normal = MemoryObservation(2.0, "YuuPo 文件內容有更新", 10, 101, None, MEMORY_URGENCY_NORMAL)
        self.assertEqual(batch_flush_policy([normal]), (float(NORMAL_IDLE_SECONDS), float(NORMAL_MIN_INTERVAL_SECONDS)))
        self.assertEqual(batch_flush_policy([normal, urgent]), (float(HIGH_IDLE_SECONDS), float(HIGH_MIN_INTERVAL_SECONDS)))


class MemoryPhase4StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = AgentDatabase(Path(self.tempdir.name) / "assistant.sqlite3")

    def tearDown(self) -> None:
        self.database.close()
        self.tempdir.cleanup()

    def _observation(self, message_id: int, content: str = "我的筆電顯卡現在是 RTX 5070") -> MemoryObservation:
        return MemoryObservation(
            monotonic_at=float(message_id),
            content=content,
            channel_id=123,
            message_id=message_id,
            created_at=f"2026-09-05T17:00:{message_id % 60:02d}+00:00",
            urgency=memory_urgency(content),
        )

    def test_nonproject_passive_memory_preserves_confidence_importance_and_provenance(self) -> None:
        draft = PassiveMemoryV2Draft("提醒", "我的筆電顯卡是 RTX 5070", 0.87, 3)
        stored = store_passive_memory_v2_drafts(
            self.database,
            1,
            2,
            [draft],
            observations=[self._observation(100), self._observation(101)],
        )
        self.assertEqual(stored, 1)

        row = self.database.connection.execute(
            "SELECT id, confidence, importance FROM user_memories WHERE guild_id = 1 AND user_id = 2"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertAlmostEqual(float(row["confidence"]), 0.87)
        self.assertEqual(int(row["importance"]), 3)

        provenance = memory_provenance_rows(self.database, 1, 2, int(row["id"]))
        self.assertEqual({int(item["message_id"]) for item in provenance}, {100, 101})

    def test_reconfirmation_adds_new_provenance_without_duplicate_memory(self) -> None:
        draft = PassiveMemoryV2Draft("提醒", "我的筆電顯卡是 RTX 5070", 0.9, 2)
        store_passive_memory_v2_drafts(
            self.database,
            1,
            2,
            [draft],
            observations=[self._observation(110)],
        )
        second = store_passive_memory_v2_drafts(
            self.database,
            1,
            2,
            [draft],
            observations=[self._observation(120)],
        )
        self.assertEqual(second, 0)
        rows = self.database.connection.execute(
            "SELECT id FROM user_memories WHERE guild_id = 1 AND user_id = 2"
        ).fetchall()
        self.assertEqual(len(rows), 1)
        provenance = memory_provenance_rows(self.database, 1, 2, int(rows[0]["id"]))
        self.assertEqual({int(item["message_id"]) for item in provenance}, {110, 120})

    def test_overlapping_project_facts_are_flagged_not_overwritten(self) -> None:
        first = PassiveMemoryV2Draft(
            "專案",
            "專案 DC_BOT：目前 Gemini 搜尋模型設定使用 gemini-2.5-flash",
            0.95,
            3,
            "DC_BOT",
            "fact",
        )
        second = PassiveMemoryV2Draft(
            "專案",
            "專案 DC_BOT：目前 Gemini 搜尋模型設定使用 gemini-3.6-flash",
            0.96,
            3,
            "DC_BOT",
            "fact",
        )
        store_passive_memory_v2_drafts(self.database, 1, 2, [first])
        store_passive_memory_v2_drafts(self.database, 1, 2, [second])

        active = self.database.connection.execute(
            "SELECT content FROM user_memories WHERE guild_id = 1 AND user_id = 2 AND project = 'DC_BOT' AND status = 'active' AND category = '自動專案'"
        ).fetchall()
        self.assertEqual(len(active), 2)
        conflicts = list_open_memory_conflicts(self.database, 1, 2)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["project"], "DC_BOT")
        self.assertGreaterEqual(float(conflicts[0]["similarity"]), 0.55)

    def test_unrelated_project_facts_do_not_create_conflict(self) -> None:
        drafts = [
            PassiveMemoryV2Draft("專案", "專案 YuuPo：資料分析使用 FFT-bin MVA", 0.95, 3, "YuuPo", "fact"),
            PassiveMemoryV2Draft("專案", "專案 YuuPo：部署主機使用 Linux", 0.95, 2, "YuuPo", "fact"),
        ]
        store_passive_memory_v2_drafts(self.database, 1, 2, drafts)
        self.assertEqual(list_open_memory_conflicts(self.database, 1, 2), [])

    def test_provenance_is_scoped_to_memory_owner(self) -> None:
        draft = PassiveMemoryV2Draft("提醒", "我的筆電顯卡是 RTX 5070", 0.9, 2)
        store_passive_memory_v2_drafts(
            self.database,
            1,
            2,
            [draft],
            observations=[self._observation(130)],
        )
        memory_id = int(
            self.database.connection.execute("SELECT id FROM user_memories WHERE guild_id = 1 AND user_id = 2").fetchone()["id"]
        )
        self.assertTrue(memory_provenance_rows(self.database, 1, 2, memory_id))
        self.assertEqual(memory_provenance_rows(self.database, 1, 999, memory_id), [])


if __name__ == "__main__":
    unittest.main()
