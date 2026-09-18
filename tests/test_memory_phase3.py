from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from discord_ai_assistant.ai.memory_phase2 import PassiveMemoryV2Draft, parse_passive_memory_v2_drafts
from discord_ai_assistant.ai.memory_project_state import (
    build_project_summary,
    list_user_projects,
    project_memory_key,
    project_snapshot_text,
    project_summary_key,
)
from discord_ai_assistant.memory_v2_runtime import store_passive_memory_v2_drafts
from discord_ai_assistant.storage.agent_database import AgentDatabase


class MemoryPhase3ParserTests(unittest.TestCase):
    def test_parser_keeps_project_and_role_metadata(self) -> None:
        drafts = parse_passive_memory_v2_drafts(
            '{"memories":['
            '{"category":"專案","content":"專案 YuuPo：目前 production ellipticity 已改為 signed b/a",'
            '"confidence":0.96,"importance":3,"project":"YuuPo","role":"status"},'
            '{"category":"專案","content":"專案 YuuPo：下一步驗證 Apollo ellipticity",'
            '"confidence":0.91,"importance":2,"project":"YuuPo","role":"next_step"},'
            '{"category":"決策","content":"專案 YuuPo：決定 production authoritative definition 使用 signed b/a",'
            '"confidence":0.95,"importance":3,"project":"YuuPo","role":"decision"}'
            ']}'
        )

        self.assertEqual(len(drafts), 3)
        self.assertEqual((drafts[0].project, drafts[0].role), ("YuuPo", "status"))
        self.assertEqual((drafts[1].project, drafts[1].role), ("YuuPo", "next_step"))
        self.assertEqual((drafts[2].project, drafts[2].role), ("YuuPo", "decision"))

    def test_mutable_project_role_requires_project_scope(self) -> None:
        drafts = parse_passive_memory_v2_drafts(
            '{"memories":[{"category":"專案","content":"目前已完成驗證",'
            '"confidence":0.95,"importance":3,"project":null,"role":"status"}]}'
        )
        self.assertEqual(drafts, [])


class MemoryPhase3ProjectStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = AgentDatabase(Path(self.tempdir.name) / "assistant.sqlite3")

    def tearDown(self) -> None:
        self.database.close()
        self.tempdir.cleanup()

    def _store(self, *drafts: PassiveMemoryV2Draft) -> int:
        return store_passive_memory_v2_drafts(self.database, 1, 2, list(drafts))

    def test_new_project_status_supersedes_old_status(self) -> None:
        self._store(
            PassiveMemoryV2Draft(
                "專案",
                "專案 YuuPo：目前還在比對 IDL FFT bins",
                0.95,
                3,
                "YuuPo",
                "status",
            )
        )
        old_row = self.database.connection.execute(
            "SELECT id FROM user_memories WHERE memory_key = ? AND status = 'active'",
            (project_memory_key("YuuPo", "status"),),
        ).fetchone()
        self.assertIsNotNone(old_row)

        self._store(
            PassiveMemoryV2Draft(
                "專案",
                "專案 YuuPo：目前 FFT-bin validation 已完成並進入 release review",
                0.97,
                3,
                "YuuPo",
                "status",
            )
        )

        rows = self.database.connection.execute(
            "SELECT id, content, status, superseded_by FROM user_memories WHERE memory_key = ? ORDER BY id",
            (project_memory_key("YuuPo", "status"),),
        ).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["status"], "superseded")
        self.assertEqual(rows[1]["status"], "active")
        self.assertEqual(rows[0]["superseded_by"], rows[1]["id"])
        self.assertIn("release review", rows[1]["content"])

    def test_blocker_and_next_step_are_independent_mutable_slots(self) -> None:
        self._store(
            PassiveMemoryV2Draft("專案", "專案 YuuPo：目前 blocker 是 Apollo reference unavailable", 0.95, 3, "YuuPo", "blocker"),
            PassiveMemoryV2Draft("專案", "專案 YuuPo：下一步補 Poynting external validation", 0.93, 2, "YuuPo", "next_step"),
        )
        self._store(
            PassiveMemoryV2Draft("專案", "專案 YuuPo：目前 blocker 改為等待教授 review", 0.96, 3, "YuuPo", "blocker"),
        )

        blocker = self.database.connection.execute(
            "SELECT content FROM user_memories WHERE memory_key = ? AND status = 'active'",
            (project_memory_key("YuuPo", "blocker"),),
        ).fetchone()
        next_step = self.database.connection.execute(
            "SELECT content FROM user_memories WHERE memory_key = ? AND status = 'active'",
            (project_memory_key("YuuPo", "next_step"),),
        ).fetchone()
        self.assertIn("教授 review", blocker["content"])
        self.assertIn("Poynting", next_step["content"])

    def test_decisions_and_events_preserve_history(self) -> None:
        self._store(
            PassiveMemoryV2Draft("決策", "專案 YuuPo：決定 ellipticity authoritative definition 使用 signed b/a", 0.98, 3, "YuuPo", "decision"),
            PassiveMemoryV2Draft("決策", "專案 YuuPo：決定 production 使用 symmetric Hann", 0.96, 3, "YuuPo", "decision"),
            PassiveMemoryV2Draft("事件", "專案 YuuPo：No8 wave-angle validation 通過", 0.96, 2, "YuuPo", "milestone"),
        )

        rows = self.database.connection.execute(
            "SELECT category, content, status FROM user_memories WHERE project = ? AND status = 'active'",
            ("YuuPo",),
        ).fetchall()
        decisions = [row for row in rows if "決策" in row["category"]]
        events = [row for row in rows if "里程碑" in row["category"]]
        self.assertEqual(len(decisions), 2)
        self.assertEqual(len(events), 1)

    def test_derived_summary_tracks_current_state_and_history(self) -> None:
        self._store(
            PassiveMemoryV2Draft("專案", "專案 YuuPo：目前 Phase 7 release gate 已完成", 0.98, 3, "YuuPo", "status"),
            PassiveMemoryV2Draft("專案", "專案 YuuPo：目前 blocker 是等待教授 final review", 0.96, 3, "YuuPo", "blocker"),
            PassiveMemoryV2Draft("專案", "專案 YuuPo：下一步處理 Apollo authoritative reference", 0.94, 2, "YuuPo", "next_step"),
            PassiveMemoryV2Draft("決策", "專案 YuuPo：決定 signed b/a 為 production authoritative ellipticity", 0.98, 3, "YuuPo", "decision"),
        )

        summary = build_project_summary(self.database, 1, 2, "YuuPo")
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertIn("Phase 7", summary)
        self.assertIn("教授 final review", summary)
        self.assertIn("Apollo", summary)
        self.assertIn("signed b/a", summary)

        rows = self.database.connection.execute(
            "SELECT content, status FROM user_memories WHERE memory_key = ? ORDER BY id",
            (project_summary_key("YuuPo"),),
        ).fetchall()
        self.assertEqual(len([row for row in rows if row["status"] == "active"]), 1)
        self.assertIn("專案 YuuPo 摘要", rows[-1]["content"])

    def test_project_summary_is_retrievable_by_normal_memory_search(self) -> None:
        self._store(
            PassiveMemoryV2Draft("專案", "專案 YuuPo：目前 release validation 已完成", 0.97, 3, "YuuPo", "status"),
            PassiveMemoryV2Draft("專案", "專案 YuuPo：下一步等待 final review", 0.93, 2, "YuuPo", "next_step"),
        )

        matches = self.database.search_user_memories(1, 2, "YuuPo 現在做到哪 下一步", limit=5)
        self.assertTrue(any(match.category == "專案摘要" for match in matches))
        summary_match = next(match for match in matches if match.category == "專案摘要")
        self.assertIn("release validation", summary_match.content)
        self.assertIn("final review", summary_match.content)

    def test_projects_remain_isolated(self) -> None:
        self._store(
            PassiveMemoryV2Draft("專案", "專案 YuuPo：目前在 release review", 0.95, 3, "YuuPo", "status"),
            PassiveMemoryV2Draft("專案", "專案 DC_BOT：目前在測試 Memory V2", 0.95, 3, "DC_BOT", "status"),
        )

        projects = list_user_projects(self.database, 1, 2)
        self.assertEqual(set(projects), {"YuuPo", "DC_BOT"})
        yuupo = project_snapshot_text(self.database, 1, 2, "YuuPo")
        dcbot = project_snapshot_text(self.database, 1, 2, "DC_BOT")
        self.assertIn("release review", yuupo)
        self.assertNotIn("目前在測試 Memory V2", yuupo)
        self.assertIn("目前在測試 Memory V2", dcbot)
        self.assertNotIn("release review", dcbot)


if __name__ == "__main__":
    unittest.main()
