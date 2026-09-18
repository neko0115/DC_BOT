from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from discord_ai_assistant.ai.memory_v2 import (
    MEMORY_KIND_EPISODIC,
    MEMORY_KIND_PROJECT,
    build_fts_query,
    infer_memory_key,
    memory_terms,
)
from discord_ai_assistant.storage.agent_database import AgentDatabase
from discord_ai_assistant.storage.database import Database


class MemoryV2HelperTests(unittest.TestCase):
    def test_mixed_cjk_and_latin_terms_are_deterministic(self) -> None:
        terms = memory_terms("YuuPo 火箭主傘 RTX 5070")
        self.assertIn("yuupo", terms)
        self.assertIn("rtx", terms)
        self.assertIn("5070", terms)
        self.assertIn("火箭", terms)
        self.assertIn("主傘", terms)
        self.assertIn('"yuupo"', build_fts_query("YuuPo"))

    def test_profile_style_memory_gets_stable_slot_key(self) -> None:
        key, subject = infer_memory_key("提醒", "我的筆電顯卡是 RTX 5070")
        self.assertEqual(key, "profile:筆電顯卡")
        self.assertEqual(subject, "筆電顯卡")

        custom_key, custom_subject = infer_memory_key("筆電顯卡", "RTX 5070")
        self.assertEqual(custom_key, "category:筆電顯卡")
        self.assertEqual(custom_subject, "筆電顯卡")


class MemoryV2DatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "assistant.sqlite3"
        self.database = AgentDatabase(self.path)

    def tearDown(self) -> None:
        self.database.close()
        self.tempdir.cleanup()

    def test_fresh_database_includes_social_memory_columns(self) -> None:
        columns = {
            str(row[1])
            for row in self.database.connection.execute("PRAGMA table_info(user_memories)").fetchall()
        }
        self.assertTrue(
            {
                "domain",
                "subdomain",
                "entity_type",
                "entity",
                "retention",
                "expires_at",
                "reinforcement_count",
                "last_reinforced",
                "socially_referenceable",
            }.issubset(columns)
        )

    def test_query_aware_search_prefers_old_relevant_memory_over_recent_noise(self) -> None:
        relevant = self.database.add_user_memory(1, 2, "專案", "我目前在做 YuuPo MVA 分析")
        for index in range(30):
            self.database.add_user_memory(1, 2, "提醒", f"一般無關備忘 {index}")

        matches = self.database.search_user_memories(1, 2, "YuuPo 的 MVA 現在做到哪", limit=5)

        self.assertTrue(matches)
        self.assertEqual(matches[0].id, relevant.id)
        self.assertIn("YuuPo", matches[0].content)

    def test_chinese_topic_retrieval_finds_rocket_episode(self) -> None:
        episode = self.database.add_user_memory(1, 2, "事件", "火箭回收段主傘曾經打到尾翼，之後才重新張開")
        self.database.add_user_memory(1, 2, "偏好", "我喜歡爵士樂")
        self.database.add_user_memory(1, 2, "習慣", "我週末通常會整理房間")

        matches = self.database.search_user_memories(1, 2, "之前火箭主傘發生什麼事", limit=3)

        self.assertEqual(matches[0].id, episode.id)
        row = self.database.connection.execute(
            "SELECT memory_kind FROM user_memories WHERE id = ?", (episode.id,)
        ).fetchone()
        self.assertEqual(row["memory_kind"], MEMORY_KIND_EPISODIC)

    def test_mutable_profile_slot_supersedes_old_value(self) -> None:
        old = self.database.add_user_memory(1, 2, "筆電顯卡", "RTX 3050")
        new = self.database.add_user_memory(1, 2, "筆電顯卡", "RTX 5070")

        visible = self.database.list_user_memories(1, 2)
        self.assertEqual([(memory.id, memory.content) for memory in visible], [(new.id, "RTX 5070")])

        old_row = self.database.connection.execute(
            "SELECT status, superseded_by FROM user_memories WHERE id = ?", (old.id,)
        ).fetchone()
        self.assertEqual(old_row["status"], "superseded")
        self.assertEqual(old_row["superseded_by"], new.id)

        matches = self.database.search_user_memories(1, 2, "我的筆電顯卡", limit=5)
        self.assertEqual([match.content for match in matches if match.subject == "筆電顯卡"], ["RTX 5070"])

    def test_exact_duplicate_confirms_instead_of_creating_duplicate(self) -> None:
        first = self.database.add_user_memory(1, 2, "偏好", "我喜歡爵士樂")
        second = self.database.add_user_memory(1, 2, "偏好", "我喜歡爵士樂")

        self.assertEqual(first.id, second.id)
        count = self.database.connection.execute(
            "SELECT COUNT(*) FROM user_memories WHERE guild_id = 1 AND user_id = 2"
        ).fetchone()[0]
        self.assertEqual(count, 1)
        confirmed = self.database.connection.execute(
            "SELECT last_confirmed FROM user_memories WHERE id = ?", (first.id,)
        ).fetchone()[0]
        self.assertIsNotNone(confirmed)

    def test_project_memory_is_typed_and_project_scoped(self) -> None:
        memory = self.database.add_user_memory(1, 2, "專案", "我目前在做 YuuPo MVA 分析")
        row = self.database.connection.execute(
            "SELECT memory_kind, project FROM user_memories WHERE id = ?", (memory.id,)
        ).fetchone()

        self.assertEqual(row["memory_kind"], MEMORY_KIND_PROJECT)
        self.assertEqual(row["project"], "YuuPo")

    def test_memory_context_uses_query_ranking_and_marks_only_selected_memories_used(self) -> None:
        rocket = self.database.add_user_memory(1, 2, "事件", "火箭回收段主傘曾經打到尾翼")
        music = self.database.add_user_memory(1, 2, "偏好", "我喜歡爵士樂")

        context = self.database.user_memory_context(1, 2, query="火箭主傘")

        self.assertIn("火箭回收段", context)
        rows = self.database.connection.execute(
            "SELECT id, use_count FROM user_memories WHERE id IN (?, ?) ORDER BY id",
            (rocket.id, music.id),
        ).fetchall()
        counts = {int(row["id"]): int(row["use_count"]) for row in rows}
        self.assertEqual(counts[rocket.id], 1)
        # With only two memories the fallback memory may still fit the bounded context;
        # the relevant one must at least be ranked first and marked used.
        matches = self.database.search_user_memories(1, 2, "火箭主傘", limit=2)
        self.assertEqual(matches[0].id, rocket.id)

    def test_legacy_database_is_migrated_in_place(self) -> None:
        self.database.close()
        legacy_path = Path(self.tempdir.name) / "legacy.sqlite3"
        legacy = Database(legacy_path)
        memory = legacy.add_user_memory(1, 2, "專案", "我目前在做 YuuPo MVA 分析")
        ambiguous = legacy.add_user_memory(1, 2, "提醒", "一般備忘內容")
        legacy.close()

        migrated = AgentDatabase(legacy_path)
        try:
            columns = {
                str(row[1])
                for row in migrated.connection.execute("PRAGMA table_info(user_memories)").fetchall()
            }
            self.assertTrue(
                {
                    "memory_kind",
                    "subject",
                    "project",
                    "memory_key",
                    "confidence",
                    "status",
                    "superseded_by",
                    "last_confirmed",
                    "domain",
                    "subdomain",
                    "entity_type",
                    "entity",
                    "retention",
                    "expires_at",
                    "reinforcement_count",
                    "last_reinforced",
                    "socially_referenceable",
                }.issubset(columns)
            )
            row = migrated.connection.execute(
                "SELECT memory_kind, project, status, last_confirmed, domain, subdomain "
                "FROM user_memories WHERE id = ?",
                (memory.id,),
            ).fetchone()
            self.assertEqual(row["memory_kind"], MEMORY_KIND_PROJECT)
            self.assertEqual(row["project"], "YuuPo")
            self.assertEqual(row["status"], "active")
            self.assertIsNotNone(row["last_confirmed"])
            self.assertEqual(row["domain"], "project")
            self.assertEqual(row["subdomain"], "YuuPo")

            ambiguous_row = migrated.connection.execute(
                "SELECT domain, subdomain FROM user_memories WHERE id = ?",
                (ambiguous.id,),
            ).fetchone()
            self.assertIsNone(ambiguous_row["domain"])
            self.assertIsNone(ambiguous_row["subdomain"])
        finally:
            migrated.close()

    def test_fts_index_is_optional_but_consistent_when_available(self) -> None:
        memory = self.database.add_user_memory(1, 2, "專案", "我目前在做 YuuPo MVA 分析")
        if not self.database._memory_fts_enabled:
            self.skipTest("Python sqlite3 was built without FTS5")

        row = self.database.connection.execute(
            "SELECT memory_id FROM user_memories_fts WHERE memory_id = ?", (memory.id,)
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(int(row["memory_id"]), memory.id)


if __name__ == "__main__":
    unittest.main()
